import time
import torch
import torch.nn as nn
import torch.optim as optim
import asyncio
import random
import collections
import traceback
from config.settings import Config
from src.models.sde_model import LatentSDE
from src.models.rl_agent import TradingAgent, SACActor, SACCritic
from src.data.collector import RealTimeDataCollector
from src.strategy.reward import calculate_reward
from src.utils.visualizer import TradingVisualizer

class TradingPipeline:

    def __init__(self):
            # ... 기존 초기화 생략 ...
            # SAC 모델 초기화
            self.actor = SACActor(self.state_dim, self.action_dim, self.latent_dim, self.portfolio_dim).to(Config.DEVICE)
            self.critic1 = SACCritic(self.state_dim, self.action_dim, self.latent_dim, self.portfolio_dim).to(Config.DEVICE)
            self.critic2 = SACCritic(self.state_dim, self.action_dim, self.latent_dim, self.portfolio_dim).to(Config.DEVICE)
            self.critic1_target = SACCritic(self.state_dim, self.action_dim, self.latent_dim, self.portfolio_dim).to(Config.DEVICE)
            self.critic2_target = SACCritic(self.state_dim, self.action_dim, self.latent_dim, self.portfolio_dim).to(Config.DEVICE)
            self.critic1_target.load_state_dict(self.critic1.state_dict())
            self.critic2_target.load_state_dict(self.critic2.state_dict())

            # 자동 엔트로피 튜닝 (Alpha)
            self.target_entropy = -torch.prod(torch.Tensor([self.action_dim]).to(Config.DEVICE)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=Config.DEVICE)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
            self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=3e-4)
            
            self.gamma = 0.99
            self.tau = 0.005 # Soft update rate


    async def update_portfolio(self, p_state, action_weight, current_price, latency, is_real_execution=False):
        if p_state["asset"] > 0 and p_state["avg_entry_price"] > 0:
            loss_pct = (p_state["avg_entry_price"] - current_price) / p_state["avg_entry_price"]
            if loss_pct > Config.STOP_LOSS_THRESHOLD:
                action_weight = 0.0 # 하드 손절

        prev_total = p_state["cash"] + p_state["asset"] * current_price
        target_asset_value = prev_total * action_weight
        diff_value = target_asset_value - (p_state["asset"] * current_price)

        slippage_pct = latency * 0.001
        sim_exec_price = current_price * (1 + slippage_pct) if diff_value > 0 else current_price * (1 - slippage_pct)

        commission_cost = 0.0
        trade_type = None

        if abs(diff_value) > 1.0:
            if is_real_execution and Config.MODE == 'REAL':
                side = 'buy' if diff_value > 0 else 'sell'
                await self.data_collector.create_order(side, abs(diff_value) / current_price)

            if diff_value > 0: # Buy
                trade_type = 'buy'
                amount = diff_value / sim_exec_price
                if p_state["cash"] >= diff_value:
                    p_state["cash"] -= diff_value
                    fee = amount * Config.COMMISSION_RATE
                    commission_cost = fee * sim_exec_price
                    net_amount = amount - fee
                    total_qty = p_state["asset"] + net_amount
                    p_state["avg_entry_price"] = (p_state["asset"] * p_state["avg_entry_price"] + net_amount * sim_exec_price) / total_qty
                    p_state["asset"] += net_amount
            else: # Sell
                trade_type = 'sell'
                amount = abs(diff_value) / sim_exec_price
                if p_state["asset"] >= amount:
                    p_state["asset"] -= amount
                    proceeds = amount * sim_exec_price
                    fee = proceeds * Config.COMMISSION_RATE
                    commission_cost = fee
                    p_state["cash"] += (proceeds - fee)

        new_total = p_state["cash"] + p_state["asset"] * current_price
        reward = calculate_reward(prev_total, new_total, commission_cost + abs(slippage_pct * diff_value), volatility=0.0)
        p_state["total"] = new_total
        return reward, p_state, commission_cost, trade_type

    def train_step_rl(self):
            if len(self.replay_buffer) < self.training_batch_size: return 0.0, 0.0
            
            batch = self.replay_buffer[:self.training_batch_size]
            self.replay_buffer = self.replay_buffer[self.training_batch_size:]
            
            # 데이터 언패킹 (state, sde_out, p_vec, action, reward, next_state, next_sde, next_p)
            # *주의*: SAC는 next_state가 필요하므로 replay_buffer 저장 시 추가해야 함
            states, sde_outs, p_states, actions, rewards, next_states, next_sde_outs, next_p_states = zip(*batch)

            # 1. Critic 학습
            with torch.no_grad():
                next_actions, next_log_probs = self.actor.sample(next_states, next_sde_outs, next_p_states)
                q1_target = self.critic1_target(next_states, next_sde_outs, next_p_states, next_actions)
                q2_target = self.critic2_target(next_states, next_sde_outs, next_p_states, next_actions)
                min_q_target = torch.min(q1_target, q2_target) - self.log_alpha.exp() * next_log_probs
                y_target = rewards + self.gamma * min_q_target

            q1 = self.critic1(states, sde_outs, p_states, actions)
            q2 = self.critic2(states, sde_outs, p_states, actions)
            critic_loss = F.mse_loss(q1, y_target) + F.mse_loss(q2, y_target)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # 2. Actor 학습
            new_actions, log_probs = self.actor.sample(states, sde_outs, p_states)
            q1_new = self.critic1(states, sde_outs, p_states, new_actions)
            q2_new = self.critic2(states, sde_outs, p_states, new_actions)
            actor_loss = (self.log_alpha.exp() * log_probs - torch.min(q1_new, q2_new)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 3. Alpha 학습 (엔트로피 자동 조정)
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # 4. Soft Target Update
            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

            return actor_loss.item(), critic_loss.item()



    def should_replace_model(self):
        """
        신규 모델(New)이 기존 모델(Old)보다 5% 이상 성능이 좋을 때만 교체 여부를 반환합니다.
        최소 100번 이상의 데이터 포인트가 쌓였을 때 판단합니다.
        """
        # 비교를 위한 최소 데이터가 쌓이지 않았다면 False
        if len(self.performance_history["new"]) < 100: 
            return False
            
        # 최근 100단계의 누적 보상 비교
        recent_old = sum(self.performance_history["old"][-100:])
        recent_new = sum(self.performance_history["new"][-100:])
        
        # 신규 모델이 기존 모델보다 5% 이상 우수하고, 수익이 양수일 때 교체
        return recent_new > recent_old * 1.05 and recent_new > 0
    
    def train_sde_step(self, x, dt, y):
        self.sde_optimizer.zero_grad()
        dt_scalar = dt.mean().item() if isinstance(dt, torch.Tensor) else dt
        _, pred_decoded = self.sde_predictor(x, dt_scalar, decode=True)
        loss = self.sde_criterion(pred_decoded, y)
        loss.backward(); self.sde_optimizer.step()
        return loss.item()

    async def sde_update_loop(self):
        print("SDE Async Update Loop Started...")
        while self.running:
            buffer_data = self.data_collector.get_buffer()
            if buffer_data is not None and len(buffer_data) > 64:
                idx = torch.randint(0, len(buffer_data)-1, (32,))
                x, y = buffer_data[idx, :Config.STATE_DIM].to(Config.DEVICE), buffer_data[idx+1, :Config.STATE_DIM].to(Config.DEVICE)
                dt = buffer_data[idx, Config.STATE_DIM-1]
                loss = self.train_sde_step(x, dt, y)
                self.sde_loss_history.append(loss)
            await asyncio.sleep(1.0)


    async def run_async(self):
        self.running = True
        # 데이터 수집기 연결 시작
        asyncio.create_task(self.data_collector.connect())

        # --- PHASE 0: 오프라인 데이터 수집 (Data Accumulation) ---
        # 설정된 OFFLINE_DATA_SIZE만큼 데이터가 쌓일 때까지 대기합니다.
        offline_size = getattr(Config, 'OFFLINE_DATA_SIZE', 2000)
        print(f"PHASE 0: Accumulating {offline_size} points for offline training...")
        
        while len(self.data_collector.buffer) < offline_size:
            if len(self.data_collector.buffer) % 100 == 0 and len(self.data_collector.buffer) > 0:
                print(f"Buffer progress: {len(self.data_collector.buffer)}/{offline_size}")
            await asyncio.sleep(1)

        # --- PHASE 1: 오프라인 SDE 집중 학습 (World Model Training) ---
        # 수집된 데이터를 바탕으로 시장의 물리적 변화 법칙을 먼저 학습합니다.
        print("PHASE 1: Offline SDE Training (Learning Market Physics)...")
        buffer_data = self.data_collector.get_buffer()
        # [T, Dim] 형태의 피처 추출
        feats = buffer_data[:, :Config.STATE_DIM].to(Config.DEVICE)
        x_sde = feats[:-1]      # 현재 상태
        y_sde = feats[1:]       # 다음 상태 (타겟)
        dt_sde = feats[:-1, -1] # 지연 시간 (delta_t)
        
        sde_epochs = getattr(Config, 'OFFLINE_SDE_EPOCHS', 100)
        for epoch in range(sde_epochs):
            loss = self.train_sde_step(x_sde, dt_sde, y_sde)
            self.sde_loss_history.append(loss)
            if epoch % 20 == 0:
                print(f"SDE Offline Epoch {epoch}/{sde_epochs} | Loss: {loss:.6f}")

        # --- PHASE 2: 오프라인 RL 집중 학습 (Policy Warm-up) ---
        # 학습된 SDE의 예측치를 바탕으로 과거 데이터 상에서 에이전트를 모의 훈련시킵니다.
        print("PHASE 2: Offline RL Training (Policy Warm-up)...")
        rl_epochs = getattr(Config, 'OFFLINE_RL_EPOCHS', 50)
        
        for epoch in range(rl_epochs):
            total_reward = 0
            # 수집된 버퍼 데이터를 시뮬레이션 환경으로 간주하여 순회
            for t in range(len(feats) - 1):
                curr_state = feats[t].unsqueeze(0)
                latency = curr_state[0, -1].item()
                abs_price = buffer_data[t, -1].item() # 마지막 컬럼은 절대 가격

                with torch.no_grad():
                    # 학습된 SDE를 사용하여 미래 잠재 상태 예측
                    future_latent, _ = self.sde_predictor(curr_state, latency, decode=True)

                p_vec = torch.tensor([self.p_state_new["cash"], self.p_state_new["asset"]]).float().unsqueeze(0).to(Config.DEVICE)
                
                # 에이전트 행동 결정 (탐험 포함)
                action, log_prob, _ = self.new_model.get_action(curr_state, future_latent, p_vec, deterministic=False)
                
                # 포트폴리오 업데이트 (오프라인이므로 MOCK 환경처럼 작동)
                reward, self.p_state_new, _, _ = await self.update_portfolio(
                    self.p_state_new, action.item(), abs_price, latency, is_real_execution=False
                )
                
                # 경험 저장 및 학습
                self.replay_buffer.append((curr_state[0].cpu(), future_latent[0].cpu(), p_vec[0].cpu(), action[0].cpu(), reward, log_prob[0].cpu()))
                
                if len(self.replay_buffer) >= self.training_batch_size:
                    self.train_step_rl()
                total_reward += reward

            print(f"RL Offline Epoch {epoch}/{rl_epochs} | Avg Reward: {total_reward/len(feats):.6f}")

        # 실전 투입 전 자산 상태 리셋 (선택 사항)
        self.p_state_new = {"cash": 1000.0, "asset": 0.0, "total": 1000.0, "avg_entry_price": 0.0}
        print("PHASE 3: Transitioning to Live Trading & Async Learning...")

        # --- PHASE 3: 실시간 트레이딩 및 비동기 업데이트 루프 ---
        # SDE를 계속 학습시키는 비동기 루프 시작
        asyncio.create_task(self.sde_update_loop())
        cum_comm_new = 0.0

        try:
            while self.running:
                try:
                    market_data = self.data_collector.get_latest_data()
                    abs_price = market_data[-1].item()
                    state_vec = market_data[:Config.STATE_DIM]
                    latency = state_vec[-1].item()
                    state = state_vec.unsqueeze(0).to(Config.DEVICE)

                    # 1. SDE 예측 (비동기 루프에서 계속 업데이트 중인 모델)
                    with torch.no_grad():
                        future_latent, _ = self.sde_predictor(state, latency, decode=True)

                    p_vec_old = torch.tensor([self.p_state_old["cash"], self.p_state_old["asset"]]).float().unsqueeze(0).to(Config.DEVICE)
                    p_vec_new = torch.tensor([self.p_state_new["cash"], self.p_state_new["asset"]]).float().unsqueeze(0).to(Config.DEVICE)

                    # 2. 기존 모델(Old)과 신규 모델(New)의 결정 수행
                    with torch.no_grad():
                        action_old, _, _ = self.old_model.get_action(state, future_latent, p_vec_old, deterministic=True)
                    
                    reward_old, self.p_state_old, _, _ = await self.update_portfolio(
                        self.p_state_old, action_old.item(), abs_price, latency, is_real_execution=True
                    )

                    action_new, log_prob_new, _ = self.new_model.get_action(state, future_latent, p_vec_new, deterministic=False)
                    
                    reward_new, self.p_state_new, comm_new, t_type = await self.update_portfolio(
                        self.p_state_new, action_new.item(), abs_price, latency, is_real_execution=False
                    )

                    # 3. 실시간 학습 및 지표 관리
                    cum_comm_new += comm_new
                    gross_wealth_new = self.p_state_new["total"] + cum_comm_new
                    
                    self.replay_buffer.append((state[0].cpu(), future_latent[0].cpu(), p_vec_new[0].cpu(), action_new[0].cpu(), reward_new, log_prob_new[0].cpu()))
                    a_loss, c_loss = self.train_step_rl()

                    self.performance_history["old"].append(reward_old)
                    self.performance_history["new"].append(reward_new)

                    # 4. 모델 교체 로직
                    swap_event = False
                    if self.should_replace_model():
                        print(">>> REPLACING PRODUCTION MODEL <<<")
                        self.old_model.load_state_dict(self.new_model.state_dict())
                        self.p_state_old = self.p_state_new.copy()
                        self.performance_history["old"], self.performance_history["new"] = [], []
                        swap_event = True

                    # 5. 시각화 업데이트
                    sde_loss_viz = self.sde_loss_history[-1] if self.sde_loss_history else 0.0
                    self.visualizer.update({
                        "wealth_net_old": self.p_state_old["total"],
                        "wealth_net_new": self.p_state_new["total"],
                        "wealth_gross_new": gross_wealth_new,
                        "p_state_new": self.p_state_new,
                        "action_new": action_new.item(),
                        "price": abs_price,
                        "commission_step": comm_new,
                        "trade_type": t_type,
                        "sde_loss": sde_loss_viz,
                        "actor_loss": a_loss,
                        "critic_loss": c_loss,  
                        "reward": reward_new,
                        "swap_event": swap_event
                    })

                    # 주기적인 대시보드 출력
                    if len(self.performance_history["new"]) % 100 == 0:
                        self.visualizer.plot_all()
                        print(f"Step: {len(self.performance_history['new'])} | Net: {self.p_state_new['total']:.2f} | SDE Loss: {sde_loss_viz:.6f}")

                except Exception as e:
                    print(f"Live Loop Error: {e}")
                    traceback.print_exc()

                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            self.running = False
        finally:
            self.running = False
            await self.data_collector.close()



if __name__ == "__main__":
    pipeline = TradingPipeline()
    try: asyncio.run(pipeline.run_async())
    except KeyboardInterrupt: pass