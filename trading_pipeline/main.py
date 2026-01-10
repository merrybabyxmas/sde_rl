import time
import torch
import torch.nn as nn
import torch.optim as optim
import asyncio
import random
import collections
import traceback
import numpy as np
from config.settings import Config
from src.models.sde_model import LatentSDE
from src.models.rl_agent import PPOAgent, SACAgent
from src.data.collector import RealTimeDataCollector
from src.strategy.reward import calculate_reward
from src.utils.visualizer import TradingVisualizer

class TradingPipeline:
    def __init__(self):
        Config.validate()
        print(f"Initializing Pipeline in {Config.MODE} mode with {Config.RL_ALGO} on {Config.DEVICE}...")

        self.state_dim = Config.STATE_DIM
        self.latent_dim = Config.LATENT_DIM
        self.portfolio_dim = Config.PORTFOLIO_DIM
        self.action_dim = Config.ACTION_DIM

        # Initialize RL Agents based on Algo
        if Config.RL_ALGO == 'SAC':
            AgentClass = SACAgent
        else:
            AgentClass = PPOAgent

        self.old_model = AgentClass(self.state_dim, self.action_dim, self.latent_dim, self.portfolio_dim).to(Config.DEVICE)
        self.new_model = AgentClass(self.state_dim, self.action_dim, self.latent_dim, self.portfolio_dim).to(Config.DEVICE)

        # SAC Target Critic (if SAC)
        if Config.RL_ALGO == 'SAC':
            self.target_critic1 = type(self.new_model.critic1)(self.new_model.critic1.net[0].in_features, 1, Config.RL_HIDDEN_DIMS).to(Config.DEVICE)
            self.target_critic2 = type(self.new_model.critic2)(self.new_model.critic2.net[0].in_features, 1, Config.RL_HIDDEN_DIMS).to(Config.DEVICE)
            self.target_critic1.load_state_dict(self.new_model.critic1.state_dict())
            self.target_critic2.load_state_dict(self.new_model.critic2.state_dict())

            # SAC Entropy (Alpha)
            self.target_entropy = -Config.ACTION_DIM
            self.log_alpha = torch.zeros(1, requires_grad=True, device=Config.DEVICE)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4) # Config?
            self.alpha = self.log_alpha.exp()

            # Optimizers
            self.actor_optimizer = optim.Adam(self.new_model.actor_net.parameters(), lr=Config.RL_LEARNING_RATE)
            self.critic_optimizer = optim.Adam(list(self.new_model.critic1.parameters()) + list(self.new_model.critic2.parameters()), lr=Config.RL_LEARNING_RATE)
        else:
            # PPO Optimizer
            self.rl_optimizer = optim.Adam(self.new_model.parameters(), lr=Config.RL_LEARNING_RATE)

        # SDE
        self.sde_predictor = LatentSDE(input_dim=self.state_dim, latent_dim=self.latent_dim).to(Config.DEVICE)
        self.sde_optimizer = optim.Adam(self.sde_predictor.parameters(), lr=1e-3)
        self.sde_criterion = nn.MSELoss()

        self.data_collector = RealTimeDataCollector(mode=Config.MODE)
        self.visualizer = TradingVisualizer(mode=Config.MODE)

        self.performance_history = {"old": [], "new": []}
        self.replay_buffer = collections.deque(maxlen=Config.REPLAY_BUFFER_SIZE)
        self.training_batch_size = Config.RL_BATCH_SIZE

        self.p_state_old = {"cash": 1000.0, "asset": 0.0, "total": 1000.0, "avg_entry_price": 0.0}
        self.p_state_new = {"cash": 1000.0, "asset": 0.0, "total": 1000.0, "avg_entry_price": 0.0, "gross_total": 1000.0}

        self.running = False
        self.sde_loss_history = []

    async def update_portfolio(self, p_state, action_weight, current_price, latency, is_real_execution=False):
        # ... (Same logic as before, just copying essential parts for brevity or full overwrite) ...
        # I need to output full file, so I will copy the logic from previous turn fully.

        if p_state["asset"] > 0 and p_state["avg_entry_price"] > 0:
            loss_pct = (p_state["avg_entry_price"] - current_price) / p_state["avg_entry_price"]
            if loss_pct > Config.STOP_LOSS_THRESHOLD:
                action_weight = 0.0

        prev_total = p_state["cash"] + p_state["asset"] * current_price
        target_asset_value = prev_total * action_weight
        current_asset_value = p_state["asset"] * current_price
        diff_value = target_asset_value - current_asset_value

        slippage_pct = latency * 0.001
        if diff_value > 0: sim_exec_price = current_price * (1 + slippage_pct)
        else: sim_exec_price = current_price * (1 - slippage_pct)

        slippage_cost = 0.0
        commission_cost = 0.0
        trade_type = None

        if abs(diff_value) > 1.0:
            if is_real_execution and Config.MODE == 'REAL':
                side = 'buy' if diff_value > 0 else 'sell'
                qty_to_trade = abs(diff_value) / current_price
                await self.data_collector.create_order(side, qty_to_trade)

            if diff_value > 0: # Buy
                trade_type = 'buy'
                amount_to_buy = diff_value / sim_exec_price
                cost = amount_to_buy * sim_exec_price
                if p_state["cash"] >= cost:
                    p_state["cash"] -= cost
                    fee_amount = amount_to_buy * Config.COMMISSION_RATE
                    net_amount = amount_to_buy - fee_amount
                    commission_cost = fee_amount * sim_exec_price
                    total_qty = p_state["asset"] + net_amount
                    if total_qty > 0:
                        p_state["avg_entry_price"] = (p_state["asset"] * p_state["avg_entry_price"] + net_amount * sim_exec_price) / total_qty
                    p_state["asset"] += net_amount
                    slippage_cost = (sim_exec_price - current_price) * amount_to_buy
                else:
                    cost = p_state["cash"]
                    amount_to_buy = cost / sim_exec_price
                    p_state["cash"] = 0.0
                    fee_amount = amount_to_buy * Config.COMMISSION_RATE
                    net_amount = amount_to_buy - fee_amount
                    commission_cost = fee_amount * sim_exec_price
                    total_qty = p_state["asset"] + net_amount
                    if total_qty > 0:
                        p_state["avg_entry_price"] = (p_state["asset"] * p_state["avg_entry_price"] + net_amount * sim_exec_price) / total_qty
                    p_state["asset"] += net_amount
                    slippage_cost = (sim_exec_price - current_price) * amount_to_buy
            else: # Sell
                trade_type = 'sell'
                amount_to_sell = abs(diff_value) / sim_exec_price
                if p_state["asset"] >= amount_to_sell:
                    p_state["asset"] -= amount_to_sell
                    proceeds = amount_to_sell * sim_exec_price
                    fee_val = proceeds * Config.COMMISSION_RATE
                    net_proceeds = proceeds - fee_val
                    commission_cost = fee_val
                    p_state["cash"] += net_proceeds
                    slippage_cost = (current_price - sim_exec_price) * amount_to_sell
                else:
                    amount_to_sell = p_state["asset"]
                    p_state["asset"] = 0.0
                    p_state["avg_entry_price"] = 0.0
                    proceeds = amount_to_sell * sim_exec_price
                    fee_val = proceeds * Config.COMMISSION_RATE
                    net_proceeds = proceeds - fee_val
                    commission_cost = fee_val
                    p_state["cash"] += net_proceeds
                    slippage_cost = (current_price - sim_exec_price) * amount_to_sell

        new_total = p_state["cash"] + p_state["asset"] * current_price
        total_cost = slippage_cost + commission_cost
        reward = calculate_reward(prev_total, new_total, total_cost, volatility=0.0)
        p_state["total"] = new_total
        return reward, p_state, commission_cost, trade_type

    def train_step_rl(self):
        if Config.RL_ALGO == 'SAC':
            return self._train_sac()
        else:
            return self._train_ppo()

    def _train_sac(self):
        if len(self.replay_buffer) < self.training_batch_size: return 0.0, 0.0

        # Sample batch (Random)
        batch = random.sample(self.replay_buffer, self.training_batch_size)
        state, sde, p_state, action, reward, next_state, next_sde, next_p_state, done = zip(*batch)

        state = torch.stack(state).to(Config.DEVICE)
        sde = torch.stack(sde).to(Config.DEVICE)
        p_state = torch.stack(p_state).to(Config.DEVICE)
        action = torch.stack(action).to(Config.DEVICE)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(Config.DEVICE)
        next_state = torch.stack(next_state).to(Config.DEVICE)
        next_sde = torch.stack(next_sde).to(Config.DEVICE)
        next_p_state = torch.stack(next_p_state).to(Config.DEVICE)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(Config.DEVICE)

        # Critic Update
        with torch.no_grad():
            next_action, next_log_prob, _ = self.new_model.get_action(next_state, next_sde, next_p_state)
            q1_next, q2_next = self.target_critic1(torch.cat([self.new_model.forward(next_state, next_sde, next_p_state), next_action], dim=1)), \
                               self.target_critic2(torch.cat([self.new_model.forward(next_state, next_sde, next_p_state), next_action], dim=1))
            # Wait, SACAgent.forward returns state feature vector. Critic takes features + action.
            # SACAgent.get_q helper does this.
            # But we need target critics which are just MLPs.
            # Let's fix this: target_critic1 is just the MLP.
            # We need to construct input for it: feature(state) + action.
            # But feature net is in new_model.actor_net part? No, SACAgent has separate structure?
            # In my SACAgent implementation, `forward` concatenates inputs. `actor_net` processes them.
            # `critic1` is MLP(input_dim + action).
            # So:
            obs_next = self.new_model.forward(next_state, next_sde, next_p_state)
            xu_next = torch.cat([obs_next, next_action], dim=1)
            q1_next = self.target_critic1(xu_next)
            q2_next = self.target_critic2(xu_next)

            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * Config.GAMMA * min_q_next

        obs = self.new_model.forward(state, sde, p_state)
        xu = torch.cat([obs, action], dim=1)
        q1 = self.new_model.critic1(xu)
        q2 = self.new_model.critic2(xu)

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Update
        new_action, log_prob, _ = self.new_model.get_action(state, sde, p_state)
        xu_new = torch.cat([obs, new_action], dim=1)
        q1_new = self.new_model.critic1(xu_new)
        q2_new = self.new_model.critic2(xu_new)
        min_q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_prob - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha Update
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # Soft Update Targets
        for param, target_param in zip(self.new_model.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(Config.SAC_TAU * param.data + (1 - Config.SAC_TAU) * target_param.data)
        for param, target_param in zip(self.new_model.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(Config.SAC_TAU * param.data + (1 - Config.SAC_TAU) * target_param.data)

        return actor_loss.item(), critic_loss.item()

    def _train_ppo(self):
        if len(self.replay_buffer) < self.training_batch_size: return 0.0, 0.0

        # PPO uses on-policy buffer usually, but here we use replay buffer as batch source.
        # Strict PPO clears buffer after update.
        batch = list(self.replay_buffer)
        self.replay_buffer.clear() # Clear for On-Policy

        state, sde, p_state, action, reward, log_prob_old = zip(*batch)

        state = torch.stack(state).to(Config.DEVICE)
        sde = torch.stack(sde).to(Config.DEVICE)
        p_state = torch.stack(p_state).to(Config.DEVICE)
        action = torch.stack(action).to(Config.DEVICE)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(Config.DEVICE)
        log_prob_old = torch.stack(log_prob_old).to(Config.DEVICE)

        total_actor_loss = 0
        total_critic_loss = 0

        for _ in range(Config.PPO_K_EPOCHS):
            mean, log_std, val = self.new_model(state, sde, p_state)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

            ratio = torch.exp(log_prob - log_prob_old)
            advantage = reward - val.detach()

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-Config.PPO_EPSILON, 1+Config.PPO_EPSILON) * advantage

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(val, reward)

            loss = actor_loss + 0.5 * critic_loss

            self.rl_optimizer.zero_grad()
            loss.backward()
            self.rl_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

        return total_actor_loss / Config.PPO_K_EPOCHS, total_critic_loss / Config.PPO_K_EPOCHS

    def train_sde_step(self, x, dt, y):
        self.sde_optimizer.zero_grad()
        if isinstance(dt, torch.Tensor) and dt.ndim > 0: dt_scalar = dt.mean().item()
        else: dt_scalar = dt
        _, pred_decoded = self.sde_predictor(x, dt_scalar, decode=True)
        loss = self.sde_criterion(pred_decoded, y)
        loss.backward()
        self.sde_optimizer.step()
        return loss.item()

    async def sde_update_loop(self):
        while self.running:
            try:
                buffer_data = self.data_collector.get_buffer()
                if buffer_data is not None and len(buffer_data) > 64:
                    indices = torch.randint(0, len(buffer_data)-1, (32,))
                    batch = buffer_data[indices]
                    next_batch = buffer_data[indices + 1]

                    x = batch[:, :Config.STATE_DIM].to(Config.DEVICE)
                    dt = batch[:, Config.STATE_DIM-1]
                    y = next_batch[:, :Config.STATE_DIM].to(Config.DEVICE)

                    loss = self.train_sde_step(x, dt, y)
                    self.sde_loss_history.append(loss)
            except: pass
            await asyncio.sleep(1.0)

    def should_replace_model(self):
        if len(self.performance_history["new"]) < 100: return False
        recent_old = sum(self.performance_history["old"][-100:])
        recent_new = sum(self.performance_history["new"][-100:])
        return recent_new > recent_old * 1.05 and recent_new > 0

    async def run_async(self):
        self.running = True
        asyncio.create_task(self.data_collector.connect())
        while len(self.data_collector.buffer) < 500: await asyncio.sleep(0.1)

        # SDE Warmup
        for i in range(50):
            buffer_data = self.data_collector.get_buffer()
            feats = buffer_data[:, :Config.STATE_DIM].to(Config.DEVICE)
            self.train_sde_step(feats[:-1], feats[:-1, -1], feats[1:])

        asyncio.create_task(self.sde_update_loop())

        # RL Loop Vars
        prev_state_tuple = None # (state, sde_latent, p_vec)
        cum_comm_new = 0.0

        try:
            while self.running:
                try:
                    market_data = self.data_collector.get_latest_data()
                    abs_price = market_data[-1].item()
                    state_vec = market_data[:Config.STATE_DIM]
                    latency = state_vec[-1].item()

                    state = state_vec.unsqueeze(0).to(Config.DEVICE)
                    with torch.no_grad():
                        future_latent, _ = self.sde_predictor(state, latency, decode=True)

                    p_vec_old = torch.tensor([self.p_state_old["cash"], self.p_state_old["asset"]]).float().unsqueeze(0).to(Config.DEVICE)
                    p_vec_new = torch.tensor([self.p_state_new["cash"], self.p_state_new["asset"]]).float().unsqueeze(0).to(Config.DEVICE)

                    # Actions
                    with torch.no_grad():
                        action_old, _, _ = self.old_model.get_action(state, future_latent, p_vec_old, deterministic=True)

                    action_new, log_prob_new, _ = self.new_model.get_action(state, future_latent, p_vec_new, deterministic=False)

                    # Execution
                    reward_old, self.p_state_old, _, _ = await self.update_portfolio(
                        self.p_state_old, action_old.item(), abs_price, latency, is_real_execution=True
                    )

                    reward_new, self.p_state_new, comm_new, trade_type_new = await self.update_portfolio(
                        self.p_state_new, action_new.item(), abs_price, latency, is_real_execution=False
                    )

                    cum_comm_new += comm_new
                    gross_wealth_new = self.p_state_new["total"] + cum_comm_new

                    # Store Transition
                    if Config.RL_ALGO == 'SAC':
                        current_tuple = (state[0].cpu(), future_latent[0].cpu(), p_vec_new[0].cpu())
                        if prev_state_tuple is not None:
                            # (s, sde, p, a, r, s', sde', p', d)
                            self.replay_buffer.append((
                                prev_state_tuple[0], prev_state_tuple[1], prev_state_tuple[2], # State
                                prev_action_cpu, prev_reward, # Action, Reward
                                current_tuple[0], current_tuple[1], current_tuple[2], # Next State
                                0.0 # Done
                            ))
                        prev_state_tuple = current_tuple
                        prev_action_cpu = action_new[0].cpu()
                        prev_reward = reward_new
                    else:
                        # PPO (s, sde, p, a, r, log_prob)
                        self.replay_buffer.append((
                            state[0].cpu(), future_latent[0].cpu(), p_vec_new[0].cpu(),
                            action_new[0].cpu(), reward_new, log_prob_new[0].cpu()
                        ))

                    actor_loss, critic_loss = self.train_step_rl()

                    self.performance_history["old"].append(reward_old)
                    self.performance_history["new"].append(reward_new)

                    # Viz Update
                    swap_event = False
                    if self.should_replace_model():
                        self.old_model.load_state_dict(self.new_model.state_dict())
                        self.performance_history["old"] = []
                        self.performance_history["new"] = []
                        self.p_state_old = self.p_state_new.copy()
                        swap_event = True

                    self.visualizer.update({
                        "wealth_net_old": self.p_state_old["total"],
                        "wealth_net_new": self.p_state_new["total"],
                        "wealth_gross_new": gross_wealth_new,
                        "p_state_new": self.p_state_new,
                        "action_new": action_new.item(),
                        "price": abs_price,
                        "swap_event": swap_event,
                        "commission_step": comm_new,
                        "trade_type": trade_type_new,
                        "actor_loss": actor_loss,
                        "critic_loss": critic_loss,
                        "reward": reward_new
                    })

                    if len(self.performance_history["new"]) % 100 == 0:
                         self.visualizer.plot_all()

                except Exception as e:
                    traceback.print_exc()

                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            self.running = False
        finally:
            self.running = False
            await self.data_collector.close()

if __name__ == "__main__":
    pipeline = TradingPipeline()
    try:
        asyncio.run(pipeline.run_async())
    except KeyboardInterrupt:
        pass
