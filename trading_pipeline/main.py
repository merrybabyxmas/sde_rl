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
from src.models.rl_agent import TradingAgent
from src.data.collector import RealTimeDataCollector
from src.strategy.reward import calculate_reward
from src.utils.training import train_sde_warmup
from src.utils.visualizer import TradingVisualizer

class TradingPipeline:
    def __init__(self):
        # Validate Config
        Config.validate()
        print(f"Initializing Pipeline in {Config.MODE} mode on {Config.DEVICE}...")

        # Dimensions
        self.state_dim = Config.STATE_DIM
        self.latent_dim = Config.LATENT_DIM
        self.portfolio_dim = Config.PORTFOLIO_DIM
        self.action_dim = Config.ACTION_DIM

        # Models - Move to Config.DEVICE
        self.old_model = TradingAgent(self.state_dim, self.action_dim, self.latent_dim, self.portfolio_dim).to(Config.DEVICE)
        self.new_model = TradingAgent(self.state_dim, self.action_dim, self.latent_dim, self.portfolio_dim).to(Config.DEVICE)
        self.sde_predictor = LatentSDE(input_dim=self.state_dim, latent_dim=self.latent_dim).to(Config.DEVICE)

        self.optimizer = optim.Adam(self.new_model.parameters(), lr=Config.RL_LEARNING_RATE)

        # Data Collector
        self.data_collector = RealTimeDataCollector(mode=Config.MODE)

        self.performance_history = {"old": [], "new": []}
        self.replay_buffer = []
        self.training_batch_size = Config.RL_BATCH_SIZE

        # State Tracking
        self.p_state_old = {"cash": 1000.0, "asset": 0.0, "total": 1000.0, "avg_entry_price": 0.0}
        self.p_state_new = {"cash": 1000.0, "asset": 0.0, "total": 1000.0, "avg_entry_price": 0.0}

        # Visualization
        self.visualizer = TradingVisualizer(interval=100) # Reset every 100 steps for demo (1000 in prod)

        self.running = False

    async def update_portfolio(self, p_state, action_weight, current_price, latency, is_real_execution=False):
        """
        Updates portfolio with Commission and Stop-Loss logic.
        """
        # --- HARD STOP-LOSS CHECK ---
        if p_state["asset"] > 0 and p_state["avg_entry_price"] > 0:
            loss_pct = (p_state["avg_entry_price"] - current_price) / p_state["avg_entry_price"]
            if loss_pct > Config.STOP_LOSS_THRESHOLD:
                action_weight = 0.0 # Force Sell

        prev_total = p_state["cash"] + p_state["asset"] * current_price

        target_asset_value = prev_total * action_weight
        current_asset_value = p_state["asset"] * current_price

        diff_value = target_asset_value - current_asset_value

        slippage_pct = latency * 0.001
        if diff_value > 0:
            sim_exec_price = current_price * (1 + slippage_pct)
        else:
            sim_exec_price = current_price * (1 - slippage_pct)

        slippage_cost = 0.0
        commission_cost = 0.0

        # Execution Logic
        if abs(diff_value) > 1.0:

            # Real Execution Order
            if is_real_execution and Config.MODE == 'REAL':
                side = 'buy' if diff_value > 0 else 'sell'
                qty_to_trade = abs(diff_value) / current_price
                order = await self.data_collector.create_order(side, qty_to_trade)
                if order:
                    print(f"REAL EXECUTION: {side} {qty_to_trade:.6f} @ ~{current_price}")

            # Simulation / State Update Logic
            if diff_value > 0: # Buy
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
        return reward, p_state

    def train_step_rl(self):
        if len(self.replay_buffer) < self.training_batch_size:
            return

        batch = self.replay_buffer[:self.training_batch_size]
        self.replay_buffer = self.replay_buffer[self.training_batch_size:]

        states, sde_outs, p_states, actions, rewards, log_probs = zip(*batch)

        # Stack and Move to Config.DEVICE
        states = torch.stack(states).to(Config.DEVICE)
        sde_outs = torch.stack(sde_outs).to(Config.DEVICE)
        p_states = torch.stack(p_states).to(Config.DEVICE)
        actions = torch.stack(actions).to(Config.DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(Config.DEVICE)

        mean, log_std, value = self.new_model(states, sde_outs, p_states)
        dist = torch.distributions.Normal(mean, log_std.exp())
        new_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)

        advantage = rewards - value.detach()
        actor_loss = -(new_log_probs * advantage).mean()
        critic_loss = nn.MSELoss()(value, rewards)
        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def should_replace_model(self):
        if len(self.performance_history["new"]) < 100: return False
        recent_old = sum(self.performance_history["old"][-100:])
        recent_new = sum(self.performance_history["new"][-100:])
        return recent_new > recent_old * 1.05 and recent_new > 0

    async def run_async(self):
        self.running = True
        collector_task = asyncio.create_task(self.data_collector.connect())

        print("Waiting for data buffer to fill...")
        while len(self.data_collector.buffer) < 50:
            await asyncio.sleep(0.1)

        print("Starting SDE Warm-up...")
        buffer_data = self.data_collector.get_buffer()
        if buffer_data is not None and len(buffer_data) > 10:
            feats = buffer_data[:, :Config.STATE_DIM]
            x = feats[:-1]
            y = feats[1:]
            dt = feats[:-1, -1]
            loader = [(x, dt, y)]
            try:
                train_sde_warmup(self.sde_predictor, loader, epochs=Config.SDE_WARMUP_EPOCHS, device=Config.DEVICE)
            except Exception as e:
                print(f"SDE Warmup Failed: {e}")

        print("Starting Trading Loop...")

        try:
            while self.running:
                try:
                    market_data = self.data_collector.get_latest_data()

                    abs_price = market_data[-1].item()
                    state_vec = market_data[:Config.STATE_DIM]
                    latency = state_vec[-1].item()

                    # Move state to DEVICE
                    state = state_vec.unsqueeze(0).to(Config.DEVICE)

                    with torch.no_grad():
                        future_dist = self.sde_predictor(state, latency)

                    # Portfolio Vector
                    p_vec_old = torch.tensor([self.p_state_old["cash"], self.p_state_old["asset"]]).float().unsqueeze(0).to(Config.DEVICE)
                    p_vec_new = torch.tensor([self.p_state_new["cash"], self.p_state_new["asset"]]).float().unsqueeze(0).to(Config.DEVICE)

                    # OLD
                    with torch.no_grad():
                        action_old, _, _ = self.old_model.get_action(state, future_dist, p_vec_old, deterministic=True)

                    reward_old, self.p_state_old = await self.update_portfolio(
                        self.p_state_old, action_old.item(), abs_price, latency, is_real_execution=True
                    )

                    # NEW
                    action_new, log_prob_new, _ = self.new_model.get_action(state, future_dist, p_vec_new, deterministic=False)

                    reward_new, self.p_state_new = await self.update_portfolio(
                        self.p_state_new, action_new.item(), abs_price, latency, is_real_execution=False
                    )

                    self.replay_buffer.append((
                        state[0].cpu(), future_dist[0].cpu(), p_vec_new[0].cpu(), action_new[0].cpu(), reward_new, log_prob_new[0].cpu()
                    ))

                    self.train_step_rl()

                    self.performance_history["old"].append(reward_old)
                    self.performance_history["new"].append(reward_new)

                    # --- Visualization Update ---
                    swap_event = False
                    if self.should_replace_model():
                        print(">>> REPLACING MODEL <<<")
                        self.old_model.load_state_dict(self.new_model.state_dict())
                        self.performance_history["old"] = []
                        self.performance_history["new"] = []
                        self.p_state_old = self.p_state_new.copy()
                        swap_event = True

                    # Update Viz every step
                    self.visualizer.update({
                        "wealth_old": self.p_state_old["total"],
                        "wealth_new": self.p_state_new["total"],
                        "p_state_new": self.p_state_new,
                        "action_new": action_new.item(),
                        "price": abs_price,
                        "swap_event": swap_event
                    })

                    # Plot periodically (e.g. every 10 steps to reduce IO)
                    if len(self.performance_history["new"]) % 10 == 0:
                         self.visualizer.plot_and_save()

                    if len(self.performance_history["new"]) % 20 == 0:
                        print(f"Step: {len(self.performance_history['new'])} | "
                              f"Price: {abs_price:.2f} | "
                              f"Wealth Old: {self.p_state_old['total']:.2f} | "
                              f"Wealth New: {self.p_state_new['total']:.2f} | "
                              f"Action New: {action_new.item():.2f}")

                except Exception as e:
                    print(f"Loop Error: {e}")
                    # traceback.print_exc()

                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            print("Stopping...")
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
