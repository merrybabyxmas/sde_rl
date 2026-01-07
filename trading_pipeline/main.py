import time
import torch
import torch.nn as nn
import torch.optim as optim
import threading
import asyncio
import random
import collections
from src.models.sde_model import LatentSDE
from src.models.rl_agent import TradingAgent
from src.data.collector import RealTimeDataCollector
from src.strategy.reward import calculate_reward
from src.utils.training import train_sde_warmup

class TradingPipeline:
    def __init__(self):
        # Dimensions
        self.state_dim = 21 # Market(20) + Latency(1)
        self.latent_dim = 8
        self.portfolio_dim = 2 # Cash, Asset_Qty (we can pass Entry Price too if needed but simpler for now)
        self.action_dim = 1 # Target Weight (0.0 - 1.0)

        # Models
        self.old_model = TradingAgent(self.state_dim, self.action_dim, self.latent_dim, self.portfolio_dim)
        self.new_model = TradingAgent(self.state_dim, self.action_dim, self.latent_dim, self.portfolio_dim)
        self.sde_predictor = LatentSDE(input_dim=self.state_dim, latent_dim=self.latent_dim)

        # Optimizers
        self.optimizer = optim.Adam(self.new_model.parameters(), lr=1e-4)

        # Data
        self.data_collector = RealTimeDataCollector(mock=True)

        # Buffers
        self.performance_history = {"old": [], "new": []}
        self.replay_buffer = [] # For Online RL
        self.training_batch_size = 20

        # Portfolio States
        # Adding avg_entry_price to track realized PnL
        self.p_state_old = {"cash": 1000.0, "asset": 0.0, "total": 1000.0, "avg_entry_price": 0.0}
        self.p_state_new = {"cash": 1000.0, "asset": 0.0, "total": 1000.0, "avg_entry_price": 0.0}

        self.running = False

    def start_data_collection(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.data_collector.connect())

    def update_portfolio(self, p_state, action_weight, current_price, latency):
        """
        Updates portfolio based on Target Weight action.
        Returns: reward, new_total_wealth
        """
        # Previous Total Wealth (Mark-to-Market)
        prev_total = p_state["cash"] + p_state["asset"] * current_price

        # Target Asset Value
        target_asset_value = prev_total * action_weight
        current_asset_value = p_state["asset"] * current_price

        diff_value = target_asset_value - current_asset_value

        # Latency & Slippage Simulation
        # Slippage: Random perturbation based on latency, biased against trade direction usually.
        # But here we stick to simple random + latency penalty.
        slippage_pct = latency * 0.01 # Reduced factor to 1% per second of latency (0.005s -> 0.005%)

        # If Buying, Price is higher. If Selling, Price is lower.
        if diff_value > 0:
            executed_price = current_price * (1 + slippage_pct)
        else:
            executed_price = current_price * (1 - slippage_pct)

        slippage_cost = 0.0
        realized_pnl = 0.0

        # Execution
        if abs(diff_value) > 1.0: # Minimum trade size $1
            if diff_value > 0: # Buy
                amount_to_buy = diff_value / executed_price
                cost = amount_to_buy * executed_price

                # Check cash constraint
                if p_state["cash"] >= cost:
                    # Update Avg Entry Price: (Old_Qty * Old_Avg + New_Qty * New_Price) / Total_Qty
                    total_qty = p_state["asset"] + amount_to_buy
                    if total_qty > 0:
                        p_state["avg_entry_price"] = (p_state["asset"] * p_state["avg_entry_price"] + amount_to_buy * executed_price) / total_qty

                    p_state["cash"] -= cost
                    p_state["asset"] += amount_to_buy
                    slippage_cost = (executed_price - current_price) * amount_to_buy

                else:
                    # Buy max possible
                    cost = p_state["cash"]
                    amount_to_buy = cost / executed_price

                    total_qty = p_state["asset"] + amount_to_buy
                    if total_qty > 0:
                        p_state["avg_entry_price"] = (p_state["asset"] * p_state["avg_entry_price"] + amount_to_buy * executed_price) / total_qty

                    p_state["cash"] = 0.0
                    p_state["asset"] += amount_to_buy
                    slippage_cost = (executed_price - current_price) * amount_to_buy

            else: # Sell
                amount_to_sell = abs(diff_value) / executed_price

                # Check asset constraint
                if p_state["asset"] >= amount_to_sell:
                    # Realized PnL = (Exit Price - Entry Price) * Quantity
                    realized_pnl = (executed_price - p_state["avg_entry_price"]) * amount_to_sell

                    p_state["asset"] -= amount_to_sell
                    p_state["cash"] += amount_to_sell * executed_price
                    slippage_cost = (current_price - executed_price) * amount_to_sell # Price - Executed (lower)
                else:
                    # Sell max
                    amount_to_sell = p_state["asset"]
                    realized_pnl = (executed_price - p_state["avg_entry_price"]) * amount_to_sell

                    p_state["asset"] = 0.0
                    p_state["cash"] += amount_to_sell * executed_price
                    slippage_cost = (current_price - executed_price) * amount_to_sell
                    p_state["avg_entry_price"] = 0.0 # Reset if empty

        # New Total Wealth
        new_total = p_state["cash"] + p_state["asset"] * current_price

        # Reward Calculation
        # We want to reward:
        # 1. Total Wealth Growth (Unrealized + Realized)
        # 2. Realized Profit (explicit bonus?) - Actually Wealth Growth captures both.
        # However, to encourage learning, we can use the step-wise change in wealth minus slippage.
        # Change in Wealth = (New Total - Old Total)
        # This inherently includes Realized PnL and Unrealized PnL changes.
        # But we explicitly penalize slippage (which is already in the wealth change, but double penalty encourages avoiding it?)
        # Let's stick to simple wealth change + explicit volatility penalty (if we had it).
        # We add Realized PnL explicitly? No, it's part of wealth.
        # BUT, the prompt asked for "Realized Profit... accurate calculation... via calculate_reward".
        # So `calculate_reward` should probably take the delta wealth.

        # Reward = (New Total - Prev Total) / Prev Total * Scale - Penalty
        # This is strictly % return.

        raw_profit = new_total - prev_total
        # raw_profit already subtracts slippage_cost implicitly because we paid more/sold for less.
        # So we don't need to double penalize unless we want extra aversion.

        reward = calculate_reward(prev_total, new_total, slippage_cost, volatility=0.0)

        p_state["total"] = new_total
        return reward, p_state

    def train_step_rl(self):
        if len(self.replay_buffer) < self.training_batch_size:
            return

        batch = self.replay_buffer[:self.training_batch_size]
        self.replay_buffer = self.replay_buffer[self.training_batch_size:]

        states, sde_outs, p_states, actions, rewards, log_probs = zip(*batch)

        states = torch.stack(states)
        sde_outs = torch.stack(sde_outs)
        p_states = torch.stack(p_states)
        actions = torch.stack(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        # log_probs = torch.stack(log_probs).detach() # Not used for now in simple PG

        # Simple Advantage (Reward - Value)
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
        if len(self.performance_history["new"]) < 100:
            return False
        recent_old = sum(self.performance_history["old"][-100:])
        recent_new = sum(self.performance_history["new"][-100:])
        # Ensure positive performance and significant improvement
        return recent_new > recent_old * 1.05 and recent_new > 0

    def run(self):
        self.running = True

        t = threading.Thread(target=self.start_data_collection)
        t.daemon = True
        t.start()

        print("Waiting for data buffer to fill...")
        while len(self.data_collector.buffer) < 50:
            time.sleep(0.1)

        print("Starting SDE Warm-up...")
        buffer_data = self.data_collector.get_buffer()
        if buffer_data is not None and len(buffer_data) > 10:
            x = buffer_data[:-1]
            y = buffer_data[1:]
            dt = buffer_data[:-1, -1]
            loader = [(x, dt, y)]
            train_sde_warmup(self.sde_predictor, loader, epochs=5)

        print("SDE Warm-up Complete. Starting Trading Loop...")

        try:
            while self.running:
                market_data = self.data_collector.get_latest_data()
                latency = market_data[-1].item()
                current_price = (market_data[0] + market_data[10]) / 2.0

                state = market_data.unsqueeze(0)

                with torch.no_grad():
                    future_dist = self.sde_predictor(state, latency)

                # Portfolio Vector: [Cash, Asset_Qty]
                # We normalize simple by passing raw for now, model learns scale
                p_vec_old = torch.tensor([self.p_state_old["cash"], self.p_state_old["asset"]]).float().unsqueeze(0)
                p_vec_new = torch.tensor([self.p_state_new["cash"], self.p_state_new["asset"]]).float().unsqueeze(0)

                # OLD MODEL
                with torch.no_grad():
                    action_old, _, _ = self.old_model.get_action(state, future_dist, p_vec_old, deterministic=True)

                reward_old, self.p_state_old = self.update_portfolio(
                    self.p_state_old, action_old.item(), current_price, latency
                )

                # NEW MODEL
                action_new, log_prob_new, _ = self.new_model.get_action(state, future_dist, p_vec_new, deterministic=False)

                reward_new, self.p_state_new = self.update_portfolio(
                    self.p_state_new, action_new.item(), current_price, latency
                )

                self.replay_buffer.append((
                    state[0], future_dist[0], p_vec_new[0], action_new[0], reward_new, log_prob_new[0]
                ))

                self.train_step_rl()

                self.performance_history["old"].append(reward_old)
                self.performance_history["new"].append(reward_new)

                if len(self.performance_history["new"]) % 20 == 0:
                    print(f"Step: {len(self.performance_history['new'])} | "
                          f"Price: {current_price:.2f} | "
                          f"Wealth Old: {self.p_state_old['total']:.2f} | "
                          f"Wealth New: {self.p_state_new['total']:.2f} | "
                          f"Action New: {action_new.item():.2f}")

                if self.should_replace_model():
                    print(">>> REPLACING MODEL: New model outperformed Old model! <<<")
                    self.old_model.load_state_dict(self.new_model.state_dict())
                    self.performance_history["old"] = []
                    self.performance_history["new"] = []
                    self.p_state_old = self.p_state_new.copy()

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("Stopping...")
            self.running = False

if __name__ == "__main__":
    pipeline = TradingPipeline()
    pipeline.run()
