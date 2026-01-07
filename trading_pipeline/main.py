import time
import torch
import torch.nn as nn
import threading
import asyncio
import random
from src.models.sde_model import LatentSDE
from src.models.rl_agent import TradingAgent
from src.data.collector import RealTimeDataCollector
from src.strategy.reward import calculate_reward

class TradingPipeline:
    def __init__(self):
        # Config
        self.state_dim = 5 # bid, b_qty, ask, a_qty, latency
        self.latent_dim = 8 # Projected latent dimension for SDE
        self.action_dim = 3 # Buy, Sell, Hold

        # Models
        # TradingAgent receives State + SDE Output (latent_dim)
        self.old_model = TradingAgent(state_dim=self.state_dim, action_dim=self.action_dim, sde_dim=self.latent_dim)
        self.new_model = TradingAgent(state_dim=self.state_dim, action_dim=self.action_dim, sde_dim=self.latent_dim)

        # SDE Predictor now uses explicit latent dim
        self.sde_predictor = LatentSDE(input_dim=self.state_dim, latent_dim=self.latent_dim)

        # Data
        self.data_collector = RealTimeDataCollector(mock=True) # Use mock for safety/environment

        # Performance Tracking
        self.performance_history = {"old": [], "new": []}

        # State
        self.running = False

    def start_data_collection(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.data_collector.connect())

    def execute_trade(self, action_probs, current_price, prev_wealth, mode="TEST"):
        """
        Executes trade with simulated slippage and calculates reward.
        Returns: (reward, new_wealth)
        """
        # Determine action from probabilities
        action = torch.argmax(action_probs).item()
        # 0: Hold, 1: Buy, 2: Sell

        latency_penalty = self.data_collector.current_latency * 10.0 # Weight for latency impact

        # Simulate execution price with slippage
        # In real world, slippage moves against you. For simplicity/randomness:
        executed_price = current_price + (random.uniform(-1, 1) * latency_penalty)

        new_wealth = prev_wealth
        if action == 1: # Buy (Simulate holding asset for a period or instant revaluation)
             # Simplified: Assume we buy and the value becomes new_wealth relative to price change
             # Ideally we hold a position. Here we simplify:
             # If we Buy, we bet price goes up.
             # Let's say we hold for 1 step.
             # But here we are updating wealth instantaneously based on execution?
             # The user prompt says: new_wealth = prev_wealth * (executed_price / current_price) if action == 1 else prev_wealth
             # This logic implies immediate realization of the price difference between "current" and "executed".
             # This is a bit odd (usually you buy at executed, then later sell).
             # But following user instruction:
             new_wealth = prev_wealth * (executed_price / current_price)
        elif action == 2: # Sell
             # If we Sell, maybe we bet down? Or just exit?
             # User example only covered action == 1.
             # I will assume Sell means Shorting or exiting.
             # Let's assume Sell is Short for symmetry or just exit (no change).
             # For this simulation, let's treat it as no position (Hold cash) or Short.
             # Let's keep it simple: Sell = Cash out. If we were already in cash (prev_wealth), no change.
             # If we treat this as a single-step bet:
             pass

        # Slippage is difference between expected (current) and actual (executed)
        slippage = abs(executed_price - current_price)

        # Calculate Reward
        reward = calculate_reward(prev_wealth, new_wealth, slippage, volatility=0.01)

        # Log trade occasionally
        if mode == "REAL" and random.random() < 0.1:
             print(f"[{mode}] Action: {action}, Price: {current_price:.2f}, Exec: {executed_price:.2f}, Reward: {reward:.4f}")

        return reward, new_wealth

    def should_replace_model(self):
        # Compare performance of last 100 trades
        if len(self.performance_history["new"]) < 100:
            return False

        recent_old = sum(self.performance_history["old"][-100:])
        recent_new = sum(self.performance_history["new"][-100:])

        # If new model is 5% better (and positive, or just relatively better)
        # Handle negative rewards gracefully if needed, but simple comparison:
        if recent_new > recent_old * 1.05:
            return True
        return False

    def run(self):
        self.running = True

        # Start data collector in background thread
        t = threading.Thread(target=self.start_data_collection)
        t.daemon = True
        t.start()

        print("Pipeline started. Waiting for data...")
        time.sleep(2) # Wait for buffer to fill

        # Initial Wealth
        prev_wealth_old = 1000.0
        prev_wealth_new = 1000.0

        try:
            while self.running:
                # 1. Get Realtime Data & Latency
                market_data = self.data_collector.get_latest_data()
                network_latency = market_data[-1].item()

                # Add batch dimension
                state = market_data.unsqueeze(0) # [1, 5]

                # 2. SDE Prediction (Future Distribution at t + delta_t)
                with torch.no_grad():
                    # Predict latent state
                    future_dist = self.sde_predictor(state, network_latency)

                # 3. Decision Making (Shadow Mode)
                action_old, _ = self.old_model(state, future_dist)
                action_new, _ = self.new_model(state, future_dist)

                # Get current mid price for execution simulation
                current_price = (market_data[0] + market_data[2]) / 2.0

                # 4. Execution & Reward Calculation
                reward_old, prev_wealth_old = self.execute_trade(action_old, current_price, prev_wealth_old, mode="REAL")
                reward_new, prev_wealth_new = self.execute_trade(action_new, current_price, prev_wealth_new, mode="TEST")

                self.performance_history["old"].append(reward_old)
                self.performance_history["new"].append(reward_new)

                # 5. Model Replacement Logic
                if self.should_replace_model():
                    print(f"Replacing Old Model with New Model... (Old: {sum(self.performance_history['old'][-100:]):.4f}, New: {sum(self.performance_history['new'][-100:]):.4f})")
                    self.old_model.load_state_dict(self.new_model.state_dict())
                    self.performance_history["old"] = []
                    self.performance_history["new"] = []

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("Stopping pipeline...")
            self.running = False

if __name__ == "__main__":
    pipeline = TradingPipeline()
    pipeline.run()
