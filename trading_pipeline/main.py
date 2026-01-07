import time
import torch
import torch.nn as nn
import threading
import asyncio
import random
from src.models.sde_model import LatentSDE
from src.models.rl_agent import TradingAgent
from src.data.collector import RealTimeDataCollector

class TradingPipeline:
    def __init__(self):
        # Config
        self.state_dim = 5 # bid, b_qty, ask, a_qty, latency
        self.hidden_dim = 5 # Must match state_dim for the provided SDE code simple integration
        self.action_dim = 3 # Buy, Sell, Hold

        # Models
        self.old_model = TradingAgent(state_dim=self.state_dim, action_dim=self.action_dim, sde_dim=self.hidden_dim)
        self.new_model = TradingAgent(state_dim=self.state_dim, action_dim=self.action_dim, sde_dim=self.hidden_dim)
        self.sde_predictor = LatentSDE(input_dim=self.state_dim, hidden_dim=self.hidden_dim)

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

    def execute_trade(self, action_probs, mode="TEST", price=None):
        # Determine action from probabilities
        action = torch.argmax(action_probs).item()
        # 0: Hold, 1: Buy, 2: Sell (Example mapping)

        # Log trade
        if mode == "REAL" and random.random() < 0.1: # Print occasionally
             print(f"[{mode}] Action: {action}, Price: {price:.2f}")

        # Return a simulated reward for tracking purposes
        # Random reward between -0.01 and 0.01
        return (random.random() - 0.45) * 0.01 if action != 0 else 0

    def should_replace_model(self):
        # Compare performance of last 100 trades
        if len(self.performance_history["new"]) < 100:
            return False

        recent_old = sum(self.performance_history["old"][-100:])
        recent_new = sum(self.performance_history["new"][-100:])

        # If new model is 5% better (and positive)
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

        try:
            while self.running:
                # 1. Get Realtime Data & Latency
                # market_data shape: [5]
                market_data = self.data_collector.get_latest_data()
                network_latency = market_data[-1].item()

                # Add batch dimension
                state = market_data.unsqueeze(0) # [1, 5]

                # 2. SDE Prediction (Future Distribution at t + delta_t)
                with torch.no_grad():
                    # Predict latent state
                    future_dist = self.sde_predictor(state, network_latency)

                # 3. Decision Making (Shadow Mode)
                action_old, val_old = self.old_model(state, future_dist)
                action_new, val_new = self.new_model(state, future_dist)

                # Get current mid price for execution simulation
                current_price = (market_data[0] + market_data[2]) / 2.0

                # 4. Execution & Reward Calculation
                reward_old = self.execute_trade(action_old, mode="REAL", price=current_price)
                reward_new = self.execute_trade(action_new, mode="TEST", price=current_price)

                self.performance_history["old"].append(reward_old)
                self.performance_history["new"].append(reward_new)

                # 5. Model Replacement Logic
                if self.should_replace_model():
                    print(f"Replacing Old Model with New Model... (Old: {sum(self.performance_history['old'][-100:]):.4f}, New: {sum(self.performance_history['new'][-100:]):.4f})")
                    self.old_model.load_state_dict(self.new_model.state_dict())
                    # Reset history or keep it? Usually reset or window moves.
                    # For simplicity, we just clear the buffer to avoid repeated replacements immediately.
                    self.performance_history["old"] = []
                    self.performance_history["new"] = []

                # Training Step (Simplified: Training the new model on the fly)
                # In real scenario, this would be a separate process or buffered batch update.
                # Here we just show where it would happen.
                # train_step(self.new_model, state, future_dist, reward_new)

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("Stopping pipeline...")
            self.running = False

if __name__ == "__main__":
    pipeline = TradingPipeline()
    # Run for a short duration for demonstration if run as script,
    # or indefinitely if intended.
    # For this environment, I'll let it run but catch interrupt.
    pipeline.run()
