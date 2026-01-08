import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import time
import numpy as np

# Set headless mode
plt.switch_backend('Agg')
sns.set_theme(style="darkgrid")

class TradingVisualizer:
    def __init__(self, save_dir='plots', interval=1000):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.interval = interval
        self.reset()

    def reset(self):
        self.data = {
            "step": [],
            "wealth_old": [],
            "wealth_new": [],
            "cash_ratio_new": [],
            "asset_ratio_new": [],
            "action_new": [],
            "price": []
        }
        self.swaps = [] # List of step indices where swap happened
        self.step_counter = 0

    def update(self, step_data):
        """
        step_data: dict with keys:
        wealth_old, wealth_new, p_state_new, action_new, price, swap_event (bool)
        """
        self.step_counter += 1

        self.data["step"].append(self.step_counter)
        self.data["wealth_old"].append(step_data["wealth_old"])
        self.data["wealth_new"].append(step_data["wealth_new"])

        # Portfolio Mix (New Model)
        p_state = step_data["p_state_new"]
        total = p_state["total"]
        if total > 0:
            self.data["cash_ratio_new"].append(p_state["cash"] / total)
            self.data["asset_ratio_new"].append((p_state["asset"] * step_data["price"]) / total)
        else:
            self.data["cash_ratio_new"].append(0)
            self.data["asset_ratio_new"].append(0)

        self.data["action_new"].append(step_data["action_new"])
        self.data["price"].append(step_data["price"])

        if step_data.get("swap_event", False):
            self.swaps.append(self.step_counter)

    def plot_and_save(self):
        if len(self.data["step"]) < 2:
            return

        # print(f"Saving plot with {len(self.data['step'])} points...")
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        # 1. Total Wealth
        ax1 = axes[0]
        ax1.plot(self.data["step"], self.data["wealth_old"], label="Old Model (Prod)", color="gray", alpha=0.7)
        ax1.plot(self.data["step"], self.data["wealth_new"], label="New Model (Challenger)", color="cyan", linewidth=2)

        # Plot Swaps
        for swap_step in self.swaps:
            ax1.axvline(x=swap_step, color='red', linestyle='--', label='Model Swap' if swap_step == self.swaps[0] else "")

        ax1.set_title("Total Wealth Comparison")
        ax1.set_ylabel("Wealth (USDT)")
        ax1.legend()

        # 2. Portfolio Mix (Stacked Area) & Price
        ax2 = axes[1]
        steps = self.data["step"]
        ax2.stackplot(steps, self.data["cash_ratio_new"], self.data["asset_ratio_new"],
                      labels=["Cash", "Asset"], colors=["green", "orange"], alpha=0.6)

        ax2_twin = ax2.twinx()
        ax2_twin.plot(steps, self.data["price"], color="white", linestyle=":", alpha=0.5, label="Price")
        ax2_twin.set_ylabel("Asset Price")

        ax2.set_title("New Model Portfolio Composition")
        ax2.set_ylabel("Ratio")
        ax2.legend(loc='upper left')

        # 3. Action (Target Weight)
        ax3 = axes[2]
        ax3.plot(steps, self.data["action_new"], color="magenta", label="Target Weight")
        ax3.set_ylim(-0.1, 1.1)
        ax3.set_title("Agent Action (Target Weight)")
        ax3.set_ylabel("Weight")
        ax3.set_xlabel("Step")

        plt.tight_layout()

        # Save
        timestamp = int(time.time())
        filename = f"{self.save_dir}/dashboard_{timestamp}.png"
        plt.savefig(filename)
        plt.close(fig)

        # Check reset
        if len(self.data["step"]) >= self.interval:
            self.reset()
            # Optionally keep last point for continuity? Or just clean cut.
            # Clean cut as per prompt "reset logic".
