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
    def __init__(self, mode='MOCK', save_dir='plots'):
        self.mode = mode
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.reset()

    def reset(self):
        # Persistent storage for the session
        self.data = {
            "step": [],
            "wealth_net_old": [],
            "wealth_net_new": [],
            "wealth_gross_new": [], # For cost comparison
            "price": [],
            "action_new": [],
            "trade_events": [] # list of (step, type 'buy'/'sell', price)
        }
        self.metrics = {
            "cum_commission": 0.0,
            "max_wealth": 1000.0, # Assuming start at 1000
            "mdd": 0.0,
            "wins": 0,
            "losses": 0
        }
        self.swaps = []
        self.step_counter = 0
        self.start_wealth = 1000.0

    def update(self, step_data):
        """
        step_data: {
            "wealth_net_old", "wealth_net_new", "wealth_gross_new",
            "p_state_new", "action_new", "price", "swap_event",
            "commission_step", "trade_type" (None, 'buy', 'sell')
        }
        """
        self.step_counter += 1

        # Append Time Series
        self.data["step"].append(self.step_counter)
        self.data["wealth_net_old"].append(step_data["wealth_net_old"])

        wealth_new = step_data["wealth_net_new"]
        self.data["wealth_net_new"].append(wealth_new)
        self.data["wealth_gross_new"].append(step_data.get("wealth_gross_new", wealth_new))
        self.data["price"].append(step_data["price"])
        self.data["action_new"].append(step_data["action_new"])

        # Metrics Calculation
        comm = step_data.get("commission_step", 0.0)
        self.metrics["cum_commission"] += comm

        # MDD
        if wealth_new > self.metrics["max_wealth"]:
            self.metrics["max_wealth"] = wealth_new
        drawdown = (self.metrics["max_wealth"] - wealth_new) / self.metrics["max_wealth"]
        if drawdown > self.metrics["mdd"]:
            self.metrics["mdd"] = drawdown

        # Win Rate (Approx based on step-wise PnL? Or trade-wise?)
        # Let's track step-wise positive wealth change for simplicity or trade based.
        # User asked for "Win Rate". Usually per trade.
        # Since we rebalance continuously, "Trade" is fuzzy.
        # Let's count intervals where Wealth increased vs decreased as "Daily Win Rate" equivalent?
        # Or simpler: Net PnL > 0 sessions?
        # Let's stick to simple "Step Win Rate" for high freq:
        # if step_data["wealth_net_new"] > prev_wealth: win
        if len(self.data["wealth_net_new"]) > 1:
            prev = self.data["wealth_net_new"][-2]
            if wealth_new > prev: self.metrics["wins"] += 1
            elif wealth_new < prev: self.metrics["losses"] += 1

        # Trade Markers
        t_type = step_data.get("trade_type")
        if t_type:
            self.data["trade_events"].append({
                "step": self.step_counter,
                "type": t_type,
                "price": step_data["price"]
            })

        # Swaps
        if step_data.get("swap_event", False):
            self.swaps.append(self.step_counter)

    def plot_and_save(self):
        if len(self.data["step"]) < 2: return

        # Determine Data Slice (Plot last 500 points to keep chart readable, or all?)
        # User asked for "Persistent... time series flow not broken".
        # Let's plot ALL data but downsample if > 2000 points.

        steps = np.array(self.data["step"])
        w_net = np.array(self.data["wealth_net_new"])
        w_gross = np.array(self.data["wealth_gross_new"])
        prices = np.array(self.data["price"])

        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1])

        # 1. Wealth & Profitability
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(steps, self.data["wealth_net_old"], label="Old Model (Net)", color="gray", alpha=0.5, linestyle="--")
        ax1.plot(steps, w_gross, label="New Model (Gross - No Fee)", color="green", alpha=0.4, linestyle="-.")
        ax1.plot(steps, w_net, label="New Model (Net - Real)", color="blue", linewidth=2)

        # Highlight Swaps
        for swap in self.swaps:
            ax1.axvline(x=swap, color='red', linestyle=':', label='Model Swap' if swap == self.swaps[0] else "")

        ax1.set_title(f"Wealth Evolution [{self.mode}]")
        ax1.set_ylabel("Wealth (USDT)")
        ax1.legend(loc='upper left')

        # KPI Box
        roi = (w_net[-1] - self.start_wealth) / self.start_wealth * 100
        total_steps = self.metrics["wins"] + self.metrics["losses"]
        win_rate = (self.metrics["wins"] / total_steps * 100) if total_steps > 0 else 0.0

        kpi_text = (
            f"ROI: {roi:.2f}%\n"
            f"MDD: {self.metrics['mdd']*100:.2f}%\n"
            f"Win Rate (Step): {win_rate:.1f}%\n"
            f"Cum. Commission: ${self.metrics['cum_commission']:.2f}"
        )
        ax1.text(0.02, 0.5, kpi_text, transform=ax1.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

        # 2. Price & Trades
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(steps, prices, color="black", alpha=0.6, label="Price")

        # Markers
        buy_steps = [e['step'] for e in self.data["trade_events"] if e['type'] == 'buy']
        buy_prices = [e['price'] for e in self.data["trade_events"] if e['type'] == 'buy']
        sell_steps = [e['step'] for e in self.data["trade_events"] if e['type'] == 'sell']
        sell_prices = [e['price'] for e in self.data["trade_events"] if e['type'] == 'sell']

        if buy_steps:
            ax2.scatter(buy_steps, buy_prices, marker='^', color='green', s=100, label='Buy', zorder=5)
        if sell_steps:
            ax2.scatter(sell_steps, sell_prices, marker='v', color='red', s=100, label='Sell', zorder=5)

        ax2.set_title("Market Price & Trade Execution")
        ax2.set_ylabel("Price")
        ax2.legend()

        # 3. Action (Weight)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(steps, self.data["action_new"], color="magenta", label="Target Weight")
        ax3.fill_between(steps, self.data["action_new"], color="magenta", alpha=0.1)
        ax3.set_ylim(-0.1, 1.1)
        ax3.set_title("Agent Target Weight")
        ax3.set_ylabel("Weight")
        ax3.set_xlabel("Steps")

        plt.tight_layout()

        # Overwrite single file
        filename = f"{self.save_dir}/live_dashboard.png"
        plt.savefig(filename)
        plt.close(fig)
