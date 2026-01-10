import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import time
import numpy as np

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
        # Persistent storage
        self.data = {
            "step": [],
            "wealth_net_old": [],
            "wealth_net_new": [],
            "wealth_gross_new": [],
            "price": [],
            "action_new": [],
            "trade_events": [],

            # Detailed Portfolio
            "cash_new": [],
            "asset_val_new": [],

            # SDE
            "sde_pred_price": [], # Top-1 Bid Price Prediction
            "sde_error": [],

            # RL
            "actor_loss": [],
            "critic_loss": [],
            "reward": []
        }
        self.metrics = {
            "cum_commission": 0.0,
            "max_wealth": 1000.0,
            "mdd": 0.0,
            "wins": 0,
            "losses": 0
        }
        self.swaps = []
        self.step_counter = 0
        self.start_wealth = 1000.0

    def update(self, step_data):
        self.step_counter += 1

        # Unpack
        self.data["step"].append(self.step_counter)
        self.data["wealth_net_old"].append(step_data["wealth_net_old"])
        wealth_new = step_data["wealth_net_new"]
        self.data["wealth_net_new"].append(wealth_new)
        self.data["wealth_gross_new"].append(step_data.get("wealth_gross_new", wealth_new))
        self.data["price"].append(step_data["price"])
        self.data["action_new"].append(step_data["action_new"])

        # Portfolio Details
        p_state = step_data["p_state_new"]
        self.data["cash_new"].append(p_state["cash"])
        self.data["asset_val_new"].append(p_state["asset"] * step_data["price"])

        # SDE
        # We assume step_data["sde_pred"] is the decoded tensor (21 dims)
        # Price is normalized. We just plot the first dim (Bid0) or Error.
        sde_pred = step_data.get("sde_pred_decoded", None)
        if sde_pred is not None:
            # Reconstruct absolute price from normalized
            # Norm = (P - Mid)/Mid * 100
            # P = Norm/100 * Mid + Mid
            # Use current mid price as reference? Or prev?
            # Prediction was made at t-1 for t. Reference was Mid(t-1).
            # This is tricky without storing Mid(t-1).
            # Let's just track the MSE error passed from main loop if available,
            # or just plot the raw normalized value vs current normalized value.
            # Let's assume main.py passes `sde_loss_item` directly.
            self.data["sde_error"].append(step_data.get("sde_loss", 0.0))
        else:
            self.data["sde_error"].append(0.0)

        # RL
        self.data["actor_loss"].append(step_data.get("actor_loss", 0.0))
        self.data["critic_loss"].append(step_data.get("critic_loss", 0.0))
        self.data["reward"].append(step_data.get("reward", 0.0))

        # Metrics (Commission, MDD, Win Rate)
        comm = step_data.get("commission_step", 0.0)
        self.metrics["cum_commission"] += comm

        if wealth_new > self.metrics["max_wealth"]:
            self.metrics["max_wealth"] = wealth_new
        drawdown = (self.metrics["max_wealth"] - wealth_new) / self.metrics["max_wealth"]
        if drawdown > self.metrics["mdd"]:
            self.metrics["mdd"] = drawdown

        if len(self.data["wealth_net_new"]) > 1:
            prev = self.data["wealth_net_new"][-2]
            if wealth_new > prev: self.metrics["wins"] += 1
            elif wealth_new < prev: self.metrics["losses"] += 1

        t_type = step_data.get("trade_type")
        if t_type:
            self.data["trade_events"].append({
                "step": self.step_counter,
                "type": t_type,
                "price": step_data["price"]
            })

        if step_data.get("swap_event", False):
            self.swaps.append(self.step_counter)

    def plot_all(self):
        if len(self.data["step"]) < 2: return

        # Generate all plots
        self.plot_dashboard() # Main
        self.plot_portfolio()
        self.plot_sde()
        self.plot_rl()

    def plot_dashboard(self):
        steps = np.array(self.data["step"])
        w_net = np.array(self.data["wealth_net_new"])
        w_gross = np.array(self.data["wealth_gross_new"])
        prices = np.array(self.data["price"])

        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1])

        # 1. Wealth
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(steps, self.data["wealth_net_old"], label="Old Model (Net)", color="gray", alpha=0.5, linestyle="--")
        ax1.plot(steps, w_gross, label="New (Gross)", color="green", alpha=0.4)
        ax1.plot(steps, w_net, label="New (Net)", color="blue", linewidth=2)

        for swap in self.swaps:
            ax1.axvline(x=swap, color='red', linestyle=':')

        ax1.set_title(f"Main Dashboard [{self.mode}]")
        ax1.legend(loc='upper left')

        # KPI
        roi = (w_net[-1] - self.start_wealth) / self.start_wealth * 100
        total_steps = self.metrics["wins"] + self.metrics["losses"]
        win_rate = (self.metrics["wins"] / total_steps * 100) if total_steps > 0 else 0.0
        kpi_text = f"ROI: {roi:.2f}%\nMDD: {self.metrics['mdd']*100:.2f}%\nWinRate: {win_rate:.1f}%\nFee: ${self.metrics['cum_commission']:.2f}"
        ax1.text(0.02, 0.5, kpi_text, transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8))

        # 2. Price & Trades
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(steps, prices, color="black", alpha=0.6)

        buy_steps = [e['step'] for e in self.data["trade_events"] if e['type'] == 'buy']
        buy_prices = [e['price'] for e in self.data["trade_events"] if e['type'] == 'buy']
        sell_steps = [e['step'] for e in self.data["trade_events"] if e['type'] == 'sell']
        sell_prices = [e['price'] for e in self.data["trade_events"] if e['type'] == 'sell']

        if buy_steps: ax2.scatter(buy_steps, buy_prices, marker='^', color='green', s=100, label='Buy')
        if sell_steps: ax2.scatter(sell_steps, sell_prices, marker='v', color='red', s=100, label='Sell')
        ax2.legend()

        # 3. Action
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(steps, self.data["action_new"], color="magenta")
        ax3.set_ylim(-0.1, 1.1)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/live_dashboard.png")
        plt.close(fig)

    def plot_portfolio(self):
        steps = np.array(self.data["step"])
        cash = np.array(self.data["cash_new"])
        asset = np.array(self.data["asset_val_new"])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, cash, label="Cash (USDT)", color="green")
        ax.plot(steps, asset, label="Asset Value (USDT)", color="orange")
        ax.stackplot(steps, cash, asset, labels=["Cash", "Asset"], colors=["green", "orange"], alpha=0.1)

        ax.set_title("Portfolio Composition Change")
        ax.set_ylabel("Value")
        ax.legend()

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/live_portfolio.png")
        plt.close(fig)

    def plot_sde(self):
        # SDE Learning Performance (Reconstruction/Prediction Error)
        steps = np.array(self.data["step"])
        errors = np.array(self.data["sde_error"])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, errors, label="SDE Prediction/Recon Error (MSE)", color="purple")

        # Smooth curve
        if len(errors) > 20:
            avg = pd.Series(errors).rolling(20).mean()
            ax.plot(steps, avg, label="Moving Avg (20)", color="yellow", linewidth=2)

        ax.set_title("SDE Model Performance (Training/Inference Error)")
        ax.set_xlabel("Step")
        ax.set_ylabel("MSE Loss")
        ax.set_yscale('log') # Log scale for loss
        ax.legend()

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/live_sde.png")
        plt.close(fig)

    def plot_rl(self):
        steps = np.array(self.data["step"])
        actor = np.array(self.data["actor_loss"])
        critic = np.array(self.data["critic_loss"])
        rewards = np.array(self.data["reward"])

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        axes[0].plot(steps, actor, label="Actor Loss", color="blue")
        axes[0].set_title("RL Actor Loss")

        axes[1].plot(steps, critic, label="Critic Loss", color="red")
        axes[1].set_title("RL Critic Loss")

        axes[2].plot(steps, rewards, label="Reward", color="green", alpha=0.3)
        if len(rewards) > 20:
            avg = pd.Series(rewards).rolling(20).mean()
            axes[2].plot(steps, avg, label="Reward MA(20)", color="black")
        axes[2].set_title("Reward History")

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/live_rl.png")
        plt.close(fig)
