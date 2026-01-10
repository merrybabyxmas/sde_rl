import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import time
import numpy as np

# Headless 환경 대응
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
        # 전체 세션 누적 데이터 (WandB 스타일)
        self.data = {
            "step": [],
            "wealth_net_old": [],
            "wealth_net_new": [],
            "wealth_gross_new": [],
            "price": [],
            "action_new": [],
            "trade_events": [],
            "cash_new": [],
            "asset_val_new": [],
            "sde_loss": [],
            "actor_loss": [],
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
        self.data["step"].append(self.step_counter)
        
        # 수익 데이터 (Net vs Gross)
        wealth_net = step_data["wealth_net_new"]
        self.data["wealth_net_old"].append(step_data["wealth_net_old"])
        self.data["wealth_net_new"].append(wealth_net)
        self.data["wealth_gross_new"].append(step_data.get("wealth_gross_new", wealth_net))
        
        self.data["price"].append(step_data["price"])
        self.data["action_new"].append(step_data["action_new"])

        # 포트폴리오 세부
        p_state = step_data["p_state_new"]
        self.data["cash_new"].append(p_state["cash"])
        self.data["asset_val_new"].append(p_state["asset"] * step_data["price"])

        # 학습 지표 (SDE & RL)
        self.data["sde_loss"].append(step_data.get("sde_loss", 0.0))
        self.data["actor_loss"].append(step_data.get("actor_loss", 0.0))
        self.data["reward"].append(step_data.get("reward", 0.0))

        # KPI 계산
        comm = step_data.get("commission_step", 0.0)
        self.metrics["cum_commission"] += comm
        
        if wealth_net > self.metrics["max_wealth"]:
            self.metrics["max_wealth"] = wealth_net
        dd = (self.metrics["max_wealth"] - wealth_net) / self.metrics["max_wealth"]
        if dd > self.metrics["mdd"]: self.metrics["mdd"] = dd

        if len(self.data["wealth_net_new"]) > 1:
            prev = self.data["wealth_net_new"][-2]
            if wealth_net > prev: self.metrics["wins"] += 1
            elif wealth_net < prev: self.metrics["losses"] += 1

        # 거래 이벤트 마커
        t_type = step_data.get("trade_type")
        if t_type:
            self.data["trade_events"].append({
                "step": self.step_counter, "type": t_type, "price": step_data["price"]
            })

        if step_data.get("swap_event", False):
            self.swaps.append(self.step_counter)

    def plot_all(self):
        """WandB 스타일의 통합 대시보드 생성 (live_dashboard.png 고정)"""
        if len(self.data["step"]) < 2: return
        
        steps = np.array(self.data["step"])
        w_net = np.array(self.data["wealth_net_new"])
        w_gross = np.array(self.data["wealth_gross_new"])
        prices = np.array(self.data["price"])

        fig = plt.figure(figsize=(15, 18))
        gs = fig.add_gridspec(4, 1, height_ratios=[2, 2, 1, 1])

        # 1. 자산 추이 (Net vs Gross)
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(steps, w_gross, label="Gross Wealth (No Fee)", color="green", alpha=0.3)
        ax1.plot(steps, w_net, label="Net Wealth (Actual)", color="blue", linewidth=2)
        ax1.plot(steps, self.data["wealth_net_old"], label="Old Model", color="gray", alpha=0.5, linestyle="--")
        
        for swap in self.swaps:
            ax1.axvline(x=swap, color='red', linestyle=':', label='Model Swap' if swap == self.swaps[0] else "")
        
        roi = (w_net[-1] - self.start_wealth) / self.start_wealth * 100
        wr = (self.metrics["wins"] / (self.metrics["wins"]+self.metrics["losses"]) * 100) if (self.metrics["wins"]+self.metrics["losses"]) > 0 else 0
        kpi = f"ROI: {roi:.2f}% | MDD: {self.metrics['mdd']*100:.2f}% | WinRate: {wr:.1f}%\nTotal Fee: ${self.metrics['cum_commission']:.2f}"
        ax1.text(0.02, 0.95, kpi, transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax1.set_title(f"TRADING CONSOLE [{self.mode}] - Total Wealth")
        ax1.legend(loc='upper left')

        # 2. 가격 및 매매 마커
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(steps, prices, color="black", alpha=0.6, label="Asset Price")
        buys = [e for e in self.data["trade_events"] if e['type'] == 'buy']
        sells = [e for e in self.data["trade_events"] if e['type'] == 'sell']
        if buys: ax2.scatter([e['step'] for e in buys], [e['price'] for e in buys], marker='^', color='green', s=120, label='BUY', zorder=5)
        if sells: ax2.scatter([e['step'] for e in sells], [e['price'] for e in sells], marker='v', color='red', s=120, label='SELL', zorder=5)
        ax2.set_title("Price & Trade Execution")
        ax2.legend()

        # 3. 포트폴리오 비중 & 액션
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        cash = np.array(self.data["cash_new"])
        asset = np.array(self.data["asset_val_new"])
        ax3.stackplot(steps, cash, asset, labels=["Cash", "Asset Value"], colors=["#A8E6CF", "#FFD3B6"], alpha=0.6)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(steps, self.data["action_new"], color="magenta", linewidth=1, label="Target Weight")
        ax3_twin.set_ylim(-0.05, 1.05)
        ax3.set_title("Portfolio Mix & Agent Decision")
        ax3.legend(loc='upper left'); ax3_twin.legend(loc='upper right')

        # 4. 학습 지표 (SDE & RL Loss)
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        sde_l = np.array(self.data["sde_loss"])
        ax4.plot(steps, sde_l, color="purple", label="SDE MSE Loss", alpha=0.8)
        if len(sde_l) > 50:
            ax4.plot(steps, pd.Series(sde_l).rolling(50).mean(), color="yellow", label="SDE MA(50)")
        ax4.set_yscale('log')
        ax4.set_title("Learning Performance (SDE Loss)")
        ax4.legend()

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/live_dashboard.png")
        plt.close(fig)