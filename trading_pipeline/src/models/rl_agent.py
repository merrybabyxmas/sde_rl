import torch
import torch.nn as nn
from torch.distributions import Normal

class TradingAgent(nn.Module):
    def __init__(self, state_dim, action_dim=1, sde_dim=8, portfolio_dim=2):
        super().__init__()
        # Input: Market State (21) + Portfolio State (2) + SDE Latent (8)
        self.input_dim = state_dim + portfolio_dim + sde_dim

        self.shared_net = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # Actor: Outputs Target Weight (0.0 to 1.0)
        # We use a Sigmoid for deterministic part, but for PPO we often use Normal distribution.
        # However, for Target Weight, bounded [0, 1] is best.
        # We can use Beta distribution or just Sigmoid + Gaussian noise for exploration.
        # For simplicity in this "Online Training" setup, let's use a mean (Sigmoid) and learnable std.
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim)) # Log std dev

        # Critic: Value function
        self.critic = nn.Linear(64, 1)

    def forward(self, state, sde_output, portfolio_state):
        # portfolio_state: [batch, 2] (Cash, Asset_Qty) -> We might want normalized features ideally
        # But for now passing raw or locally normalized in main.

        if sde_output is not None:
            # Ensure sde_output shape matches batch
            if sde_output.dim() == 1: sde_output = sde_output.unsqueeze(0)
            if state.dim() == 1: state = state.unsqueeze(0)
            if portfolio_state.dim() == 1: portfolio_state = portfolio_state.unsqueeze(0)

            combined = torch.cat([state, portfolio_state, sde_output], dim=-1)
        else:
            combined = torch.cat([state, portfolio_state], dim=-1) # Fallback

        features = self.shared_net(combined)

        # Actor
        # Sigmoid to bound mean between 0 and 1 (Target Weight)
        mean = torch.sigmoid(self.actor_mean(features))

        # Critic
        value = self.critic(features)

        return mean, self.actor_log_std, value

    def get_action(self, state, sde_output, portfolio_state, deterministic=False):
        mean, log_std, value = self.forward(state, sde_output, portfolio_state)
        std = log_std.exp()

        if deterministic:
            return mean, None, value

        # Sampling
        # Note: Normal distribution is unbounded, but we want [0,1].
        # A common trick is sampling from Normal then apply Sigmoid, but here mean is already Sigmoid.
        # If we add noise to Sigmoid output, we might go out of bound.
        # Better: Actor outputs logit, add noise, then Sigmoid.
        # But let's stick to simple Normal approx clamped or Beta.
        # Let's use Normal around mean and clamp action to [0, 1].

        dist = Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, 0.0, 1.0)

        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        return action, log_prob, value
