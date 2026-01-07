import torch
import torch.nn as nn

class TradingAgent(nn.Module):
    def __init__(self, state_dim, action_dim, sde_dim=0):
        super().__init__()
        # Input dimension is state_dim (market data) + sde_dim (latent prediction)
        input_dim = state_dim + sde_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.actor = nn.Linear(64, action_dim)   # Buy/Sell/Hold prob
        self.critic = nn.Linear(64, 1)           # Value estimation

    def forward(self, state, sde_output):
        # Concatenate current state and SDE predicted future distribution
        if sde_output is not None:
            combined = torch.cat([state, sde_output], dim=-1)
        else:
            combined = state

        features = self.network(combined)
        return torch.softmax(self.actor(features), dim=-1), self.critic(features)
