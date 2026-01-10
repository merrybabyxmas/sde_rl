import torch
import torch.nn as nn
from torch.distributions import Normal
from config.settings import Config

import torch
import torch.nn as nn
import torch.nn.functional as F

class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, portfolio_dim):
        super().__init__()
        # SDE 잠재 벡터와 상태, 포트폴리오를 결합하여 입력
        input_size = state_dim + latent_dim + portfolio_dim
        hidden_dims = [256, 256]
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dims[1], action_dim)
        self.log_std = nn.Linear(hidden_dims[1], action_dim)

    def forward(self, state, sde_latent, p_vec):
        x = torch.cat([state, sde_latent, p_vec], dim=-1)
        x = self.net(x)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -20, 2) # 안정성을 위해 클램핑
        return mu, log_std

    def sample(self, state, sde_latent, p_vec):
        mu, log_std = self.forward(state, sde_latent, p_vec)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample() # Reparameterization trick
        action = torch.sigmoid(z) # 0~1 사이 비중으로 변환
        log_prob = dist.log_prob(z) - torch.log(action * (1 - action) + 1e-6) # Jacobian 보정
        return action, log_prob.sum(-1, keepdim=True)

class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, portfolio_dim):
        super().__init__()
        input_size = state_dim + latent_dim + portfolio_dim + action_dim
        self.q_net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, sde_latent, p_vec, action):
        x = torch.cat([state, sde_latent, p_vec, action], dim=-1)
        return self.q_net(x)



class TradingAgent(nn.Module):
    def __init__(self, state_dim, action_dim=1, sde_dim=8, portfolio_dim=2):
        super().__init__()

        # Use Config for architecture
        hidden_dims = Config.RL_HIDDEN_DIMS

        # Input: Market State + Portfolio State + SDE Latent
        self.input_dim = state_dim + portfolio_dim + sde_dim

        # Build Shared Network Dynamically
        layers = []
        input_size = self.input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_size, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            input_size = h_dim

        self.shared_net = nn.Sequential(*layers)

        # Output Heads (Actor & Critic)
        # They connect to the last hidden layer
        last_dim = hidden_dims[-1] if hidden_dims else self.input_dim

        self.actor_mean = nn.Linear(last_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        self.critic = nn.Linear(last_dim, 1)

    def forward(self, state, sde_output, portfolio_state):
        if sde_output is not None:
            if sde_output.dim() == 1: sde_output = sde_output.unsqueeze(0)
            if state.dim() == 1: state = state.unsqueeze(0)
            if portfolio_state.dim() == 1: portfolio_state = portfolio_state.unsqueeze(0)

            combined = torch.cat([state, portfolio_state, sde_output], dim=-1)
        else:
            combined = torch.cat([state, portfolio_state], dim=-1)

        features = self.shared_net(combined)

        # Actor (Sigmoid for Target Weight 0-1)
        mean = torch.sigmoid(self.actor_mean(features))

        # Critic
        value = self.critic(features)

        return mean, self.actor_log_std, value

    def get_action(self, state, sde_output, portfolio_state, deterministic=False):
        mean, log_std, value = self.forward(state, sde_output, portfolio_state)
        std = log_std.exp()

        if deterministic:
            return mean, None, value

        dist = Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, 0.0, 1.0)

        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        return action, log_prob, value
