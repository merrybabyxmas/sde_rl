import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from config.settings import Config

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --- PPO Agent ---
class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim=1, sde_dim=8, portfolio_dim=2):
        super().__init__()
        self.input_dim = state_dim + portfolio_dim + sde_dim
        hidden_dims = Config.RL_HIDDEN_DIMS

        # Shared Encoder? Or separate. Standard PPO often separates or shares.
        # Let's keep shared feature extractor for simplicity.
        self.feature_net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )

        self.actor_mean = nn.Linear(hidden_dims[1], action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Linear(hidden_dims[1], 1)

    def forward(self, state, sde_output, portfolio_state):
        if sde_output is not None:
            if sde_output.dim() == 1: sde_output = sde_output.unsqueeze(0)
            if state.dim() == 1: state = state.unsqueeze(0)
            if portfolio_state.dim() == 1: portfolio_state = portfolio_state.unsqueeze(0)
            x = torch.cat([state, portfolio_state, sde_output], dim=-1)
        else:
            x = torch.cat([state, portfolio_state], dim=-1)

        feats = self.feature_net(x)
        mean = torch.sigmoid(self.actor_mean(feats)) # Action [0, 1]
        val = self.critic(feats)
        return mean, self.actor_log_std, val

    def get_action(self, state, sde_output, portfolio_state, deterministic=False):
        mean, log_std, val = self.forward(state, sde_output, portfolio_state)
        std = log_std.exp()

        if deterministic:
            return mean, None, val

        dist = Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, 0.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob, val

# --- SAC Agent ---
class SACAgent(nn.Module):
    def __init__(self, state_dim, action_dim=1, sde_dim=8, portfolio_dim=2):
        super().__init__()
        self.input_dim = state_dim + portfolio_dim + sde_dim
        self.action_dim = action_dim
        hidden_dims = Config.RL_HIDDEN_DIMS

        # Actor (Policy Network)
        # Outputs Mean and LogStd for Gaussian Policy
        # We squash output with Sigmoid for [0, 1] range target weight
        # Standard SAC uses Tanh for [-1, 1], here we want [0, 1].
        # We can predict Logits and apply Sigmoid.
        self.actor_net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.actor_mean = nn.Linear(hidden_dims[1], action_dim)
        self.actor_log_std = nn.Linear(hidden_dims[1], action_dim)

        # Double Critic (Q1, Q2)
        # Input: State + Action
        critic_input_dim = self.input_dim + action_dim
        self.critic1 = MLP(critic_input_dim, 1, hidden_dims)
        self.critic2 = MLP(critic_input_dim, 1, hidden_dims)

    def forward(self, state, sde_output, portfolio_state):
        # Helper to construct full state vector
        if sde_output.dim() == 1: sde_output = sde_output.unsqueeze(0)
        if state.dim() == 1: state = state.unsqueeze(0)
        if portfolio_state.dim() == 1: portfolio_state = portfolio_state.unsqueeze(0)
        return torch.cat([state, portfolio_state, sde_output], dim=-1)

    def get_action(self, state, sde_output, portfolio_state, deterministic=False):
        obs = self.forward(state, sde_output, portfolio_state)
        feats = self.actor_net(obs)
        mean = self.actor_mean(feats)
        log_std = self.actor_log_std(feats)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        dist = Normal(mean, std)

        if deterministic:
            x_t = mean
        else:
            x_t = dist.rsample() # Reparameterization trick

        # Squash to [0, 1] using Sigmoid
        # y = sigmoid(x)
        # log_prob correction for change of variables?
        # Standard SAC uses Tanh. Sigmoid is Tanh(x/2)/2 + 0.5 roughly.
        # Let's use Tanh then scale to [0, 1] for stability.
        # Tanh output [-1, 1] -> (x + 1) / 2 -> [0, 1]

        action = torch.tanh(x_t)

        # Log Prob Calculation
        log_prob = dist.log_prob(x_t).sum(dim=-1, keepdim=True)
        # Correction for Tanh
        log_prob -= (2 * (np.log(2) - x_t - F.softplus(-2 * x_t))).sum(dim=-1, keepdim=True)

        # Transform [-1, 1] to [0, 1]
        final_action = (action + 1) / 2.0

        return final_action, log_prob, None # Critic val not returned here usually

    def get_q(self, state, sde_output, portfolio_state, action):
        obs = self.forward(state, sde_output, portfolio_state)
        xu = torch.cat([obs, action], dim=1)
        return self.critic1(xu), self.critic2(xu)

import numpy as np
