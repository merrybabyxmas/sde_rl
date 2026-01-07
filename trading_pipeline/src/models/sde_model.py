import torch
import torch.nn as nn
import torchsde

class LatentSDE(nn.Module):
    def __init__(self, input_dim, latent_dim, noise_type='diagonal'):
        super().__init__()
        self.sde_type = 'ito'
        self.noise_type = noise_type

        self.latent_dim = latent_dim
        # 1. Encoder: Project input data to SDE latent dimension
        self.encoder = nn.Linear(input_dim, latent_dim)

        self.theta = nn.Parameter(torch.tensor(0.1))  # Mean reversion speed

        # 2. Drift & Diffusion: Now y and mu(y) have matching dimensions (latent_dim)
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.sigma = nn.Linear(latent_dim, latent_dim)

    def f(self, t, y): # Drift function
        # Dimensions match, safe to compute
        return self.theta * (self.mu(y) - y)

    def g(self, t, y): # Diffusion function
        # Apply sigmoid for volatility stability as suggested
        return torch.sigmoid(self.sigma(y))

    def forward(self, x, delta_t):
        # delta_t is the measured latency
        batch_size = x.shape[0] if x.ndim > 1 else 1
        device = x.device

        # If delta_t is a scalar or single value
        if isinstance(delta_t, (float, int)):
            dt = float(delta_t)
        elif isinstance(delta_t, torch.Tensor):
            dt = delta_t.item()
        else:
            dt = 0.01 # Fallback

        # Ensure minimum time interval
        if dt <= 1e-6:
             dt = 1e-6

        # Project input to latent space
        z0 = self.encoder(x) # [batch, latent_dim]

        ts = torch.tensor([0, dt]).float().to(device)

        # Brownian motion simulation
        # z0 is the initial state in latent space
        z_t = torchsde.sdeint(self, z0, ts, method='euler')[1]
        return z_t # predicted latent state/distribution at t + delta_t
