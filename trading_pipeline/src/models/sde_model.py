import torch
import torch.nn as nn
import torchsde
from config.settings import Config

class LatentSDE(nn.Module):
    def __init__(self, input_dim, latent_dim, noise_type='diagonal'):
        super().__init__()
        self.sde_type = 'ito'
        self.noise_type = noise_type
        self.latent_dim = latent_dim
        self.input_dim = input_dim # 21 (State Dim)

        # 1. Encoder: Raw -> Latent
        self.encoder = nn.Linear(input_dim, latent_dim)

        # 2. SDE Nets (Drift & Diffusion)
        hidden_dim = Config.SDE_HIDDEN_DIM
        self.theta = nn.Parameter(torch.tensor(0.1))
        self.mu_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.sigma_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # 3. Decoder: Latent -> Raw (For reconstruction/prediction training)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def f(self, t, y): return self.theta * (self.mu_net(y) - y)
    def g(self, t, y): return torch.sigmoid(self.sigma_net(y))

    def forward(self, x, delta_t, decode=False):
        device = x.device

        # Handle scalar/tensor dt
        if isinstance(delta_t, torch.Tensor):
            dt_val = delta_t.item()
        else:
            dt_val = float(delta_t)

        if dt_val < 0.01: dt_val = 0.01

        # 1. Project current state
        z0 = self.encoder(x)
        ts = torch.tensor([0, dt_val]).float().to(device)

        # 2. SDE Simulation (Predict future latent state)
        # Pass dt to method to ensure fixed step size if needed, though sdeint manages it via ts
        z_t = torchsde.sdeint(self, z0, ts, method='euler', dt=dt_val)[1]

        if decode:
            # 3. Restore future latent state to Raw data dimension (for training)
            decoded_y = self.decoder(z_t)
            return z_t, decoded_y

        return z_t
