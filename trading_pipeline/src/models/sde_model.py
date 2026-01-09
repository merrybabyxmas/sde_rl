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
        self.input_dim = input_dim

        # Encoder: Data -> Latent
        self.encoder = nn.Linear(input_dim, latent_dim)

        # Decoder: Latent -> Data (For visualization and reconstruction loss)
        self.decoder = nn.Linear(latent_dim, input_dim)

        self.theta = nn.Parameter(torch.tensor(0.1))

        # Drift & Diffusion as MLPs using Config.SDE_HIDDEN_DIM
        hidden_dim = Config.SDE_HIDDEN_DIM

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

    def f(self, t, y): # Drift function
        # theta * (mu(y) - y)
        return self.theta * (self.mu_net(y) - y)

    def g(self, t, y): # Diffusion function
        return torch.sigmoid(self.sigma_net(y))

    def forward(self, x, delta_t, decode=False):
        batch_size = x.shape[0] if x.ndim > 1 else 1
        device = x.device

        if isinstance(delta_t, (float, int)):
            dt = float(delta_t)
        elif isinstance(delta_t, torch.Tensor):
            dt = delta_t.item()
        else:
            dt = 0.01

        if dt < 0.01: dt = 0.01

        z0 = self.encoder(x)

        # Ensure ts is on the same device as input x
        ts = torch.tensor([0, dt]).float().to(device)

        # Explicitly pass dt as step size
        z_t = torchsde.sdeint(self, z0, ts, method='euler', dt=dt)[1]

        if decode:
            return z_t, self.decoder(z_t)

        return z_t
