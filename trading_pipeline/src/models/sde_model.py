import torch
import torch.nn as nn
import torchsde

class LatentSDE(nn.Module):
    def __init__(self, input_dim, latent_dim, noise_type='diagonal'):
        super().__init__()
        self.sde_type = 'ito'
        self.noise_type = noise_type

        self.latent_dim = latent_dim
        # Encoder: Project input data (21 dims) to SDE latent dimension
        self.encoder = nn.Linear(input_dim, latent_dim)

        self.theta = nn.Parameter(torch.tensor(0.1))  # Mean reversion speed

        # Drift & Diffusion
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.sigma = nn.Linear(latent_dim, latent_dim)

    def f(self, t, y): # Drift function
        return self.theta * (self.mu(y) - y)

    def g(self, t, y): # Diffusion function
        return torch.sigmoid(self.sigma(y))

    def forward(self, x, delta_t):
        batch_size = x.shape[0] if x.ndim > 1 else 1
        device = x.device

        if isinstance(delta_t, (float, int)):
            dt = float(delta_t)
        elif isinstance(delta_t, torch.Tensor):
            dt = delta_t.item()
        else:
            dt = 0.01

        if dt <= 1e-6:
             dt = 1e-6

        z0 = self.encoder(x)

        # Ensure dt is large enough to avoid recursion depth issues in brownian tree construction
        # or use a different noise type/method if dt is very small.
        # But 1e-6 should be fine for euler.
        # The recursion error often comes from floating point issues or extremely small intervals relative to something else.
        # Or if dt is 0.
        # We already ensured dt > 1e-6.
        # Let's increase min dt slightly or check if dt is somehow becoming 0.

        # Another possibility: torchsde has issues with small dt in 'euler'.
        # We can try fixed step size options if needed, but for now let's clamp dt higher.
        if dt < 0.01: dt = 0.01 # Enforce minimum 10ms for stability

        ts = torch.tensor([0, dt]).float().to(device)

        # Explicitly pass dt as step size to avoid adaptive logic issues or too small steps
        z_t = torchsde.sdeint(self, z0, ts, method='euler', dt=dt)[1]
        return z_t
