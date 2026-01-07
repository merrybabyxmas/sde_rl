import torch
import torch.nn as nn
import torchsde

class LatentSDE(nn.Module):
    def __init__(self, input_dim, hidden_dim, noise_type='diagonal'):
        super().__init__()
        self.sde_type = 'ito'
        self.noise_type = noise_type

        self.theta = nn.Parameter(torch.tensor(0.1))  # Mean reversion speed
        self.mu = nn.Linear(input_dim, hidden_dim)    # Drift (trend)
        self.sigma = nn.Linear(input_dim, hidden_dim) # Diffusion (volatility)

    def f(self, t, y): # Drift function
        return self.theta * (self.mu(y) - y)

    def g(self, t, y): # Diffusion function
        # Ensure positivity or proper scaling if necessary,
        # but for now using the linear output directly as per prompt.
        # For diagonal noise, output shape should match y.
        # If input_dim != hidden_dim, we need to be careful.
        # The prompt implies y has dimension 'hidden_dim' likely,
        # or input_dim is the state dimension.
        # Let's assume the state passed to f and g is of size 'input_dim' (or hidden_dim if projected).
        # Looking at forward, 'x' is passed.
        # If 'x' is raw market data (e.g. 10 dims), then f and g should handle it.
        # Ideally, we might want to project x to a latent state first,
        # but following the prompt literally:
        return self.sigma(y)

    def forward(self, x, delta_t):
        # delta_t is the measured latency
        # ts must be monotonically increasing.
        # We handle batch processing if x is a batch.

        # Ensure delta_t is a tensor and has correct shape if needed.
        # But sdeint expects a 1D tensor of times.

        batch_size = x.shape[0] if x.ndim > 1 else 1
        device = x.device

        # If delta_t is a scalar or single value
        if isinstance(delta_t, (float, int)):
            dt = float(delta_t)
        elif isinstance(delta_t, torch.Tensor):
            dt = delta_t.item()
        else:
            dt = 0.01 # Fallback

        # Avoid 0 time interval
        if dt <= 1e-6:
             dt = 1e-6

        ts = torch.tensor([0, dt]).float().to(device)

        # Brownian motion simulation
        # x is the initial state at t=0
        # method='euler' as per prompt

        # We need to make sure input dimensions match what mu/sigma expect.
        # In __init__, mu/sigma take input_dim -> hidden_dim.
        # But f/g take 'y'. If y evolves, it must match the dimension.
        # The prompt's LatentSDE structure is slightly ambiguous:
        # self.mu = nn.Linear(input_dim, hidden_dim)
        # f(t, y) calls self.mu(y).
        # This implies y has size 'input_dim', BUT output is 'hidden_dim'.
        # If input_dim != hidden_dim, y cannot be fed back into f in next step if sizes mismatch.
        # So input_dim must equal hidden_dim for this specific code to work as a state evolving SDE,
        # OR there is a projection happening elsewhere.
        # Given the prompt: "self.mu = nn.Linear(input_dim, hidden_dim)",
        # if input_dim != hidden_dim, this will fail in the integration step
        # because the output of f (derivative) must have same shape as y.
        # I will assume input_dim == hidden_dim or adjust the code to project first.
        # To be safe and close to prompt, I will verify if I can change it.
        # I will modify it slightly to ensure stability: project input to latent space first if needed,
        # but the prompt treats 'x' as the state. So I will assume input_dim == hidden_dim for the SDE state.

        z = torchsde.sdeint(self, x, ts, method='euler')[1]
        return z # predicted latent state/distribution at t + delta_t
