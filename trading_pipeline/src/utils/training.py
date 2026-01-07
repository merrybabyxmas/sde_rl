import torch
import torch.nn as nn
import torch.optim as optim

def train_sde_warmup(sde_model, train_loader, epochs=50, device="cpu"):
    """
    Trains the SDE model to learn drift and diffusion from historical data.

    Args:
        sde_model (nn.Module): The LatentSDE model.
        train_loader (DataLoader): DataLoader providing (batch_x, batch_dt, batch_y).
        epochs (int): Number of training epochs.
        device (str): Device to run training on.
    """
    sde_model.to(device)
    optimizer = optim.Adam(sde_model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    print(f"Starting SDE Warm-up for {epochs} epochs...")

    for epoch in range(epochs):
        total_loss = 0
        count = 0
        for batch_x, batch_dt, batch_y in train_loader:
            # batch_x: Current state
            # batch_dt: Latency (delta_t)
            # batch_y: Target state at t + delta_t

            batch_x = batch_x.to(device)
            batch_dt = batch_dt.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # Since SDE returns a path (tensor of size 2 if ts=[0, dt]),
            # we take the last point [1].
            # However, batch_dt might be variable across batch.
            # torchsde.sdeint usually expects fixed ts for the batch or requires careful handling.
            # If batch_dt is a tensor of shape (batch,), we might need to loop or use specific method.
            # For simplicity in this warmup, we assume batch_dt is uniform or we average it,
            # or we process one by one if performance allows.
            # Ideally, we should group by similar dt or use a fixed dt for training if data allows.
            # Here we assume batch_dt is a scalar (mean of batch) for the sdeint call
            # or we rely on the model handling it (our model implementation currently takes a scalar or tensor).

            # Our implementation of forward takes delta_t.
            # If delta_t is a batch, we need to handle it.
            # Current sde_model.forward uses `ts = torch.tensor([0, dt])`.
            # If dt is a batch, sdeint might fail if it expects scalar times.
            # We will use the mean dt for the batch for now as a simplification
            # or pass the tensor if the custom forward handles it (it takes .item() currently).

            # Let's fix the forward to handle batch dt correctly or just use mean.
            dt = batch_dt.mean()

            # Project batch_y to latent space if necessary
            # The SDE model outputs latent state (dim 8), but batch_y is raw data (dim 21).
            # We should train the SDE to predict the future LATENT state.
            # But we only have raw data.
            # Approach: SDE Model should probably output back to data space (Decoder) if it is a generative model.
            # Or, for this specific pipeline where SDE is used for feature extraction,
            # we can train it to minimize reconstruction loss of the future state.
            # However, the current LatentSDE class doesn't have a decoder.
            # To fix the immediate error and make it meaningful:
            # We will use the encoder to project batch_y to latent space and use that as target.
            # This trains the SDE to predict the *latent representation* of the future.

            with torch.no_grad():
                target_latent = sde_model.encoder(batch_y)

            pred_y = sde_model(batch_x, dt)

            # pred_y shape [batch, latent_dim], target_latent [batch, latent_dim]
            loss = criterion(pred_y, target_latent)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count if count > 0 else 0
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} SDE Warm-up Loss: {avg_loss:.6f}")

    print("SDE Warm-up complete.")
