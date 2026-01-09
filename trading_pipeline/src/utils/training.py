import torch
import torch.nn as nn
import torch.optim as optim

def train_sde_warmup(sde_model, train_loader, epochs=50, device="cpu"):
    """
    Trains the SDE model to learn drift and diffusion from historical data.
    Uses Reconstruction Loss: MSE(decoder(SDE(enc(x))), y).
    """
    sde_model.to(device)
    optimizer = optim.Adam(sde_model.parameters(), lr=1e-3) # Slightly higher LR for warmup
    criterion = nn.MSELoss()

    print(f"Starting SDE Warm-up for {epochs} epochs...")

    for epoch in range(epochs):
        total_loss = 0
        count = 0
        for batch_x, batch_dt, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_dt = batch_dt.to(device)
            batch_y = batch_y.to(device) # Target is RAW data

            optimizer.zero_grad()

            # Use mean dt for batch training simplicity
            dt = batch_dt.mean()

            # Forward with decode=True
            _, pred_y_raw = sde_model(batch_x, dt, decode=True)

            # Loss in Data Space (Reconstruction + Prediction)
            loss = criterion(pred_y_raw, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count if count > 0 else 0
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{epochs} SDE Warm-up Loss: {avg_loss:.6f}")

    print("SDE Warm-up complete.")
