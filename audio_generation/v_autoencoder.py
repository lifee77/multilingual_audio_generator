import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define VAE (Variational Autoencoder) model
class VAE(nn.Module):
    def __init__(self, input_channels=1, freq_bins=1025, time_frames=128):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64 * (freq_bins // 4) * (time_frames // 4), 64)
        self.fc_logvar = nn.Linear(64 * (freq_bins // 4) * (time_frames // 4), 64)
        self.decoder_fc = nn.Linear(64, 64 * (freq_bins // 4) * (time_frames // 4))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(-1, 64, freq_bins // 4, time_frames // 4)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + 0.01 * KLD

# Prepare data for training
spectrograms_tensor = torch.FloatTensor(spectrograms).unsqueeze(1)  # Add channel dim: (N, 1, 1025, 128)
dataset = TensorDataset(spectrograms_tensor, spectrograms_tensor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Small batch size for CPU

# Initialize model and optimizer
model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        x = batch[0]
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        loss = loss_function(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader.dataset)}")

# Save model
torch.save(model.state_dict(), "vae_model.pth")