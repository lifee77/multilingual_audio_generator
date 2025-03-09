# Load trained model
model = VAE()
model.load_state_dict(torch.load("vae_model.pth"))
model.eval()

# Generate a spectrogram
with torch.no_grad():
    z = torch.randn(1, 64)  # Random latent vector
    generated_spectrogram = model.decode(z).squeeze(0).numpy()  # Shape: (1, 1025, 128) -> (1025, 128)

# Denormalize to original scale
generated_spectrogram = generated_spectrogram * (spec_max - spec_min) + spec_min