import pandas as pd
import librosa
import numpy as np
import os

# Paths
csv_path = "data.csv"  # Adjust to your CSV file path
audio_dir = "audio_files"  # Adjust to your audio directory

# Load CSV and filter for Jeevan
df = pd.read_csv(csv_path)
jeevan_files = df[df["speaker_label"] == "Jeevan"]["filename"].tolist()
print(f"Found {len(jeevan_files)} files for Jeevan: {jeevan_files}")

# Parameters for STFT
sr = 16000  # Sample rate
n_fft = 2048  # Window size
hop_length = 512  # Hop length

# Convert audio files to spectrograms
spectrograms = []
for filename in jeevan_files:
    audio_path = os.path.join(audio_dir, filename)
    audio, _ = librosa.load(audio_path, sr=sr)
    # Compute magnitude spectrogram
    spec = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    # Log-scale for better training
    spec = librosa.amplitude_to_db(spec, ref=np.max)
    # Pad or truncate to a fixed length (e.g., 128 frames) for consistency
    if spec.shape[1] > 128:
        spec = spec[:, :128]
    else:
        spec = np.pad(spec, ((0, 0), (0, 128 - spec.shape[1])), mode="constant")
    spectrograms.append(spec)

# Convert to numpy array: (num_samples, freq_bins, time_frames)
spectrograms = np.array(spectrograms)  # Shape: (N, 1025, 128)
print(f"Spectrogram shape: {spectrograms.shape}")

# Normalize to [0, 1] for model training
spec_min, spec_max = spectrograms.min(), spectrograms.max()
spectrograms = (spectrograms - spec_min) / (spec_max - spec_min)