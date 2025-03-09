# Convert spectrogram to audio
audio = librosa.griffinlim(
    librosa.db_to_amplitude(generated_spectrogram),  # Convert dB back to amplitude
    n_iter=32,
    hop_length=hop_length,
    n_fft=n_fft
)

# Save the audio
librosa.output.write_wav("generated_jeevan_audio.wav", audio, sr=sr)
print("Generated audio saved as 'generated_jeevan_audio.wav'")