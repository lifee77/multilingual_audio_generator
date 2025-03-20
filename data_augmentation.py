import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import os
from scipy import signal

# Load existing metadata
metadata = pd.read_csv('data/metadata.csv')

# Create output directory
output_dir = "data/augmented_data"
os.makedirs(output_dir, exist_ok=True)

# Track new files
new_data = []

# Select a subset of files to augment
jeevan_english = metadata[(metadata['speaker_label'] == 'Jeevan') & 
                         (metadata['language_label'] == 'English')].head(5)
jeevan_not_english = metadata[(metadata['speaker_label'] == 'Jeevan') & 
                             (metadata['language_label'] == 'Not_English')].head(5)
not_jeevan_english = metadata[(metadata['speaker_label'] == 'Not_Jeevan') & 
                             (metadata['language_label'] == 'English')].head(5)
not_jeevan_not_english = metadata[(metadata['speaker_label'] == 'Not_Jeevan') & 
                                 (metadata['language_label'] == 'Not_English')].head(5)

files_to_augment = pd.concat([jeevan_english, jeevan_not_english, 
                              not_jeevan_english, not_jeevan_not_english])

# Define augmentation methods
def change_pitch(audio, sr, semitones):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)

def change_speed(audio, speed_factor):
    return librosa.effects.time_stretch(audio, rate=speed_factor)

def add_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise

# Perform augmentation
for idx, row in files_to_augment.iterrows():
    input_file = os.path.join('data', row['filename'])
    
    # Load audio
    audio, sr = librosa.load(input_file, sr=None)
    
    # Apply different augmentations
    # 1. Pitch shift up
    pitch_up = change_pitch(audio, sr, 2)
    out_file = f"pitch_up_{row['filename']}"
    sf.write(os.path.join(output_dir, out_file), pitch_up, sr)
    new_data.append({
        "filename": out_file,
        "speaker_label": row['speaker_label'],
        "language_label": row['language_label']
    })
    
    # 2. Pitch shift down
    pitch_down = change_pitch(audio, sr, -2)
    out_file = f"pitch_down_{row['filename']}"
    sf.write(os.path.join(output_dir, out_file), pitch_down, sr)
    new_data.append({
        "filename": out_file,
        "speaker_label": row['speaker_label'],
        "language_label": row['language_label']
    })
    
    # 3. Speed up (maintains the speaker characteristics but changes tempo)
    speed_up = change_speed(audio, 1.2)
    out_file = f"speed_up_{row['filename']}"
    sf.write(os.path.join(output_dir, out_file), speed_up, sr)
    new_data.append({
        "filename": out_file,
        "speaker_label": row['speaker_label'],
        "language_label": row['language_label']
    })
    
    # 4. Add background noise
    noisy = add_noise(audio, 0.005)
    out_file = f"noisy_{row['filename']}"
    sf.write(os.path.join(output_dir, out_file), noisy, sr)
    new_data.append({
        "filename": out_file,
        "speaker_label": row['speaker_label'],
        "language_label": row['language_label']
    })

# Create metadata for augmented samples
augmented_df = pd.DataFrame(new_data)
augmented_df.to_csv(os.path.join(output_dir, "augmented_metadata.csv"), index=False)

print(f"Generated {len(new_data)} augmented audio files with preserved labels")