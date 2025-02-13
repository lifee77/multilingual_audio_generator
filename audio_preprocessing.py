import os
import numpy as np
import librosa
import soundfile as sf

def remove_silence(audio, sr, top_db=20):
    """
    Removes silence from the beginning and end of an audio signal.

    Parameters:
        audio (np.ndarray): Audio waveform.
        sr (int): Sampling rate.
        top_db (int): The threshold (in decibels) below the reference to consider as silence.

    Returns:
        np.ndarray: Audio waveform with beginning and ending silence removed.
    """
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio

def remove_long_silence(audio, sr, top_db=20, gap_threshold=2.0):
    """
    Removes silence (or very low energy segments) from the center of an audio if they are longer than gap_threshold seconds.

    This function uses librosa.effects.split() to detect non-silent intervals.
    If the gap between two non-silent intervals is longer than the gap_threshold (in seconds),
    the gap is removed (i.e., the segments are concatenated).

    Parameters:
        audio (np.ndarray): Audio waveform.
        sr (int): Sampling rate.
        top_db (int): Threshold (in decibels) for silence detection.
        gap_threshold (float): Duration (in seconds) of silence to remove from the center.

    Returns:
        np.ndarray: Audio waveform with long silences removed.
    """
    # Get non-silent intervals (each as [start, end] in samples)
    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0:
        return audio  # Return original if no non-silent parts are found

    output_audio = []
    # Always include the first non-silent interval
    output_audio.append(audio[intervals[0][0]:intervals[0][1]])
    
    # Iterate over subsequent intervals
    for i in range(1, len(intervals)):
        prev_end = intervals[i-1][1]
        current_start = intervals[i][0]
        gap_duration = (current_start - prev_end) / sr
        if gap_duration < gap_threshold:
            # If the silence is short, include it.
            output_audio.append(audio[prev_end:current_start])
        # Otherwise, skip the gap.
        output_audio.append(audio[intervals[i][0]:intervals[i][1]])
    
    return np.concatenate(output_audio)

def normalize_audio(audio):
    """
    Normalizes the audio signal to have zero mean and unit variance.

    Parameters:
        audio (np.ndarray): Audio waveform.

    Returns:
        np.ndarray: Normalized audio waveform.
    """
    mean = np.mean(audio)
    std = np.std(audio)
    if std > 0:
        normalized_audio = (audio - mean) / std
    else:
        normalized_audio = audio - mean
    return normalized_audio

def pad_audio(audio, target_length, sr):
    """
    Pads the audio signal with zeros to reach the target length (in seconds).

    Parameters:
        audio (np.ndarray): Audio waveform.
        target_length (int): Target length in seconds.
        sr (int): Sampling rate.

    Returns:
        np.ndarray: Padded audio waveform.
    """
    target_samples = target_length * sr
    if len(audio) < target_samples:
        padded_audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
    else:
        padded_audio = audio
    return padded_audio

def split_audio(audio, sr, split_length=7):
    """
    Splits the audio into multiple segments of fixed length.

    Parameters:
        audio (np.ndarray): Audio waveform.
        sr (int): Sampling rate.
        split_length (int): Desired segment length in seconds.

    Returns:
        list of np.ndarray: List of audio segments.
    """
    num_samples_per_segment = split_length * sr
    total_samples = len(audio)
    segments = []
    for start in range(0, total_samples, num_samples_per_segment):
        end = start + num_samples_per_segment
        segment = audio[start:end]
        # Option: Only add segments that are exactly split_length long.
        # Alternatively, you could pad the last segment.
        if len(segment) == num_samples_per_segment:
            segments.append(segment)
    return segments

def process_audio_file(file_path, output_directory, sr=16000, split_length=7, top_db=20, gap_threshold=2.0):
    """
    Processes an audio file by removing silence, normalizing, optionally removing long internal silences,
    and splitting it into segments. Saves each segment as a separate WAV file in output_directory.

    Parameters:
        file_path (str): Path to the input audio file.
        output_directory (str): Directory to save the processed audio segments.
        sr (int): Sampling rate.
        split_length (int): Length (in seconds) of each segment.
        top_db (int): Threshold for silence removal.
        gap_threshold (float): Duration (in seconds) of silence in the center to remove.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Load audio
    audio, sr = librosa.load(file_path, sr=sr)
    
    # Remove silence from beginning and end
    audio = remove_silence(audio, sr, top_db=top_db)
    
    # Remove long silence from the center (if gap > gap_threshold seconds)
    audio = remove_long_silence(audio, sr, top_db=top_db, gap_threshold=gap_threshold)
    
    # Normalize audio
    audio = normalize_audio(audio)
    
    # Split audio into segments of fixed length
    segments = split_audio(audio, sr, split_length=split_length)
    
    # Save segments
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    for i, segment in enumerate(segments):
        output_file = os.path.join(output_directory, f"{base_filename}_segment_{i+1}.wav")
        sf.write(output_file, segment, sr)
        print(f"Saved: {output_file}")

def process_directory(input_directory, output_directory, sr=16000, split_length=7, top_db=20, gap_threshold=2.0):
    """
    Processes all WAV audio files in the input_directory by applying pre-processing and splitting.
    Saves all processed segments in output_directory.

    Parameters:
        input_directory (str): Directory containing the input audio files.
        output_directory (str): Directory to save processed audio segments.
        sr (int): Sampling rate.
        split_length (int): Length (in seconds) of each segment.
        top_db (int): Threshold for silence removal.
        gap_threshold (float): Duration (in seconds) of silence in the center to remove.
    """
    for filename in os.listdir(input_directory):
        if filename.lower().endswith('.wav'):
            file_path = os.path.join(input_directory, filename)
            print(f"Processing file: {file_path}")
            process_audio_file(file_path, output_directory, sr=sr, split_length=split_length, top_db=top_db, gap_threshold=gap_threshold)

# Additional audio pre-processing functions can be added here
# For example, noise reduction, bandpass filtering, or data augmentation functions.

if __name__ == "__main__":
    # Example usage:
    # Specify the input directory containing your raw audio files
    input_dir = "data/raw_audio"  # Change this to your input audio directory
    # Specify the output directory where processed segments will be saved
    output_dir = "data/audio_data"  # This directory will be created if it does not exist
    
    process_directory(input_dir, output_dir)
