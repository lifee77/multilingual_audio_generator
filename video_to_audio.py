import os
import subprocess

def extract_audio_ffmpeg(input_folder, output_folder, video_extensions=('.mp4', '.avi', '.mov', '.mkv')):
    """
    Extracts audio from all video files in the input folder using FFmpeg and saves them as MP3 files in the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(video_extensions):
            video_path = os.path.join(input_folder, file_name)
            output_file_name = os.path.splitext(file_name)[0] + '.mp3'
            audio_path = os.path.join(output_folder, output_file_name)
            
            try:
                print(f"Processing {file_name} with FFmpeg...")
                # The command below extracts the audio stream and saves it as MP3.
                command = [
                    "ffmpeg", "-i", video_path, "-vn", "-ab", "128k", "-ar", "44100", "-y", audio_path
                ]
                subprocess.run(command, check=True)
                print(f"Saved audio to {audio_path}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

# Example usage:
extract_audio_ffmpeg('data/video_data', 'data/raw_audio')
