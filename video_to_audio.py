import os
from moviepy.editor import VideoFileClip

def extract_audio_from_videos(input_folder, output_folder, video_extensions=('.mp4', '.avi', '.mov', '.mkv')):
    """
    Extracts audio from all video files in the input folder and saves them as MP3 files in the output folder.

    Parameters:
        input_folder (str): Path to the folder containing video files.
        output_folder (str): Path to the folder where the extracted audio files will be saved.
        video_extensions (tuple): Tuple of video file extensions to process (default: ('.mp4', '.avi', '.mov', '.mkv')).
    """
    # Create the output folder if it doesn't exist.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each file in the input folder.
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(video_extensions):
            video_path = os.path.join(input_folder, file_name)
            # Generate the output file name by replacing the video extension with .mp3.
            output_file_name = os.path.splitext(file_name)[0] + '.mp3'
            audio_path = os.path.join(output_folder, output_file_name)
            
            try:
                print(f"Processing {file_name}...")
                clip = VideoFileClip(video_path)
                clip.audio.write_audiofile(audio_path)
                clip.close()
                print(f"Saved audio to {audio_path}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")


# Extract audio from video files in the 'data/video_data' folder and save them in the 'data/raw_audio' folder (needs further processing).
extract_audio_from_videos('data/video_data', 'data/raw_audio')
