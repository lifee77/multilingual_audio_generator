import os
import yt_dlp

def download_youtube_audio(url, output_path="data/raw_audio", output_filename="youtube_audio.mp3"):
    """
    Downloads the audio from a YouTube video and converts it to MP3 using yt-dlp.

    Parameters:
        url (str): URL of the YouTube video.
        output_path (str): Directory where the MP3 file will be saved.
        output_filename (str): Desired filename for the output MP3 file.

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Set up yt-dlp options to extract audio and convert to mp3
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_path, output_filename),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        # Uncomment the next line if you encounter SSL issues:
        # 'nocheckcertificate': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print("Downloading and converting audio...")
        ydl.download([url])
    print(f"Audio saved as: {os.path.join(output_path, output_filename)}")

if __name__ == "__main__":
    # Example YouTube video URL
    youtube_url = "https://www.youtube.com/watch?v=Dw9-E8Gb4Nc"
    download_youtube_audio(youtube_url)
