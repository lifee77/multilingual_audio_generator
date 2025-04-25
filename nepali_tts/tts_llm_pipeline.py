#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nepali TTS-LLM Pipeline
This script creates a voice assistant that combines:
1. Speech recognition for Nepali and English
2. Google's Gemini LLM for processing queries
3. Fine-tuned Nepali TTS for spoken responses
"""

import os
import sys
import json
import time
import queue
import threading
import tempfile
import argparse
import subprocess
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
from google.cloud import speech
import google.generativeai as genai

# Configure paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = None  # Will be set during initialization
CONFIG_PATH = None  # Will be set during initialization

# Sample rate for audio
SAMPLE_RATE = 16000
CHANNELS = 1
LANGUAGE_CODES = {
    "en": "en-US",
    "ne": "ne-NP"
}

# Queue for audio processing
audio_queue = queue.Queue()
response_queue = queue.Queue()

# Initialize languages
input_language = "ne"  # Default input language
output_language = "ne"  # Default output language

# Configure logging
def log(message, level="INFO"):
    """Log messages with timestamp"""
    print(f"[{level}] {message}")

def check_requirements():
    """Check and install required packages"""
    required_packages = [
        "sounddevice",
        "soundfile",
        "numpy",
        "google-cloud-speech",
        "google-generativeai",
        "TTS==0.22.0"
    ]
    
    for package in required_packages:
        try:
            module_name = package.split("==")[0].replace("-", "_")
            __import__(module_name.replace("-", "_"))
            log(f"{module_name} is already installed")
        except ImportError:
            log(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            log(f"Installed {package}")
    
    # Check for API keys
    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') and not os.environ.get('GEMINI_API_KEY'):
        log("Warning: Neither GOOGLE_APPLICATION_CREDENTIALS nor GEMINI_API_KEY environment variables are set.", "WARNING")
        log("Speech recognition and Gemini API calls may fail.", "WARNING")
    
    return True

def find_tts_model():
    """Find the best TTS model in the output directory"""
    global MODEL_DIR, CONFIG_PATH
    
    # First look for the best model
    for model_path in OUTPUT_DIR.glob("**/best_model.pth"):
        MODEL_DIR = model_path
        CONFIG_PATH = model_path.parent / "config.json"
        log(f"Found best model at: {MODEL_DIR}")
        log(f"Config path: {CONFIG_PATH}")
        return True
    
    # If no best model, try to find the latest checkpoint
    checkpoints = list(OUTPUT_DIR.glob("**/checkpoint_*.pth"))
    if checkpoints:
        checkpoints.sort()
        MODEL_DIR = checkpoints[-1]
        CONFIG_PATH = MODEL_DIR.parent / "config.json"
        log(f"No best model found. Using checkpoint: {MODEL_DIR}")
        log(f"Config path: {CONFIG_PATH}")
        return True
    
    # If no model found, we'll use Google TTS as fallback
    log("No fine-tuned TTS model found. Will use Google TTS as fallback.", "WARNING")
    return False

def initialize_gemini():
    """Initialize the Gemini API"""
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        log("GEMINI_API_KEY not found in environment variables", "ERROR")
        log("Please set the GEMINI_API_KEY environment variable", "ERROR")
        return False
    
    genai.configure(api_key=api_key)
    
    # Set up the model
    try:
        global gemini_model
        gemini_model = genai.GenerativeModel('gemini-pro')
        log("Gemini API initialized successfully")
        return True
    except Exception as e:
        log(f"Error initializing Gemini API: {e}", "ERROR")
        return False

def initialize_speech_recognition():
    """Initialize Google Cloud Speech Recognition"""
    try:
        global speech_client
        speech_client = speech.SpeechClient()
        log("Speech recognition initialized")
        return True
    except Exception as e:
        log(f"Error initializing speech recognition: {e}", "ERROR")
        return False

def record_audio(duration=5, device=None):
    """Record audio from microphone"""
    log(f"Recording for {duration} seconds...")
    
    # Record audio
    recording = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        device=device
    )
    
    # Wait for recording to complete
    sd.wait()
    
    log("Recording complete")
    return recording

def listen_continuously(stop_event, duration=3):
    """Listen continuously for voice input"""
    log("Listening for speech...")
    
    while not stop_event.is_set():
        # Record audio
        recording = record_audio(duration)
        
        # Convert to float32 if needed
        if recording.dtype != np.float32:
            recording = recording.astype(np.float32)
        
        # Check if there's audio (not just silence)
        if np.abs(recording).mean() > 0.01:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, recording, SAMPLE_RATE)
                log(f"Saved recording to {temp_file.name}")
                
                # Add to processing queue
                audio_queue.put(temp_file.name)
        else:
            log("Silence detected, continuing to listen...")
            
        # Small pause to prevent CPU overuse
        time.sleep(0.1)

def process_audio_files(stop_event):
    """Process audio files from the queue"""
    while not stop_event.is_set() or not audio_queue.empty():
        try:
            # Get file from queue with a timeout
            audio_file = audio_queue.get(timeout=1)
            
            # Transcribe the audio
            transcript = transcribe_audio(audio_file)
            
            if transcript:
                log(f"Transcribed: {transcript}")
                
                # Process with Gemini
                response = query_gemini(transcript)
                
                if response:
                    # Add to response queue
                    response_queue.put((transcript, response))
                    
                    # Speak the response
                    speak_response(response)
            else:
                log("No transcript detected")
                
            # Clean up the temporary file
            try:
                os.unlink(audio_file)
            except:
                pass
                
        except queue.Empty:
            # No audio to process
            time.sleep(0.1)
            continue
        except Exception as e:
            log(f"Error processing audio: {e}", "ERROR")

def transcribe_audio(audio_file):
    """Transcribe audio using Google Speech Recognition"""
    try:
        # Read the audio file
        with open(audio_file, "rb") as f:
            content = f.read()
        
        # Configure the speech recognition request
        audio = speech.RecognitionAudio(content=content)
        
        # Try first with Nepali
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=LANGUAGE_CODES[input_language],
            enable_automatic_punctuation=True
        )
        
        # Perform speech recognition
        response = speech_client.recognize(config=config, audio=audio)
        
        # Extract transcript
        if response.results:
            return response.results[0].alternatives[0].transcript
            
        # If no results with primary language, try English as fallback
        if input_language != "en":
            log("No results with primary language, trying English...")
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=SAMPLE_RATE,
                language_code=LANGUAGE_CODES["en"],
                enable_automatic_punctuation=True
            )
            
            response = speech_client.recognize(config=config, audio=audio)
            if response.results:
                return response.results[0].alternatives[0].transcript
        
        return None
        
    except Exception as e:
        log(f"Error transcribing audio: {e}", "ERROR")
        return None

def query_gemini(text):
    """Send query to Gemini and get response"""
    try:
        # Create a context-aware prompt
        language_context = "in Nepali" if output_language == "ne" else "in English"
        prompt = f"""
        The user is speaking {input_language}. Please respond {language_context}.
        User query: {text}
        
        Provide a helpful, concise response. If the query is in Nepali, respond in Nepali.
        If the query is in English, respond in English unless specifically requested to respond in Nepali.
        """
        
        # Send to Gemini
        response = gemini_model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        log(f"Error querying Gemini: {e}", "ERROR")
        if output_language == "ne":
            return "माफ गर्नुहोस्, मैले तपाईंको प्रश्न बुझ्न सकिनँ। कृपया फेरि प्रयास गर्नुहोस्।"
        else:
            return "I'm sorry, I couldn't understand your question. Please try again."

def speak_response(text):
    """Speak the response using TTS"""
    try:
        # Check if we have a fine-tuned model
        if MODEL_DIR and CONFIG_PATH:
            # Use our fine-tuned model
            log("Using fine-tuned TTS model")
            
            # Save response to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                output_file = temp_file.name
            
            # Call TTS synthesize
            cmd = [
                sys.executable, "-m", "TTS.bin.synthesize",
                "--model_path", str(MODEL_DIR),
                "--config_path", str(CONFIG_PATH),
                "--text", text,
                "--out_path", output_file
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
        else:
            # Use Google TTS as fallback
            log("Using Google TTS fallback")
            from gtts import gTTS
            
            # Determine language for gTTS
            gtts_lang = "hi" if output_language == "ne" else "en"  # Hindi as proxy for Nepali
            
            # Generate speech
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                output_file = temp_file.name
                
            tts = gTTS(text=text, lang=gtts_lang)
            tts.save(output_file)
        
        # Play the audio
        log(f"Playing response from {output_file}")
        data, fs = sf.read(output_file)
        sd.play(data, fs)
        sd.wait()
        
        # Clean up
        try:
            os.unlink(output_file)
        except:
            pass
            
    except Exception as e:
        log(f"Error speaking response: {e}", "ERROR")

def change_language(lang_code):
    """Change the input and output language"""
    global input_language, output_language
    
    if lang_code in LANGUAGE_CODES:
        input_language = lang_code
        output_language = lang_code
        log(f"Changed language to {LANGUAGE_CODES[lang_code]}")
    else:
        log(f"Unsupported language code: {lang_code}", "ERROR")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Nepali TTS-LLM Pipeline")
    parser.add_argument(
        "--language", "-l",
        choices=["ne", "en"],
        default="ne",
        help="Language for input and output (default: ne)"
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=5,
        help="Duration of each recording in seconds (default: 5)"
    )
    args = parser.parse_args()
    
    # Set language
    change_language(args.language)
    
    # Check requirements
    if not check_requirements():
        log("Failed to satisfy requirements", "ERROR")
        return
    
    # Find TTS model
    find_tts_model()
    
    # Initialize APIs
    if not initialize_gemini():
        log("Failed to initialize Gemini API", "ERROR")
        return
        
    if not initialize_speech_recognition():
        log("Failed to initialize speech recognition", "ERROR")
        return
    
    # Start listening thread
    stop_event = threading.Event()
    listen_thread = threading.Thread(
        target=listen_continuously,
        args=(stop_event, args.duration)
    )
    listen_thread.daemon = True
    listen_thread.start()
    
    # Start processing thread
    process_thread = threading.Thread(
        target=process_audio_files,
        args=(stop_event,)
    )
    process_thread.daemon = True
    process_thread.start()
    
    log("=== Nepali Voice Assistant Started ===")
    log("Press Ctrl+C to exit")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log("Stopping...")
        stop_event.set()
        listen_thread.join(timeout=2)
        process_thread.join(timeout=2)
        log("Voice assistant stopped")

if __name__ == "__main__":
    main()