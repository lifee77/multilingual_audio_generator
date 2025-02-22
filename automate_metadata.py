"""This Python script that scans a specified directory for files (here, filtering for .wav files), 
then creates a metadata.csv file with three columns: filename, speaker_label, and language_label. 
By default, each file will be labeled as "Not_Jeevan" for speaker and "Not_English" for language.

"""
import os
import csv

# Set the directory to search for audio files
input_directory = "data/audio_data"  # Change this to your directory if needed

# Set the output CSV filename
output_csv = "data/metadata.csv"

# List all files in the input directory and filter for .wav files
filenames = [f for f in os.listdir(input_directory) if f.lower().endswith('.wav')]

# Open the CSV file for writing
with open(output_csv, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    # Write the header
    writer.writerow(["filename", "speaker_label", "language_label"])
    
    # Write each file's metadata with default values
    for filename in filenames:
        writer.writerow([filename, "Not_Jeevan", "Not_English"])

print(f"Metadata saved to {output_csv}")
