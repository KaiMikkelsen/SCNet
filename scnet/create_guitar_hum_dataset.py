import os
import random
from pydub import AudioSegment
from pydub.utils import make_chunks # Still imported, but won't be used for splitting
import numpy as np

# --- Configuration ---
CLEAN_GUITAR_FOLDER = "/Users/kaimikkelsen/SCNet_guitar/data/audio_mono-pickup_mix"
HUM_FOLDER = "/Users/kaimikkelsen/SCNet_guitar/data/recorded_hums"
OUTPUT_DATASET_FOLDER = "/Users/kaimikkelsen/SCNet_guitar/data/guitar_hum_dataset"

# Desired length of the mixed audio segments in milliseconds
# Set to None to use full length of guitar recording.
SEGMENT_LENGTH_MS = None # <--- THIS IS THE CHANGE

# Hum level range relative to the guitar signal (in dB)
MIN_HUM_LEVEL_DB = -35
MAX_HUM_LEVEL_DB = -15

# --- Script ---

def create_dataset():
    if not os.path.exists(OUTPUT_DATASET_FOLDER):
        os.makedirs(OUTPUT_DATASET_FOLDER)

    clean_guitar_files = [
        os.path.join(CLEAN_GUITAR_FOLDER, f)
        for f in os.listdir(CLEAN_GUITAR_FOLDER)
        if f.endswith((".wav", ".mp3"))
    ]
    hum_files = [
        os.path.join(HUM_FOLDER, f)
        for f in os.listdir(HUM_FOLDER)
        if f.endswith((".wav", ".mp3"))
    ]

    if not clean_guitar_files:
        print(f"No clean guitar recordings found in {CLEAN_GUITAR_FOLDER}")
        return
    if not hum_files:
        print(f"No hum recordings found in {HUM_FOLDER}")
        return

    print(f"Found {len(clean_guitar_files)} clean guitar recordings.")
    print(f"Found {len(hum_files)} hum recordings.")

    dataset_counter = 0
    for guitar_file_path in clean_guitar_files:
        guitar_filename = os.path.basename(guitar_file_path)
        guitar_name_without_ext = os.path.splitext(guitar_filename)[0]

        try:
            clean_guitar_audio = AudioSegment.from_file(guitar_file_path)
        except Exception as e:
            print(f"Could not load guitar file {guitar_file_path}: {e}")
            continue

        # This block now always treats the whole file as one chunk
        if SEGMENT_LENGTH_MS:
            # This part will not be executed if SEGMENT_LENGTH_MS is None
            chunks = make_chunks(clean_guitar_audio, SEGMENT_LENGTH_MS)
        else:
            chunks = [clean_guitar_audio] # Treat the whole file as one chunk

        for i, chunk in enumerate(chunks): # This loop will run only once per guitar file
            # Select a random hum file
            random_hum_file_path = random.choice(hum_files)
            hum_filename = os.path.basename(random_hum_file_path)

            try:
                hum_audio = AudioSegment.from_file(random_hum_file_path)
            except Exception as e:
                print(f"Could not load hum file {random_hum_file_path}: {e}")
                continue

            # Ensure hum audio is long enough for the chunk (which is now the full guitar file)
            if len(hum_audio) < len(chunk):
                # Loop the hum if it's shorter than the guitar segment
                num_repeats = (len(chunk) // len(hum_audio)) + 1
                hum_audio = hum_audio * num_repeats
            hum_audio = hum_audio[:len(chunk)] # Trim to exact length

            # Set a random hum level relative to the guitar signal
            hum_relative_volume_db = random.uniform(MIN_HUM_LEVEL_DB, MAX_HUM_LEVEL_DB)
            
            guitar_loudness = chunk.dBFS
            target_hum_loudness = guitar_loudness + hum_relative_volume_db
            hum_gain_change = target_hum_loudness - hum_audio.dBFS
            adjusted_hum_audio = hum_audio + hum_gain_change

            # Mix the guitar and hum
            mixed_audio = chunk.overlay(adjusted_hum_audio)

            # Create output folder for this specific mix
            # The folder name will no longer include a segment number
            output_folder_name = f"{guitar_name_without_ext}_{os.path.splitext(hum_filename)[0]}"
            
            current_output_dir = os.path.join(OUTPUT_DATASET_FOLDER, output_folder_name)
            os.makedirs(current_output_dir, exist_ok=True)

            # Export the files
            mixture_path = os.path.join(current_output_dir, "mixture.wav")
            clean_path = os.path.join(current_output_dir, "clean.wav")
            hum_path = os.path.join(current_output_dir, "hum.wav")

            try:
                mixed_audio.export(mixture_path, format="wav")
                chunk.export(clean_path, format="wav")
                adjusted_hum_audio.export(hum_path, format="wav")
                print(f"Generated: {output_folder_name}")
                dataset_counter += 1
            except Exception as e:
                print(f"Error exporting files for {output_folder_name}: {e}")

    print(f"\nDataset creation complete. Total {dataset_counter} mixed examples generated in {OUTPUT_DATASET_FOLDER}")

if __name__ == "__main__":
    create_dataset()