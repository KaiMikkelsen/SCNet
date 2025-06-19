import os
import soundfile as sf
import numpy as np
import shutil

# --- Configuration ---
# Point this to the base folder where your train, valid, and test splits are located
BASE_DATASET_FOLDER = "/Users/kaimikkelsen/SCNet_guitar/data/guitar_hum_dataset_split"

# Subfolders to check
SUBFOLDERS = ["train", "valid", "test"]

# --- Script ---

def verify_and_fix_dataset_lengths():
    print(f"Starting verification and fixing for dataset in: {BASE_DATASET_FOLDER}")

    for split_folder_name in SUBFOLDERS:
        split_path = os.path.join(BASE_DATASET_FOLDER, split_folder_name)
        
        if not os.path.exists(split_path):
            print(f"Skipping {split_path} as it does not exist.")
            continue

        print(f"\nProcessing split: {split_folder_name}")
        track_folders = [
            os.path.join(split_path, d)
            for d in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, d))
        ]

        if not track_folders:
            print(f"No track folders found in {split_path}.")
            continue

        for track_folder in track_folders:
            mixture_path = os.path.join(track_folder, "mixture.wav")
            clean_path = os.path.join(track_folder, "clean.wav")
            hum_path = os.path.join(track_folder, "hum.wav")

            files_to_check = {
                "mixture": mixture_path,
                "clean": clean_path,
                "hum": hum_path
            }

            lengths = {}
            sample_rates = {}
            all_files_exist = True
            
            for name, path in files_to_check.items():
                if not os.path.exists(path):
                    print(f"  WARNING: Missing {name} file in {track_folder}. Skipping this track.")
                    all_files_exist = False
                    break # Skip this track if any file is missing

                try:
                    info = sf.info(path)
                    lengths[name] = info.frames # Number of samples
                    sample_rates[name] = info.samplerate
                except Exception as e:
                    print(f"  ERROR: Could not read {name} file {path} for info: {e}. Skipping this track.")
                    all_files_exist = False
                    break # Skip this track if loading info fails

            if not all_files_exist:
                continue

            # Check for sample rate consistency first
            if len(set(sample_rates.values())) > 1:
                print(f"  ERROR: Sample rate mismatch in {os.path.basename(track_folder)}. All files must have the same sample rate.")
                for name, sr in sample_rates.items():
                    print(f"    - {name}: {sr} Hz")
                print("    -> This must be fixed in the dataset generation step. Skipping this track for length fix.")
                continue # Do not attempt to fix lengths if sample rates are inconsistent

            # If sample rates are consistent, check and fix lengths
            if len(set(lengths.values())) > 1:
                min_length = min(lengths.values())
                
                print(f"  DISCREPANCY in {os.path.basename(track_folder)}:")
                for name, length in lengths.items():
                    print(f"    - {name}: {length} samples")
                print(f"    -> Fixing by truncating all to {min_length} samples.")

                try:
                    # Load audio data using soundfile (using the consistent sample rate)
                    mix_data, sr = sf.read(mixture_path)
                    clean_data, _ = sf.read(clean_path)
                    hum_data, _ = sf.read(hum_path)

                    # Truncate to the minimum length
                    mix_data = mix_data[:min_length]
                    clean_data = clean_data[:min_length]
                    hum_data = hum_data[:min_length]

                    # Overwrite the original files
                    sf.write(mixture_path, mix_data, sr)
                    sf.write(clean_path, clean_data, sr)
                    sf.write(hum_path, hum_data, sr)
                    print(f"    Fixed {os.path.basename(track_folder)}: All files now {min_length} samples.")

                except Exception as e:
                    print(f"  ERROR during fixing for {os.path.basename(track_folder)}: {e}")
            else:
                # print(f"  {os.path.basename(track_folder)}: All files have consistent length ({list(lengths.values())[0]} samples).")
                pass # Suppress constant success messages for cleaner output

    print("\nVerification and fixing complete.")
    print("Review any 'ERROR' or 'WARNING' messages for tracks that couldn't be fixed automatically.")

if __name__ == "__main__":
    # Ensure soundfile is installed
    try:
        import soundfile as sf
    except ImportError:
        print("soundfile library not found. Please install it: pip install soundfile")
        print("\nNote: soundfile requires the 'libsndfile' C library to be installed on your system.")
        print("  - On macOS (using Homebrew): brew install libsndfile")
        print("  - On Linux (Debian/Ubuntu): sudo apt-get update && sudo apt-get install libsndfile1")
        print("  - On Windows: Download binaries from 'libsndfile' website or use Chocolatey 'choco install libsndfile'")
        exit(1)

    verify_and_fix_dataset_lengths()