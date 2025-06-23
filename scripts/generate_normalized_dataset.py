import os
import soundfile as sf
import numpy as np
from tqdm import tqdm
import random

# --- Configuration ---
# IMPORTANT: Confirm these paths are correct for your system
CLEAN_DI_PATH = "/Users/kaimikkelsen/SCNet_guitar/data/audio_mono-pickup_mix"
HUM_PATH = "/Users/kaimikkelsen/SCNet_guitar/data/recorded_hums"
OUTPUT_DATASET_ROOT = "/Users/kaimikkelsen/SCNet_guitar/data/guitar_hum_dataset_final_random_config" # New output folder name

RANDOM_SDR_RANGE_DB = (2.0, 15.0) # Min and Max SDR in dB for random generation. Hum is always quieter.
# NUM_RANDOM_SDRS_PER_PAIR is no longer needed as each clean-hum pair generates only one random SDR mix

TRAIN_SPLIT_RATIO = 0.7
VALID_SPLIT_RATIO = 0.1
TEST_SPLIT_RATIO = 0.2 # Remaining after train/valid. Sum of ratios should be 1.0

# Set random seed for reproducibility of file splits and random hum/SDR selection
random.seed(42)

# --- Helper function to ensure audio is stereo ---
def ensure_stereo(audio_data):
    """
    Converts audio data to a stereo format (samples, 2).
    If input is mono (1D or (samples, 1)), it duplicates the channel.
    If input is multi-channel (>2 channels), it averages to mono then duplicates.
    Assumes (samples,) or (samples, channels) format.
    """
    if audio_data.ndim == 1:  # Mono (samples,)
        # Duplicate channel to create stereo (samples, 2)
        return np.stack([audio_data, audio_data], axis=-1)
    elif audio_data.ndim == 2:
        if audio_data.shape[1] == 1: # Mono with explicit channel dim (samples, 1)
            return np.stack([audio_data[:, 0], audio_data[:, 0]], axis=-1)
        elif audio_data.shape[1] == 2: # Already stereo (samples, 2)
            return audio_data
        else: # More than 2 channels, average to mono then duplicate
            mono_audio = np.mean(audio_data, axis=1)
            return np.stack([mono_audio, mono_audio], axis=-1)
    else:
        raise ValueError(f"Unsupported audio data dimension: {audio_data.ndim}. Expected 1 or 2.")

def create_combined_dataset():
    """
    Generates a new dataset by combining clean DI and hum audio files.
    Each clean DI track is assigned one random hum and combined at one random SDR.
    Ensures normalization, proper train/valid/test splits, and stereo output.
    Shorter files are looped to match longer ones.
    """
    print(f"Starting new dataset generation at: {OUTPUT_DATASET_ROOT}\n")

    # 1. Load all clean DI and hum file paths
    clean_files = sorted([f for f in os.listdir(CLEAN_DI_PATH) if f.endswith('.wav')])
    hum_files = sorted([f for f in os.listdir(HUM_PATH) if f.endswith('.wav')])

    if not clean_files:
        print(f"Error: No clean DI files found in {CLEAN_DI_PATH}. Please check the path.")
        return
    if not hum_files:
        print(f"Error: No hum files found in {HUM_PATH}. Please check the path.")
        return

    # Confirm counts
    print(f"Confirmed: Found {len(clean_files)} guitar clean DI files at {CLEAN_DI_PATH}")
    print(f"Confirmed: Found {len(hum_files)} hum files at {HUM_PATH}")

    # 2. Split clean DI files for train/valid/test to prevent data leakage
    # Random seed is set globally for reproducibility of this split and later random choices
    random.shuffle(clean_files) 

    num_clean = len(clean_files)
    num_train = int(num_clean * TRAIN_SPLIT_RATIO)
    num_valid = int(num_clean * VALID_SPLIT_RATIO)
    num_test = num_clean - num_train - num_valid # Test gets the rest

    train_clean_files = clean_files[:num_train]
    valid_clean_files = clean_files[num_train : num_train + num_valid]
    test_clean_files = clean_files[num_train + num_valid :]

    splits = {
        'train': train_clean_files,
        'valid': valid_clean_files,
        'test': test_clean_files
    }
    
    print(f"\nDataset split based on clean DI files:")
    print(f"  Train: {len(train_clean_files)} files")
    print(f"  Valid: {len(valid_clean_files)} files")
    print(f"  Test:  {len(test_clean_files)} files")

    # 3. Create output directories for the new dataset
    for split_name in splits.keys():
        os.makedirs(os.path.join(OUTPUT_DATASET_ROOT, split_name), exist_ok=True)

    # 4. Pre-load all hum audio, ensuring they are stereo
    print("\nPre-loading hum audio files (converting to stereo)...")
    hum_audios = []
    hum_srs = []
    for h_file in tqdm(hum_files, desc="Loading hums"):
        hum_path_full = os.path.join(HUM_PATH, h_file)
        hum_audio, sr = sf.read(hum_path_full, dtype='float32')
        hum_audios.append(ensure_stereo(hum_audio)) # Ensure hums are stereo
        hum_srs.append(sr)

    # Determine target sample rate. Assume consistency across all inputs.
    target_sr = hum_srs[0] if hum_srs else None
    if not target_sr:
        print("Error: Could not determine a common sample rate from hum files. Aborting.")
        return
    print(f"Using target sample rate for all generated audio: {target_sr} Hz")

    # 5. Process and combine audio files
    # Total tracks calculation: 1 mix per clean file (each with 1 random hum, 1 random SDR)
    total_tracks_to_generate = sum(len(files) for files in splits.values()) # This is simply the total number of clean DI files
    print(f"\nTotal tracks to generate: {total_tracks_to_generate}")
    progress_bar = tqdm(total=total_tracks_to_generate, desc="Generating tracks")

    for split_name, clean_list in splits.items():
        for clean_file_name in clean_list:
            clean_file_path = os.path.join(CLEAN_DI_PATH, clean_file_name)
            
            try:
                clean_audio, sr_clean = sf.read(clean_file_path, dtype='float32')
                clean_audio = ensure_stereo(clean_audio) # Ensure clean is stereo

                if sr_clean != target_sr:
                    tqdm.write(f"Warning: Sample rate mismatch for clean file '{clean_file_name}'. Expected {target_sr}, got {sr_clean}. Skipping this clean file.")
                    progress_bar.update(1) # Still update progress for this skipped file
                    continue

                # --- Randomly select one hum file for this clean track ---
                random_hum_idx = random.randint(0, len(hum_audios) - 1)
                hum_audio_preloaded = hum_audios[random_hum_idx]
                hum_file_name = hum_files[random_hum_idx] # Get original hum filename for naming

                if hum_srs[random_hum_idx] != target_sr:
                    tqdm.write(f"Warning: Sample rate mismatch for randomly selected hum file '{hum_file_name}'. Expected {target_sr}, got {hum_srs[random_hum_idx]}. Skipping combinations for this clean file.")
                    progress_bar.update(1) # Still update progress for this skipped file
                    continue

                # --- Loop shorter audio to match the longer one ---
                len_clean = clean_audio.shape[0]
                len_hum = hum_audio_preloaded.shape[0]

                if len_clean > len_hum:
                    current_clean_audio = clean_audio
                    target_len = len_clean
                    # Loop hum
                    repetitions = int(np.ceil(target_len / len_hum))
                    looped_hum_audio = np.tile(hum_audio_preloaded, (repetitions, 1)) 
                    current_hum_audio = looped_hum_audio[:target_len, :]
                else: # len_hum >= len_clean
                    current_hum_audio = hum_audio_preloaded
                    target_len = len_hum
                    # Loop clean
                    repetitions = int(np.ceil(target_len / len_clean))
                    looped_clean_audio = np.tile(clean_audio, (repetitions, 1))
                    current_clean_audio = looped_clean_audio[:target_len, :]

                # Calculate power for scaling across all channels. Adding epsilon to prevent division by zero for silent files.
                current_clean_power = np.sum(current_clean_audio**2) + 1e-8
                current_hum_power = np.sum(current_hum_audio**2) + 1e-8


                # --- Generate ONE random SDR for this specific (clean, assigned_hum) combination ---
                sdr_db = random.uniform(RANDOM_SDR_RANGE_DB[0], RANDOM_SDR_RANGE_DB[1])
                
                # Calculate Hum Scaling Factor for target SDR
                target_hum_power = current_clean_power / (10**(sdr_db / 10.0))
                hum_scaling_factor = np.sqrt(target_hum_power / current_hum_power)

                hum_scaled = current_hum_audio * hum_scaling_factor

                # Create Temporary Mixture (before final global normalization)
                temp_mixture = current_clean_audio + hum_scaled

                # Global Normalization for the final mixture and components
                max_abs_temp_mixture = np.max(np.abs(temp_mixture))
                overall_normalization_factor = 1.0 / (max_abs_temp_mixture + 1e-8)

                final_clean = current_clean_audio * overall_normalization_factor
                final_hum = hum_scaled * overall_normalization_factor
                final_mixture = temp_mixture * overall_normalization_factor


                # 6. Save generated files (will be saved as stereo due to (samples, 2) shape)
                clean_id = os.path.splitext(clean_file_name)[0]
                hum_id = os.path.splitext(hum_file_name)[0]
                
                # Format SDR value for folder name: e.g., "5_2dB", "neg_3_7dB"
                sdr_str = f"{sdr_db:.1f}".replace('.', '_').replace('-', 'neg') 
                
                # Output folder name (no 'vX' suffix needed as there's only one mix per clean track)
                output_folder_name = f"{clean_id}_{hum_id}_sdr_{sdr_str}dB"
                output_folder_path = os.path.join(OUTPUT_DATASET_ROOT, split_name, output_folder_name)
                os.makedirs(output_folder_path, exist_ok=True)

                sf.write(os.path.join(output_folder_path, 'clean.wav'), final_clean, target_sr)
                sf.write(os.path.join(output_folder_path, 'hum.wav'), final_hum, target_sr)
                sf.write(os.path.join(output_folder_path, 'mixture.wav'), final_mixture, target_sr)
                
                progress_bar.update(1) # Update progress bar for each generated track

            except Exception as e:
                tqdm.write(f"Error processing clean file '{clean_file_name}' with a random hum: {e}")
                progress_bar.update(1) # Still update progress for this skipped file
                continue # Move to next clean file in the list

    progress_bar.close()
    print(f"\nDataset generation complete. Total tracks generated: {total_tracks_to_generate}")
    print(f"New dataset saved to: {OUTPUT_DATASET_ROOT}")


if __name__ == "__main__":
    # Ensure numpy operations use float64 for better precision during intermediate calculations
    # before converting back to float32 for soundfile.
    np.set_printoptions(precision=4)
    print(f"NumPy default float type for calculations: {np.float64}")
    
    create_combined_dataset()