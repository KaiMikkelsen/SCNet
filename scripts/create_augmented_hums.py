import soundfile as sf
import numpy as np
import os
import random
from pedalboard import Pedalboard, Gain, HighpassFilter, LowpassFilter, PitchShift, Distortion, Chorus, Phaser, Delay, Reverb, Compressor
# Ensure LadderFilter is imported if you use it, or remove it from the list if not needed

# --- Configuration ---
INPUT_HUMS_DIR = '/Users/kaimikkelsen/SCNet_guitar/data/recorded_hums'
OUTPUT_AUGMENTED_HUMS_DIR = '/Users/kaimikkelsen/SCNet_guitar/data/augmented_hums' # New output directory for stereo hums

NUM_AUGMENTATIONS_PER_HUM = 60 
SAMPLE_RATE = 44100 # Ensure this matches your model's expected SR

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_AUGMENTED_HUMS_DIR, exist_ok=True)
print(f"Output directory created/ensured: {OUTPUT_AUGMENTED_HUMS_DIR}")

# --- Define Augmentation Functions (adapted for stereo processing) ---

# All pedalboard effects handle stereo automatically if the input is stereo.
# So, the core logic within these functions doesn't change much, just the input/output expectation.

def apply_core_hum_variations(audio_array):
    """Applies fundamental hum variations (gain, pitch, subtle EQ)."""
    board = Pedalboard([
        Gain(gain_db=random.uniform(-15.0, 15.0)),
        PitchShift(semitones=random.uniform(-1.0, 1.0)),
        HighpassFilter(cutoff_frequency_hz=random.uniform(20.0, 180.0)),
        LowpassFilter(cutoff_frequency_hz=random.uniform(500.0, 7000.0)),
    ])
    return board(audio_array, SAMPLE_RATE)

def apply_guitar_amp_like_distortion(audio_array):
    """Applies heavy, guitar-like distortion."""
    if random.random() < 0.9:
        drive_db = random.uniform(15.0, 70.0)
        mix = random.uniform(0.7, 1.0)
        board = Pedalboard([Distortion(drive_db=drive_db)])
        return board(audio_array, SAMPLE_RATE)
    return audio_array

def apply_random_modulation_effects(audio_array):
    """Adds chorus or phaser effects, which can also affect hum."""
    board_effects = []
    if random.random() < 0.4:
        board_effects.append(Chorus(
            rate_hz=random.uniform(0.1, 5.0), depth=random.uniform(0.1, 0.8), mix=random.uniform(0.2, 0.7)))
    if random.random() < 0.4:
        board_effects.append(Phaser(
            rate_hz=random.uniform(0.1, 3.0), depth=random.uniform(0.1, 0.8), mix=random.uniform(0.2, 0.7)))
    
    random.shuffle(board_effects)
    if board_effects:
        board = Pedalboard(board_effects)
        return board(audio_array, SAMPLE_RATE)
    return audio_array

def apply_random_time_based_effects(audio_array):
    """Adds delay or reverb effects."""
    board_effects = []
    if random.random() < 0.3:
        board_effects.append(Delay(
            delay_seconds=random.uniform(0.1, 0.5), mix=random.uniform(0.1, 0.4), feedback=random.uniform(0.0, 0.5)))
    if random.random() < 0.5:
        board_effects.append(Reverb(
            room_size=random.uniform(0.1, 0.9), damping=random.uniform(0.1, 0.9), wet_level=random.uniform(0.1, 0.6)))
    
    random.shuffle(board_effects)
    if board_effects:
        board = Pedalboard(board_effects)
        return board(audio_array, SAMPLE_RATE)
    return audio_array

def add_random_noise_and_compression(audio_array):
    """Adds white noise and dynamic range compression. Now handles stereo noise."""
    augmented_audio = audio_array.copy() # Make a copy to avoid modifying original array

    # Add white noise
    if random.random() < 0.6:
        # Determine noise level relative to hum RMS, now for stereo
        hum_rms = np.sqrt(np.mean(audio_array**2)) # RMS over all samples and channels
        noise_rms = hum_rms * (10**(random.uniform(-40.0, -15.0) / 20.0))
        
        # Generate stereo white noise
        noise = np.random.normal(0, noise_rms, audio_array.shape).astype(np.float32)
        augmented_audio = augmented_audio + noise
        
        max_val = np.max(np.abs(augmented_audio))
        if max_val > 1.0: 
            augmented_audio = augmented_audio / max_val

    # Apply Compressor
    if random.random() < 0.7:
        board = Pedalboard([
            Compressor(
                threshold_db=random.uniform(-30.0, -10.0),
                ratio=random.uniform(2.0, 10.0),
                attack_ms=random.uniform(1.0, 20.0),
                release_ms=random.uniform(50.0, 500.0)
            )
        ])
        augmented_audio = board(augmented_audio, SAMPLE_RATE)

    return augmented_audio

def combine_all_hum_augmentations(audio_array):
    """Applies a full sequence of hum-specific and guitar-like augmentations."""
    augmented_audio = audio_array.copy()

    # Apply core hum variations first
    augmented_audio = apply_core_hum_variations(augmented_audio)

    # effect_pipeline = [
    #     #apply_guitar_amp_like_distortion,
    #     #apply_random_modulation_effects,
    #     apply_random_time_based_effects,
    #     add_random_noise_and_compression
    # ]
    
    # random.shuffle(effect_pipeline)

    # for effect_func in effect_pipeline:
        # augmented_audio = effect_func(augmented_audio)

    return np.clip(augmented_audio, -1.0, 1.0).astype(np.float32)

# --- Main Augmentation Loop ---

hum_files = [f for f in os.listdir(INPUT_HUMS_DIR) if f.lower().endswith('.wav')]

if not hum_files:
    print(f"Error: No WAV files found in {INPUT_HUMS_DIR}. Please check the path and file extensions.")
else:
    print(f"Found {len(hum_files)} hum files in {INPUT_HUMS_DIR}.")
    for hum_filename in hum_files:
        hum_path = os.path.join(INPUT_HUMS_DIR, hum_filename)
        
        try:
            hum_audio, sr = sf.read(hum_path, dtype='float32')
            
            if sr != SAMPLE_RATE:
                print(f"Warning: {hum_filename} has sample rate {sr}Hz, expected {SAMPLE_RATE}Hz. "
                      "Pedalboard effects will internally resample if necessary.")
            
            # --- CRITICAL CHANGE FOR STEREO ---
            if hum_audio.ndim == 1:
                # Convert mono to stereo by duplicating the channel
                hum_audio = np.stack([hum_audio, hum_audio], axis=1) # Shape (num_samples, 2)
                print(f"Converted {hum_filename} from mono to stereo.")
            elif hum_audio.ndim > 2: # Handle cases with more than 2 channels, e.g., 5.1, by taking first two
                hum_audio = hum_audio[:, :2]
                print(f"Warning: {hum_filename} has more than 2 channels. Taking first two channels for stereo.")
            
            # Ensure the hum is long enough if you plan to take segments
            # For this offline augmentation, we process the whole file.
            # However, if your hums are super short, consider padding or looping them *here*
            # to make them usable for arbitrary length segments later in the DataLoader.

            print(f"Processing {hum_filename} ({NUM_AUGMENTATIONS_PER_HUM} augmentations)...")

            for i in range(NUM_AUGMENTATIONS_PER_HUM):
                augmented_hum = combine_all_hum_augmentations(hum_audio)

                output_filename = f"{os.path.splitext(hum_filename)[0]}_aug_{i:03d}.wav"
                output_path = os.path.join(OUTPUT_AUGMENTED_HUMS_DIR, output_filename)
                
                # Save as stereo (soundfile handles this automatically if array is (N, 2))
                sf.write(output_path, augmented_hum, sr) 
                
            print(f"Finished augmenting {hum_filename}.")

        except sf.LibsndfileError as e:
            print(f"Error reading {hum_filename}: {e}. Skipping this file.")
        except Exception as e:
            print(f"An unexpected error occurred processing {hum_filename}: {e}. Skipping this file.")

    total_augmented = len(hum_files) * NUM_AUGMENTATIONS_PER_HUM
    print(f"\nAll hum augmentations complete. Augmented files saved to {OUTPUT_AUGMENTED_HUMS_DIR}")
    print(f"Total original hums processed: {len(hum_files)}")
    print(f"Total augmented hums created: {total_augmented}")