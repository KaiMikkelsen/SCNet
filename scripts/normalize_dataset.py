import os
import soundfile as sf
import numpy as np
from tqdm import tqdm # For a progress bar

def normalize_and_remake_mixtures(dataset_root_path):
    """
    Iterates through the dataset, normalizes 'hum.wav' and 'clean.wav'
    such that their sum (the new mixture) stays within [-1.0, 1.0],
    and then recreates/overwrites the 'mixture.wav' file.

    Args:
        dataset_root_path (str): The root path to your dataset,
                                 e.g., '/Users/kaimikkelsen/SCNet_guitar/data/guitar_hum_dataset_split'
    """
    subsets = ['test', 'train', 'valid']

    print(f"Starting dataset preprocessing in: {dataset_root_path}\n")
    print("This script will normalize 'hum' and 'clean' components such that their sum (the new mixture)")
    print("does not exceed a peak amplitude of 1.0, and then overwrite the 'mixture.wav' files.")
    print("--------------------------------------------------------------------------------------")

    for subset in subsets:
        subset_path = os.path.join(dataset_root_path, subset)
        if not os.path.isdir(subset_path):
            print(f"Skipping subset '{subset}': Directory not found at {subset_path}")
            continue

        print(f"\nProcessing subset: {subset}")
        song_folders = sorted([d for d in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, d))])

        for song_folder in tqdm(song_folders, desc=f"Processing songs in '{subset}'"):
            song_path = os.path.join(subset_path, song_folder)
            hum_path = os.path.join(song_path, 'hum.wav')
            clean_path = os.path.join(song_path, 'clean.wav')
            mixture_path = os.path.join(song_path, 'mixture.wav')

            # --- Basic checks for file existence ---
            if not (os.path.exists(hum_path) and os.path.exists(clean_path)):
                tqdm.write(f"Warning: Skipping {song_folder} in {subset}. 'hum.wav' or 'clean.wav' not found.")
                continue

            try:
                # Load audio files as float32
                hum_audio, sr_hum = sf.read(hum_path, dtype='float32')
                clean_audio, sr_clean = sf.read(clean_path, dtype='float32')

                # --- Basic checks for consistency ---
                if sr_hum != sr_clean:
                    tqdm.write(f"Error: Sample rate mismatch in {song_folder}. Skipping. Hum: {sr_hum}, Clean: {sr_clean}")
                    continue
                if hum_audio.shape != clean_audio.shape:
                    min_len = min(len(hum_audio), len(clean_audio))
                    hum_audio = hum_audio[:min_len]
                    clean_audio = clean_audio[:min_len]
                    if hum_audio.ndim > 1: # Handle stereo truncation
                         hum_audio = hum_audio[:min_len, :]
                         clean_audio = clean_audio[:min_len, :]
                    tqdm.write(f"Warning: Length mismatch in {song_folder}. Truncated to {min_len} samples.")

                # --- Normalization Strategy for hum and clean before summing ---
                # Calculate the peak amplitude of their *sum* if they were added as is.
                # This ensures the new_mixture will not clip.
                # np.abs(hum_audio) + np.abs(clean_audio) gives the maximum possible amplitude
                # at any given sample if they were summed.
                max_combined_amplitude = np.max(np.abs(hum_audio) + np.abs(clean_audio))

                # Apply scaling if the combined amplitude exceeds 1.0 (with a small epsilon for safety)
                if max_combined_amplitude > 1.0 + 1e-8:
                    scaling_factor = 1.0 / max_combined_amplitude
                    hum_normalized = hum_audio * scaling_factor
                    clean_normalized = clean_audio * scaling_factor
                else:
                    # No scaling needed if already within bounds
                    hum_normalized = hum_audio
                    clean_normalized = clean_audio

                # --- Recreate the mixture from the normalized components ---
                new_mixture = hum_normalized + clean_normalized

                # --- Save the new mixture, overwriting the old one ---
                sf.write(mixture_path, new_mixture, sr_hum)

            except Exception as e:
                tqdm.write(f"Error processing {song_folder} in {subset}: {e}")
                continue # Continue to the next song even if one fails

    print("\n--------------------------------------------------------------------------------------")
    print("Dataset preprocessing complete. 'mixture.wav' files have been recreated.")
    print("You can now verify the new 'mixture.wav' files for proper normalization.")

if __name__ == "__main__":
    # --- Set your dataset root path here ---
    dataset_root_path = "/Users/kaimikkelsen/SCNet_guitar/data/guitar_hum_dataset_split"

    normalize_and_remake_mixtures(dataset_root_path)