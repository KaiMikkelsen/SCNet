import os
import argparse
import soundfile as sf
import torch
import numpy as np

def convert_audio_channels_for_script(wav_tensor, target_channels=2):
    """
    Converts an audio tensor to the specified number of channels.
    This version is simplified and self-contained for this specific script.
    Assumes wav_tensor is (channels, samples) or (samples,) for mono.
    """
    print(f"  [Channel Conv] Input WAV tensor shape: {wav_tensor.shape}, ndim: {wav_tensor.ndim}")
    print(f"  [Channel Conv] Target channels: {target_channels}")

    if wav_tensor.ndim == 1:
        src_channels = 1
        # Unsqueeze to (1, samples) for consistent processing
        wav_tensor = wav_tensor.unsqueeze(0)
    else: # Assume (channels, samples)
        src_channels = wav_tensor.shape[0]

    print(f"  [Channel Conv] Detected source channels: {src_channels}")

    if src_channels == target_channels:
        print("  [Channel Conv] Channels already match. No conversion needed.")
        return wav_tensor
    elif src_channels == 1 and target_channels > 1:
        print(f"  [Channel Conv] Upmixing mono to {target_channels} channels by duplication.")
        return wav_tensor.expand(target_channels, -1)
    elif src_channels > target_channels:
        print(f"  [Channel Conv] Downmixing from {src_channels} to {target_channels} channels.")
        if target_channels == 1: # Downmix to mono by averaging
            return wav_tensor.mean(dim=0, keepdim=True)
        else: # Downmix by truncation (e.g., 4ch to 2ch)
            return wav_tensor[:target_channels, :]
    else:
        # This covers cases like (2 channels to 3 channels) - not mono, but less than target
        print(f"  [Channel Conv] Upmixing non-mono from {src_channels} to {target_channels} channels by duplication/padding.")
        num_repetitions = (target_channels + src_channels - 1) // src_channels # Ceiling division
        temp_wav = torch.cat([wav_tensor] * num_repetitions, dim=0) # Concatenate along channel dim
        return temp_wav[:target_channels, :] # Truncate to exact requested channels


def process_clean_wav_files(base_dir: str):
    """
    Traverses the directory structure, finds 'clean.wav' files,
    converts them to stereo, and overwrites the original files.

    Args:
        base_dir (str): The root directory containing song subfolders.
    """
    print(f"--- Starting Clean WAV to Stereo Conversion ---")
    print(f"Processing directory: {base_dir}")
    print(f"Files will be converted to stereo (2 channels) and overwritten.")
    print("-" * 50)

    if not os.path.isdir(base_dir):
        print(f"Error: Base directory '{base_dir}' not found or is not a directory.")
        print("Please provide a valid path to your test set root.")
        return

    processed_count = 0
    skipped_count = 0
    error_count = 0

    # Walk through the directory structure
    for root, dirs, files in os.walk(base_dir):
        if 'clean.wav' in files:
            clean_wav_path = os.path.join(root, 'clean.wav')
            print(f"\nProcessing clean.wav: {clean_wav_path}")

            try:
                # Load audio
                data, sample_rate = sf.read(clean_wav_path, dtype='float32')
                print(f"  Loaded audio. Original shape: {data.shape}, sample rate: {sample_rate}")

                # Convert numpy array to torch tensor, ensuring (channels, samples) format
                # sf.read can return (samples,) for mono or (samples, channels) for multi-channel
                if data.ndim == 1:
                    input_wav_tensor = torch.from_numpy(data).float() # (samples,)
                else: # (samples, channels)
                    input_wav_tensor = torch.from_numpy(data.T).float() # Transpose to (channels, samples)

                # Convert to stereo using the helper function
                converted_wav_tensor = convert_audio_channels_for_script(input_wav_tensor, target_channels=2)
                
                # Convert back to numpy array for soundfile.write
                # Ensure it's (samples, channels) for sf.write
                output_data_np = converted_wav_tensor.cpu().numpy().T 

                # Save the converted audio, overwriting the original
                sf.write(clean_wav_path, output_data_np, sample_rate)
                print(f"  Successfully converted and saved: {clean_wav_path} (new shape: {output_data_np.shape})")
                processed_count += 1

            except Exception as e:
                print(f"  Error processing '{clean_wav_path}': {e}")
                error_count += 1
        else:
            # Optionally print if a folder doesn't contain clean.wav but is a song folder
            # print(f"Skipping folder {root}: 'clean.wav' not found.")
            pass # Keep it quiet for folders without clean.wav

    print("\n--- Conversion Process Summary ---")
    print(f"Total 'clean.wav' files processed successfully: {processed_count}")
    print(f"Total files/folders skipped (e.g., no 'clean.wav'): {skipped_count}") # Note: this count is rough if not explicit.
    print(f"Total files failed to process: {error_count}")
    print("Clean WAV to stereo conversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert all 'clean.wav' files in a specified test set to stereo (2 channels)."
    )
    parser.add_argument(
        '--test_set_root',
        type=str,
        required=True,
        help='The root directory of your test set, e.g., /home/kaim/projects/def-ichiro/kaim/data/guitar_hum_dataset_split/test'
    )

    args = parser.parse_args()

    process_clean_wav_files(args.test_set_root)
