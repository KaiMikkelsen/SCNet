import soundfile as sf
import os
from pathlib import Path
import numpy as np

def convert_mono_to_stereo(file_path):
    """
    Converts a mono WAV file to stereo by duplicating its single channel.
    If the file is already stereo or multi-channel, it remains unchanged.
    """
    try:
        data, samplerate = sf.read(file_path, dtype='float32')

        # Check number of channels
        # soundfile.read returns (samples,) for mono, (samples, channels) for stereo/multi
        if data.ndim == 1:
            # It's mono, duplicate the channel to make it stereo
            stereo_data = np.stack([data, data], axis=-1)
            sf.write(file_path, stereo_data, samplerate)
            print(f"  Converted mono to stereo: {file_path}")
            return True
        elif data.shape[1] == 1:
            # It's technically 2D with 1 channel, also treat as mono
            stereo_data = np.stack([data[:, 0], data[:, 0]], axis=-1)
            sf.write(file_path, stereo_data, samplerate)
            print(f"  Converted mono (2D, 1ch) to stereo: {file_path}")
            return True
        else:
            # Already stereo or multi-channel
            print(f"  Skipping (already stereo/multi-channel): {file_path}")
            return False
    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
        return False

def process_dataset_directory(base_dir):
    """
    Traverses the dataset directory and converts mono WAV files to stereo.
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: Base directory '{base_dir}' not found. Please check the path.")
        return

    sub_dirs = ['test', 'train', 'valid']
    
    total_files_processed = 0
    total_files_converted = 0

    print(f"Starting WAV conversion in: {base_dir}")
    print("-" * 50)

    for sub_dir_name in sub_dirs:
        current_sub_dir_path = base_path / sub_dir_name
        if not current_sub_dir_path.exists():
            print(f"Warning: Subdirectory '{current_sub_dir_path}' not found. Skipping.")
            continue

        print(f"Processing '{sub_dir_name}' directory...")

        # Iterate through immediate subfolders within test/train/valid
        for item in current_sub_dir_path.iterdir():
            if item.is_dir():
                print(f"  Entering folder: {item.name}")
                # Check for the specific WAV files inside these subfolders
                for wav_file_name in ['mixture.wav', 'clean.wav', 'hum.wav']:
                    wav_file_path = item / wav_file_name
                    if wav_file_path.is_file():
                        total_files_processed += 1
                        if convert_mono_to_stereo(wav_file_path):
                            total_files_converted += 1
                    else:
                        print(f"  File not found: {wav_file_path.name} in {item.name}")
        print("-" * 50)

    print("\nConversion Summary:")
    print(f"Total WAV files checked: {total_files_processed}")
    print(f"Total mono WAV files converted to stereo: {total_files_converted}")
    print("Process complete.")


if __name__ == "__main__":
    # Define your base dataset directory
    # IMPORTANT: Adjust this path to your actual dataset location
    base_dataset_directory = "/Users/kaimikkelsen/SCNet_guitar/data/guitar_hum_dataset_with_amps"

    process_dataset_directory(base_dataset_directory)