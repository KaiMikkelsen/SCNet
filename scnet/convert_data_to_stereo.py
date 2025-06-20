import os
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from tqdm import tqdm # For a progress bar

def ensure_stereo_wav(file_path):
    """
    Checks if a WAV file is stereo. If not, converts it to stereo
    by duplicating the mono channel and overwriting the original file.
    """
    try:
        audio = AudioSegment.from_wav(file_path)

        if audio.channels == 1:
            print(f"  -> Converting mono file to stereo: {os.path.basename(file_path)}")
            # Duplicate the mono channel to create a stereo file
            stereo_audio = audio.set_channels(2)
            stereo_audio.export(file_path, format="wav")
            print(f"  -> Converted {os.path.basename(file_path)} to stereo.")
        elif audio.channels == 2:
            pass # Already stereo, no action needed
        else:
            print(f"  WARNING: File {os.path.basename(file_path)} has {audio.channels} channels. "
                  "This script only handles mono (1) or stereo (2) and will not modify it.")

    except CouldntDecodeError:
        print(f"  ERROR: Could not decode {os.path.basename(file_path)}. Is it a valid WAV file?")
    except Exception as e:
        print(f"  ANOTHER ERROR processing {os.path.basename(file_path)}: {e}")

def process_dataset_for_stereo(dataset_root_dir):
    """
    Traverses the dataset directory, identifies all .wav files,
    and ensures they are stereo.
    """
    print(f"Starting stereo conversion check for dataset: {dataset_root_dir}")
    wav_files_found = []
    for root, _, files in os.walk(dataset_root_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_files_found.append(os.path.join(root, file))

    if not wav_files_found:
        print(f"No WAV files found in {dataset_root_dir}. Exiting.")
        return

    print(f"Found {len(wav_files_found)} WAV files. Checking/converting...")
    for wav_file in tqdm(wav_files_found, desc="Processing WAV files"):
        ensure_stereo_wav(wav_file)
    
    print("\nStereo conversion check complete for all WAV files.")


if __name__ == "__main__":
    # IMPORTANT: Set this to the path of your 'guitar_hum_dataset_split' directory
    # Based on your screenshot, it would be the full path to that folder.
    # Example: dataset_path = "/path/to/your/project/guitar_hum_dataset_split"
    # or if this script is in the parent directory of guitar_hum_dataset_split:
    dataset_path = "/Users/kaimikkelsen/SCNet_guitar/data/guitar_hum_dataset_split" 
    
    # Make sure the path exists
    if not os.path.isdir(dataset_path):
        print(f"Error: Dataset directory not found at '{dataset_path}'")
        print("Please update 'dataset_path' in the script to your actual dataset location.")
    else:
        process_dataset_for_stereo(dataset_path)