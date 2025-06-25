import os
import argparse
import subprocess
import sys

def run_inference_on_test_set(test_set_root: str, checkpoint_path: str, global_output_root: str):
    """
    Runs the scnet.inference command on each mixture.wav file found
    in subdirectories of the test_set_root. Separated results are stored
    in respective subfolders within the global_output_root.

    Args:
        test_set_root (str): The root directory of your test set,
                             containing subfolders like song1/, song2/, etc.
        checkpoint_path (str): The absolute path to your model checkpoint file.
        global_output_root (str): The centralized root directory where all
                                  separated song results will be stored.
    """
    print("--- Starting SCNet Inference Runner ---")
    print(f"Test set root directory: {test_set_root}")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"All separated results will be saved under: {global_output_root}")
    print("-" * 50)

    # 1. Validate TEST_SET_ROOT exists and is a directory
    if not os.path.isdir(test_set_root):
        print(f"Error: Test set directory '{test_set_root}' not found or is not a directory.")
        print("Please ensure the 'test_set_root' argument points to a valid directory.")
        sys.exit(1)

    # 2. Validate CHECKPOINT_PATH exists and is a file
    if not os.path.isfile(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found.")
        print("Please ensure the 'checkpoint_path' argument points to a valid file.")
        sys.exit(1)

    # 3. Create the global output root directory if it doesn't exist
    try:
        os.makedirs(global_output_root, exist_ok=True)
        print(f"Ensured global output directory '{global_output_root}' exists.")
    except OSError as e:
        print(f"Error: Could not create global output directory '{global_output_root}': {e}")
        sys.exit(1)

    processed_count = 0
    skipped_count = 0
    error_count = 0

    # Iterate through each entry in the test_set_root
    for entry_name in os.listdir(test_set_root):
        song_dir_input = os.path.join(test_set_root, entry_name)

        # Check if the entry is a directory (assuming each song is in its own folder)
        if os.path.isdir(song_dir_input):
            mixture_file = os.path.join(song_dir_input, 'mixture.wav')
            # Calculate the specific output directory for this song within the global output root
            song_output_dir = os.path.join(global_output_root, entry_name)

            print(f"\nProcessing song folder: {entry_name}")
            print(f"  Mixture file expected at: {mixture_file}")
            print(f"  Separated output to: {song_output_dir}")

            # Check if mixture.wav exists in the current song directory
            if os.path.isfile(mixture_file):
                # Create the specific output directory for this song's results
                try:
                    os.makedirs(song_output_dir, exist_ok=True)
                    print(f"  Ensured song-specific output directory '{song_output_dir}' exists.")
                except OSError as e:
                    print(f"  Error: Could not create song-specific output directory '{song_output_dir}': {e}")
                    error_count += 1
                    continue # Skip to next song folder

                # Construct the command to run scnet.inference
                command = [
                    sys.executable,  # Use the Python executable that runs this script
                    '-m', 'scnet.inference',
                    '--input_dir', song_dir_input, # This is the original song folder
                    '--output_dir', song_output_dir, # This is the new centralized output folder for the song
                    '--checkpoint_path', checkpoint_path
                ]

                print(f"  Executing command: {' '.join(command)}")

                try:
                    # Run the subprocess command
                    result = subprocess.run(command, capture_output=True, text=True, check=True)

                    print(f"  Successfully processed '{entry_name}'.")
                    if result.stdout:
                        print("  STDOUT:\n" + result.stdout)
                    if result.stderr:
                        print("  STDERR:\n" + result.stderr)
                    processed_count += 1

                except subprocess.CalledProcessError as e:
                    print(f"  Error: Command failed for '{entry_name}' with exit code {e.returncode}.")
                    print(f"  STDOUT:\n{e.stdout}")
                    print(f"  STDERR:\n{e.stderr}")
                    error_count += 1
                except FileNotFoundError:
                    print(f"  Error: 'python' or 'scnet.inference' command not found. "
                          "Ensure Python and the scnet package are correctly installed and in your PATH.")
                    error_count += 1
                except Exception as e:
                    print(f"  An unexpected error occurred while processing '{entry_name}': {e}")
                    error_count += 1
            else:
                print(f"  Skipping '{entry_name}': 'mixture.wav' not found in '{song_dir_input}'.")
                skipped_count += 1
        else:
            print(f"Skipping non-directory entry: {entry_name}")
            skipped_count += 1
        print("-" * 50)

    print("\n--- Inference Process Summary ---")
    print(f"Total songs processed successfully: {processed_count}")
    print(f"Total entries skipped (not directories or missing mixture.wav): {skipped_count}")
    print(f"Total songs failed to process: {error_count}")
    print("Inference process complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run scnet.inference on a test set organized in song folders."
    )
    parser.add_argument(
        '--test_set_root',
        type=str,
        default='/home/kaim/projects/def-ichiro/kaim/data/hum_guitar',
        help='The root directory of your test set (e.g., /path/to/test_data/)'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='/home/kaim/projects/def-ichiro/kaim/SCNet/result/checkpoint.th',
        help='The absolute path to your model checkpoint file (default: /home/kaim/projects/def-ichiro/kaim/SCNet/result/checkpoint.th)'
    )
    parser.add_argument(
        '--global_output_root',
        type=str,
        default='/home/kaim/projects/def-ichiro/kaim/data/separated',
        help='The centralized root directory where all separated song results will be stored (default: /home/kaim/projects/def-ichiro/kaim/data/guitar_hum_dataset_split/separated)'
    )

    args = parser.parse_args()

    run_inference_on_test_set(args.test_set_root, args.checkpoint_path, args.global_output_root)
