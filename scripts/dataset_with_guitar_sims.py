from pedalboard import Pedalboard, Reverb, load_plugin, Chorus, Phaser, Delay, Distortion, Compressor, Convolution, Limiter
from pedalboard.io import AudioFile
from mido import Message # not part of Pedalboard, but convenient!


import random

import numpy as np

import os
import librosa
import soundfile as sf # soundfile is often used with librosa for writing/reading various audio formats

def get_random_ir_path(ir_folder="/Users/kaimikkelsen/SCNet_guitar/data/IR_folder/wavIR"):
    wav_files = [f for f in os.listdir(ir_folder) if f.lower().endswith('.wav')]
    if not wav_files:
        raise FileNotFoundError("No .wav files found in IR folder!")
    return os.path.join(ir_folder, random.choice(wav_files))



def create_random_pedalboard():
    """
    Creates a random Pedalboard with a Reverb effect and a VST3 plugin.
    
    Returns:
        Pedalboard: A Pedalboard instance with a Reverb and a VST3 plugin.
    """
    # Create a Pedalboard instance
    board = Pedalboard()

    # sometimes we dont want any signal just dry guitar
    if random.randint(1, 5) == 1:
        return board

    random_index = random.randint(0, 3)

    amp_effect = load_plugin("/Library/Audio/Plug-Ins/Components/BC Free Amp AU.component")
    if random_index == 0:
        
        amp_effect.model = 'Classic Clean'
        print("Using Classic Clean model")
    elif random_index == 1:
        amp_effect.model = 'Classic Drive'
        print("Using Classic Drive model")
    elif random_index == 2:
        amp_effect.model = 'Modern Drive'
        print("Using Modern Drive model")
    else:
        pass

    # amp_effect.parameters['model'].value = 'Modern Drive'
    # print("Using Modern Drive model")

    # print("Amp effect parameters:", amp_effect.parameters.keys())

    # print("Amp effect model:", amp_effect.parameters['model'].value)

    # param_names = ['gain', 'drive', 'tone', 'bass', 'mid', 'treble', 'volume']

    # for name in param_names:
    #     param = amp_effect.parameters[name]
    #     if hasattr(param, 'valid_values') and param.valid_values is not None:
    #         print(f"{name}: valid values = {param.valid_values}")
    #     elif hasattr(param, 'min_value') and hasattr(param, 'max_value'):
    #         print(f"{name}: min = {param.min_value}, max = {param.max_value}")
    #     else:
    #         print(f"{name}: No range info available")


    #gain_value = random.randint(-20, 20)  # Random gain value between 0.1 and 1.0
    amp_effect.gain = 0
    amp_effect.drive = random.randint(3, 11)  # Random drive value between -20 and 20
    amp_effect.tone = random.randint(70, 100)  # Random tone value between -20 and 20
    amp_effect.bass= random.randint(-5, 5)  # Random presence value between -20 and 20
    amp_effect.mid = random.randint(-5, 5)  # Random mid value between -20 and 20
    amp_effect.treble = random.randint(-5, 5)
    #amp_effect.parameters['tone'].value = random.randint(-20, 20)
    amp_effect.parameters['volume'].value = 0
    
    # if(gain_value < 0):
    #     amp_effect.parameters['volume'].value = gain_value
    # else:   
    #     amp_effect.parameters['volume'].value = -1 * gain_value  # Random presence value between -20 and 20


    chorus = Chorus(rate_hz=random.uniform(0.5, 2.0), depth=random.uniform(0.1, 0.5), mix=random.uniform(0.1, 0.4))
    #phaser = Phaser(rate_hz=random.uniform(0.5, 4.0), depth=random.uniform(0.1, 0.5), mix=random.uniform(0.3, 0.7))
    delay = Delay(delay_seconds=random.uniform(1, 3), mix=random.uniform(0.3, 0.7), feedback=random.uniform(0.0, 0.5))
    reverb = Reverb(room_size=random.uniform(0.1, 0.9), damping=random.uniform(0.1, 0.9), wet_level=random.uniform(0.1, 0.3), dry_level=0.9)
    speaker_ir = Convolution(get_random_ir_path(), 1.0)

    # /Library/Audio/Impulse Responses/Apple/05 Warped/06 Speakers


    # import sys
    # sys.exit()
    effects = [amp_effect, speaker_ir, chorus, delay, reverb]
    

    #Pick a random number of effects to apply (at least 1, at most all)
    num_effects = random.randint(1, len(effects))
    selected_effects = effects[:num_effects]

    for effect in selected_effects:
        print("adding effect:", type(effect).__name__)
        board.append(effect)

    board.append(Limiter(threshold_db=-1.0))

    #board.append(amp_effect)

    return board



#     print("Amp effect parameters:", amp_effect.parameters.keys())

#     model_param = amp_effect.parameters['gain']

# # # Print all valid values for 'model'
#     print("Valid values for 'model':", model_param.valid_values)



#     board.append(Reverb(room_size=0.5, damping=0.5, wet_level=0.5, dry_level=0.5))

#     # Load a VST3 plugin (replace 'path/to/plugin.vst3' with your actual plugin path)

#     return board

def process_audio_files_in_folders(base_directory):
    """
    Iterates through subfolders in a given base directory, identifies
    'hum', 'clean', and 'mixture.wav' audio files, and loads them using librosa.

    Args:
        base_directory (str): The path to the main directory containing subfolders.
    """
    print(f"Starting to scan directory: {base_directory}\n")

    # Check if the base directory exists
    if not os.path.isdir(base_directory):
        print(f"Error: Base directory '{base_directory}' not found.")
        return


    print("Scanning for subfolders containing 'hum', 'clean', and 'mixture.wav' files...\n")
    # os.walk generates the file names in a directory tree by walking the tree
    # top-down or bottom-up. For each directory in the tree rooted at directory top
    # (including top itself), it yields a 3-tuple (dirpath, dirnames, filenames).
    for dirpath, dirnames, filenames in os.walk(base_directory):
        # We are interested in the immediate subfolders that contain the audio files.
        # So, we check if any of the target files are in the current 'filenames' list.
        # This prevents us from processing the base_directory itself if it doesn't
        # directly contain the audio files, and focuses on the subfolders.
        
        # Define the expected audio file names (case-sensitive check)
        expected_files = ['hum.wav', 'clean.wav', 'mixture.wav']
        
        # Check if all expected files exist in the current directory
        # The 'all()' function returns True if all items in an iterable are true, otherwise it returns False.
        # Here, it checks if all expected_files are present in the filenames list for the current directory.
        if all(f in filenames for f in expected_files):
            print(f"Found a subfolder with all expected files: {dirpath}")

            # Construct full paths to the audio files
            hum_file_path = os.path.join(dirpath, 'hum.wav')
            clean_file_path = os.path.join(dirpath, 'clean.wav')
            mixture_file_path = os.path.join(dirpath, 'mixture.wav')

            print(f"  Hum file path: {hum_file_path}")
            print(f"  Clean file path: {clean_file_path}")
            print(f"  Mixture file path: {mixture_file_path}\n")




            # --- Audio File Loading Logic ---
            #try:
                # Load the hum audio file
                # librosa.load returns the audio time series (numpy array) and the sampling rate
            hum_audio, sr_hum = librosa.load(hum_file_path, sr=None) # sr=None to preserve original sampling rate
            print(f"    Loaded 'hum': Shape={hum_audio.shape}, Sample Rate={sr_hum} Hz")

                    # Load the clean audio file
            clean_audio, sr_clean = librosa.load(clean_file_path, sr=None)
            print(f"    Loaded 'clean': Shape={clean_audio.shape}, Sample Rate={sr_clean} Hz")


            board = create_random_pedalboard()

            # Check if the board has any effects
            # if len(board.effects) == 0:
            #     print("    No effects applied. Skipping processing for this folder.")
            #     continue

            # Apply the pedalboard effects to the hum audio
            # Apply the pedalboard effects to the hum and clean audio
            hum_audio = board(hum_audio, sr_hum)
            clean_audio = board(clean_audio, sr_clean)

            # Scale hum_audio
            hum_max = np.max(np.abs(hum_audio))
            if hum_max > 0.5:
                hum_audio = hum_audio * (0.5 / hum_max)

            # Scale clean_audio
            clean_max = np.max(np.abs(clean_audio))
            if clean_max > 0.5:
                clean_audio = clean_audio * (0.5 / clean_max)

            print("Scaled hum_audio min:", np.min(hum_audio))
            print("Scaled hum_audio max:", np.max(hum_audio))
            print("Scaled clean_audio min:", np.min(clean_audio))
            print("Scaled clean_audio max:", np.max(clean_audio))


            # # Normalize to prevent clipping
            # def normalize_audio(audio):
            #     max_val = np.max(np.abs(audio))
            #     if max_val > 1.0:
            #         return audio / max_val
            #     return audio

            # hum_audio = normalize_audio(hum_audio)
            # clean_audio = normalize_audio(clean_audio)

            # Save the new hum and clean files
            sf.write(hum_file_path, hum_audio, sr_hum)
            sf.write(clean_file_path, clean_audio, sr_clean)
            #print(f"    Saved simulated hum: {os.path.join(dirpath, 'hum_simulated.wav')}")
            #print(f"    Saved simulated clean: {os.path.join(dirpath, 'clean_simulated.wav')}")

            # Combine and normalize for mixture
            mixture_audio = hum_audio + clean_audio
            print("Scaled mixture_audio min:", np.min(mixture_audio))
            print("Scaled mixture_audio max:", np.max(mixture_audio))

            #mixture_audio = normalize_audio(mixture_audio)
            sf.write(mixture_file_path, mixture_audio, sr_hum)
            print(f"    Saved simulated mixture: {mixture_file_path}")  

            




            # import sys
            # sys.exit()

            #         # Load the mixture audio file
            # mixture_audio, sr_mixture = librosa.load(mixture_file_path, sr=None)
            # print(f"    Loaded 'mixture.wav': Shape={mixture_audio.shape}, Sample Rate={sr_mixture} Hz")

                # --- Placeholder for your processing logic ---
                # At this point, you have the audio data (hum_audio, clean_audio, mixture_audio)
                # and their respective sampling rates (sr_hum, sr_clean, sr_mixture) in memory.
                # You can now apply your processing here.
                # For example:
                # result_audio = mixture_audio - hum_audio # Simple noise reduction
                # sf.write('output_processed.wav', result_audio, sr_mixture) # Save processed audio

            print(f"    Successfully loaded all files for processing in {dirpath}.\n")

                # Call a dedicated function for actual processing, passing the loaded data
                # process_single_set_of_audio(hum_audio, sr_hum, clean_audio, sr_clean, mixture_audio, sr_mixture)

            # except FileNotFoundError:
            #     print(f"    Error: One or more files not found in {dirpath}. Skipping this folder.")
            # except Exception as e:
            #     print(f"    An error occurred while loading audio files in {dirpath}: {e}. Skipping this folder.")
            # # -----------------------------------------------


# # Load a VST3 or Audio Unit plugin from a known path on disk:
# fender_amp_effect = load_plugin("/Library/Audio/Plug-Ins/Components/BC Free Amp AU.component")
# fender_amp_effect.parameters['model'].value = 'Classic Clean'

# marshall_amp_effect = load_plugin("/Library/Audio/Plug-Ins/Components/BC Free Amp AU.component")
# marshall_amp_effect.parameters['model'].value = 'Classic Drive'

# metal_amp_effect = load_plugin("/Library/Audio/Plug-Ins/Components/BC Free Amp AU.component")
# # metal_amp_effect.parameters['model'].value = 'Modern Drive'


# print(fender_amp_effect.parameters.keys())

dataset_base_path = "/Users/kaimikkelsen/SCNet_guitar/data/guitar_hum_dataset_with_amps"

# Call the function to start processing
process_audio_files_in_folders(dataset_base_path)

# You can define a placeholder function for actual audio processing if needed
# def process_single_set_of_audio(hum_data, hum_sr, clean_data, clean_sr, mixture_data, mixture_sr):
#     """
#     This function would contain your actual audio processing logic.
#     It receives the loaded audio data (numpy arrays) and their sampling rates.
#     """
#     print(f"  [Simulating processing with loaded data. Mixture length: {len(mixture_data)} samples]")
#     # Your audio processing code goes here.
#     # For example, applying a filter, performing source separation, etc.


# model_param = fender_amp_effect.parameters['gain']

# # Print all valid values for 'model'
# print("Valid values for 'model':", model_param.valid_values)