import torchaudio as ta
import os

file_path = "/home/kaim/scratch/MUSDB18HQ/valid/Leaf - Summerghost/mixture.wav"

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    try:
        waveform, sample_rate = ta.load(file_path)
        print(f"Successfully loaded: {file_path}")
        print(f"Waveform shape: {waveform.shape}")
        print(f"Sample rate: {sample_rate}")
    except RuntimeError as e:
        print(f"Error loading {file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
