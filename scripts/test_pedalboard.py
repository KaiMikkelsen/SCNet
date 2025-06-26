from pedalboard import Pedalboard, Reverb, load_plugin, Chorus, Phaser, Delay, Distortion, Compressor, Convolution
from pedalboard.io import AudioFile
from mido import Message # not part of Pedalboard, but convenient!
from pedalboard.io import AudioFile


import random

import numpy as np

import os
import librosa
import soundfile as sf # soundfile is often used with librosa for writing/reading various audio formats



# Create a Pedalboard instance
board = Pedalboard()
# sometimes we dont want any signal just dry guitar

random_index = random.randint(0, 3)
amp_effect = load_plugin("/Library/Audio/Plug-Ins/Components/BC Free Amp AU.component")


amp_effect.parameters['model'].value = 'Modern Drive'

amp_effect.model = 'Modern Drive'  # Set the amp model to 'Modern Drive'
amp_effect.drive = 11 # Set the gain to 5

board.append(amp_effect)

input_path = "/Users/kaimikkelsen/Downloads/cpls.wav"
output_path = "/Users/kaimikkelsen/Downloads/cpls_processed.wav"

# Load audio
samplerate = 44100.0
with AudioFile(input_path).resampled_to(samplerate) as f:
  audio = f.read(f.frames)

# Process through pedalboard
processed_audio = board(audio, samplerate)

# Normalize if needed
max_val = np.max(np.abs(processed_audio))
if max_val > 1.0:
    processed_audio = processed_audio / max_val


# Write the audio back as a wav file:
effected = board(audio, samplerate)

# Write the audio back as a wav file:
with AudioFile(output_path, 'w', samplerate, effected.shape[0]) as f:
  f.write(effected)
# Save output
# sf.write(output_path, processed_audio, sr)
# print(f"Processed file saved to: {output_path}")