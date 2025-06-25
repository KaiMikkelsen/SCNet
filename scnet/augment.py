#From HT demucs https://github.com/facebookresearch/demucs/tree/release_v4?tab=readme-ov-file

import random
import torch as th
from torch import nn
import numpy as np
import os
import soundfile as sf
# Import all necessary effects from pedalboard
from pedalboard import (
    Pedalboard,
    Distortion,
    Reverb,
    Chorus,
    Phaser,
    Delay,
    Compressor,
    Gain,
    Clipping,
    Convolution,
    # Add more effects here as you see fit: e.g., Phaser, Flanger, Tremolo, EQ, etc.
)

class Shift(nn.Module):
    """
    Randomly shift audio in time by up to `shift` samples.
    """
    def __init__(self, shift=8192, same=False):
        super().__init__()
        self.shift = shift
        self.same = same

    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        length = time - self.shift
        if self.shift > 0:
            if not self.training:
                wav = wav[..., :length]
            else:
                srcs = 1 if self.same else sources
                offsets = th.randint(self.shift, [batch, srcs, 1, 1], device=wav.device)
                offsets = offsets.expand(-1, sources, channels, -1)
                indexes = th.arange(length, device=wav.device)
                wav = wav.gather(3, indexes + offsets)
        return wav


class FlipChannels(nn.Module):
    """
    Flip left-right channels.
    """
    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        if self.training and wav.size(2) == 2:
            left = th.randint(2, (batch, sources, 1, 1), device=wav.device)
            left = left.expand(-1, -1, -1, time)
            right = 1 - left
            wav = th.cat([wav.gather(2, left), wav.gather(2, right)], dim=2)
        return wav


class FlipSign(nn.Module):
    """
    Random sign flip.
    """
    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        if self.training:
            signs = th.randint(2, (batch, sources, 1, 1), device=wav.device, dtype=th.float32)
            wav = wav * (2 * signs - 1)
        return wav


class Remix(nn.Module):
    """
    Shuffle sources to make new mixes.
    """
    def __init__(self, proba=1, group_size=4):
        """
        Shuffle sources within one batch.
        Each batch is divided into groups of size `group_size` and shuffling is done within
        each group separatly. This allow to keep the same probability distribution no matter
        the number of GPUs. Without this grouping, using more GPUs would lead to a higher
        probability of keeping two sources from the same track together which can impact
        performance.
        """
        super().__init__()
        self.proba = proba
        self.group_size = group_size

    def forward(self, wav):
        batch, streams, channels, time = wav.size()
        device = wav.device

        if self.training and random.random() < self.proba:
            group_size = self.group_size or batch
            if batch % group_size != 0:
                raise ValueError(f"Batch size {batch} must be divisible by group size {group_size}")
            groups = batch // group_size
            wav = wav.view(groups, group_size, streams, channels, time)
            permutations = th.argsort(th.rand(groups, group_size, streams, 1, 1, device=device),
                                      dim=1)
            wav = wav.gather(1, permutations.expand(-1, -1, -1, channels, time))
            wav = wav.view(batch, streams, channels, time)
        return wav


class Scale(nn.Module):
    def __init__(self, proba=1., min=0.25, max=1.25):
        super().__init__()
        self.proba = proba
        self.min = min
        self.max = max

    def forward(self, wav):
        batch, streams, channels, time = wav.size()
        device = wav.device
        if self.training and random.random() < self.proba:
            scales = th.empty(batch, streams, 1, 1, device=device).uniform_(self.min, self.max)
            wav *= scales
        return wav

class PedalboardEffectModule(nn.Module):
    """
    A randomized Pedalboard effect module for diversifying audio data.
    Applies a random subset of specified effects with randomized parameters
    INDIVIDUALLY to each stereo source (stream).
    Input `wav` is expected as (batch, streams, channels=2, time).
    The module processes each stream separately and returns the same shape.
    """
    def __init__(self, sample_rate, proba=1.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.proba = proba # Overall probability of applying *any* augmentation on a given call

        
        # You can adjust these if you want to allow more or fewer effects per chain
        self.min_effects_per_chain = 0 
        self.max_effects_per_chain = 3 
        
        # The pedalboard instance will be created dynamically within the forward pass
        self.pedalboard = None 


    def create_random_pedalboard(self):

       # Define a small pool of effects for this basic example
       possible_effects = [
            Chorus(),
            Phaser(),
            #Delay(delay_seconds=random.uniform(1, 3), mix=random.uniform(0.3, 0.7), feedback=random.uniform(0.0, 0.5)),
            # Distortion(drive_db=random.uniform(10, 50)),
            Compressor(threshold_db=random.uniform(-40, -20), ratio=random.uniform(2, 10))
       ]

       # Randomly select a number of effects (e.g., 1 to 3)
       num_effects = random.randint(0, min(4, len(possible_effects)))
       selected_effects = random.sample(possible_effects, num_effects)

       # Randomize their order
       random.shuffle(selected_effects)

       #print(f"Creating Pedalboard with {num_effects} effects: {[type(effect).__name__ for effect in selected_effects]}")

       return Pedalboard(selected_effects)
       #return Pedalboard()
  

    def forward(self, wav):
        
        batch, streams, channels, time = wav.size()
        device = wav.device

        if channels != 2:
            raise ValueError(f"PedalboardEffectModule expects stereo input (channels=2), but got {channels}.")

        if not self.training or random.random() >= self.proba:
            return wav
        
        processed_streams = th.empty_like(wav)

        for b in range(batch):
            # Each batch item gets its own *single* randomized pedalboard,
            # which is then applied to all streams within that batch item.
            #current_pedalboard = self._get_random_pedalboard()

            board = self.create_random_pedalboard()

            
            for s in range(streams):
                current_stereo_stream_np = wav[b, s].cpu().numpy().astype(np.float32)

                effected_audio_np = board(current_stereo_stream_np, self.sample_rate)

                if effected_audio_np.ndim != 2 or effected_audio_np.shape[0] != 2:
                    raise RuntimeError(
                        f"Pedalboard returned unexpected shape {effected_audio_np.shape} for stream {s}, batch {b}. Expected (2, time)."
                    )
                
                processed_streams[b, s] = th.from_numpy(effected_audio_np).to(device)

        #print(f"Processed {batch} batches with {streams} streams each using Pedalboard effects: {[type(effect).__name__ for effect in board.effects]}")
        return processed_streams