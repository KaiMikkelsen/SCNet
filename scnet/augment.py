#From HT demucs https://github.com/facebookresearch/demucs/tree/release_v4?tab=readme-ov-file

import random
import torch as th
from torch import nn
from pedalboard import Pedalboard, Distortion, Reverb

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

# --- SIMPLIFIED Pedalboard Augmentation Class for STEREO Testing ---
class SimplifiedPedalboardEffectModule(nn.Module):
    """
    A simplified Pedalboard effect module for testing, specifically designed for STEREO audio.
    Applies a fixed set of basic effects (Distortion, Reverb) to the STEREO audio mixture.
    Input `wav` is expected as (batch, streams, channels=2, time).
    The module sums `streams` to get the stereo mix, applies effects, and then scales
    the original `streams` so their sum matches the effected stereo mix.
    """
    def __init__(self, sample_rate, proba=1.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.proba = proba # Probability of applying this augmentation

        # Define a super simple, fixed set of effects for testing
        self.effects = [
            Distortion(drive_db=15), # Gentle distortion
            Reverb(room_size=0.2, wet_level=0.1), # Small, subtle reverb
        ]
        self.pedalboard = Pedalboard(self.effects)

    def forward(self, wav):
        batch, streams, channels, time = wav.size()
        device = wav.device

        # Assert that input is stereo
        if channels != 2:
            raise ValueError(f"SimplifiedPedalboardEffectModule expects stereo input (channels=2), but got {channels}.")

        # Skip augmentation if not training or random check fails
        if not self.training or random.random() >= self.proba:
            return wav

        # 1. Create the stereo mixture from the input 'wav' tensor
        # Sum across the 'streams' dimension: (batch, channels, time) -> (batch, 2, time)
        mix_to_effect = wav.sum(dim=1)

        processed_mix_batch = []
        for i in range(batch):
            # Extract one stereo mix from the batch, convert to numpy (float32, on CPU)
            # Input to Pedalboard should be (channels, samples)
            current_stereo_mix_np = mix_to_effect[i].cpu().numpy().astype(np.float32) # Shape (2, time)

            # Apply Pedalboard effects
            # Pedalboard will automatically handle stereo input if shaped (2, samples)
            effected_audio_np = self.pedalboard(current_stereo_mix_np, self.sample_rate)

            # Ensure the output is still (channels, time) for stacking back to PyTorch
            if effected_audio_np.ndim != 2 or effected_audio_np.shape[0] != 2:
                # This check ensures Pedalboard didn't unexpectedly change channel count
                raise RuntimeError(
                    f"Pedalboard returned unexpected shape {effected_audio_np.shape}. Expected (2, time)."
                )

            processed_mix_batch.append(th.from_numpy(effected_audio_np).to(device))

        # Stack the processed stereo mixes back into a tensor
        # Resulting shape: (batch, 2, time)
        effected_mix_tensor = th.stack(processed_mix_batch, dim=0)

        # 2. Re-distribute the effect back onto the original sources (streams)
        # Calculate original sum for scaling - this will also be (batch, 2, time)
        original_sum = wav.sum(dim=1)

        # Calculate scaling factor to adjust individual sources
        # Add epsilon for numerical stability when dividing by potentially zero values
        # Unsqueeze(1) broadcasts the (batch, 2, time) scale factors over the 'streams' dim
        scale_factors = (effected_mix_tensor / (original_sum + 1e-8)).unsqueeze(1)

        # Apply scaling to the original individual sources
        # wav_out will have the same shape as wav: (batch, streams, channels=2, time)
        wav_out = wav * scale_factors

        return wav_out