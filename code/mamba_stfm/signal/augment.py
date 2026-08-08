from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


class Transform(Protocol):
    def __call__(self, signal: Tensor, generator: torch.Generator | None = None) -> Tensor: ...


def probability(generator: torch.Generator | None, device: torch.device) -> Tensor:
    return torch.rand((), generator=generator, device=device)


@dataclass(frozen=True)
class Compose:
    transforms: tuple[Transform, ...]

    def __call__(self, signal: Tensor, generator: torch.Generator | None = None) -> Tensor:
        result = signal
        for transform in self.transforms:
            result = transform(result, generator)
        return result


@dataclass(frozen=True)
class RandomApply:
    transform: Transform
    chance: float

    def __call__(self, signal: Tensor, generator: torch.Generator | None = None) -> Tensor:
        if not 0.0 <= self.chance <= 1.0:
            raise ValueError("chance must be in [0, 1]")
        return self.transform(signal, generator) if probability(generator, signal.device) < self.chance else signal


@dataclass(frozen=True)
class GaussianNoise:
    standard_deviation: float

    def __call__(self, signal: Tensor, generator: torch.Generator | None = None) -> Tensor:
        noise = torch.randn(signal.shape, dtype=signal.dtype, device=signal.device, generator=generator)
        return signal + self.standard_deviation * noise


@dataclass(frozen=True)
class RelativeNoise:
    ratio: float

    def __call__(self, signal: Tensor, generator: torch.Generator | None = None) -> Tensor:
        scale = signal.std(dim=-1, keepdim=True).clamp_min(1e-6)
        noise = torch.randn(signal.shape, dtype=signal.dtype, device=signal.device, generator=generator)
        return signal + self.ratio * scale * noise


@dataclass(frozen=True)
class AmplitudeScale:
    minimum: float
    maximum: float

    def __call__(self, signal: Tensor, generator: torch.Generator | None = None) -> Tensor:
        if self.minimum <= 0.0 or self.maximum < self.minimum:
            raise ValueError("invalid amplitude interval")
        shape = (*signal.shape[:-1], 1)
        random = torch.rand(shape, dtype=signal.dtype, device=signal.device, generator=generator)
        return signal * (self.minimum + random * (self.maximum - self.minimum))


@dataclass(frozen=True)
class DCShift:
    magnitude: float

    def __call__(self, signal: Tensor, generator: torch.Generator | None = None) -> Tensor:
        shape = (*signal.shape[:-1], 1)
        shift = torch.rand(shape, dtype=signal.dtype, device=signal.device, generator=generator)
        return signal + self.magnitude * (2.0 * shift - 1.0)


@dataclass(frozen=True)
class TimeReverse:
    def __call__(self, signal: Tensor, generator: torch.Generator | None = None) -> Tensor:
        return signal.flip(-1)


@dataclass(frozen=True)
class PolarityFlip:
    def __call__(self, signal: Tensor, generator: torch.Generator | None = None) -> Tensor:
        return -signal


@dataclass(frozen=True)
class CircularShift:
    maximum_samples: int

    def __call__(self, signal: Tensor, generator: torch.Generator | None = None) -> Tensor:
        if self.maximum_samples < 0:
            raise ValueError("maximum shift must be nonnegative")
        shift = int(torch.randint(-self.maximum_samples, self.maximum_samples + 1, (), generator=generator, device=signal.device).item())
        return torch.roll(signal, shift, dims=-1)


@dataclass(frozen=True)
class TemporalCropResize:
    minimum_fraction: float

    def __call__(self, signal: Tensor, generator: torch.Generator | None = None) -> Tensor:
        if not 0.0 < self.minimum_fraction <= 1.0:
            raise ValueError("minimum fraction must be in (0, 1]")
        length = signal.shape[-1]
        fraction = self.minimum_fraction + (1.0 - self.minimum_fraction) * float(probability(generator, signal.device).item())
        crop = max(2, int(round(length * fraction)))
        maximum_start = length - crop
        start = int(torch.randint(0, maximum_start + 1, (), generator=generator, device=signal.device).item())
        selected = signal[..., start : start + crop]
        flat = selected.reshape(-1, 1, crop)
        resized = torch.nn.functional.interpolate(flat, size=length, mode="linear", align_corners=False)
        return resized.reshape_as(signal)


@dataclass(frozen=True)
class ChannelDropout:
    ratio: float

    def __call__(self, signal: Tensor, generator: torch.Generator | None = None) -> Tensor:
        if not 0.0 <= self.ratio < 1.0:
            raise ValueError("ratio must be in [0, 1)")
        channels = signal.shape[-2]
        count = int(round(channels * self.ratio))
        if count == 0:
            return signal
        order = torch.randperm(channels, generator=generator, device=signal.device)
        mask = torch.ones(channels, dtype=signal.dtype, device=signal.device)
        mask[order[:count]] = 0.0
        shape = (1,) * (signal.ndim - 2) + (channels, 1)
        return signal * mask.reshape(shape)


@dataclass(frozen=True)
class ChannelShuffle:
    def __call__(self, signal: Tensor, generator: torch.Generator | None = None) -> Tensor:
        order = torch.randperm(signal.shape[-2], generator=generator, device=signal.device)
        return signal.index_select(-2, order)


@dataclass(frozen=True)
class ChannelMix:
    strength: float

    def __call__(self, signal: Tensor, generator: torch.Generator | None = None) -> Tensor:
        channels = signal.shape[-2]
        matrix = torch.randn((channels, channels), dtype=signal.dtype, device=signal.device, generator=generator)
        matrix = matrix / matrix.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        identity = torch.eye(channels, dtype=signal.dtype, device=signal.device)
        transform = (1.0 - self.strength) * identity + self.strength * matrix
        return torch.einsum("ij,...jt->...it", transform, signal)


@dataclass(frozen=True)
class FrequencyMask:
    sample_rate: float
    minimum_hz: float
    maximum_hz: float
    width_hz: float

    def __call__(self, signal: Tensor, generator: torch.Generator | None = None) -> Tensor:
        length = signal.shape[-1]
        spectrum = torch.fft.rfft(signal, dim=-1)
        frequencies = torch.fft.rfftfreq(length, d=1.0 / self.sample_rate).to(signal.device)
        span = self.maximum_hz - self.minimum_hz - self.width_hz
        start = self.minimum_hz + max(0.0, span) * float(probability(generator, signal.device).item())
        keep = ~((frequencies >= start) & (frequencies < start + self.width_hz))
        return torch.fft.irfft(spectrum * keep, n=length, dim=-1)


@dataclass(frozen=True)
class FrequencyShift:
    sample_rate: float
    maximum_hz: float

    def __call__(self, signal: Tensor, generator: torch.Generator | None = None) -> Tensor:
        length = signal.shape[-1]
        frequency_resolution = self.sample_rate / length
        maximum_bins = int(self.maximum_hz / frequency_resolution)
        bins = int(torch.randint(-maximum_bins, maximum_bins + 1, (), generator=generator, device=signal.device).item())
        spectrum = torch.fft.rfft(signal, dim=-1)
        shifted = torch.roll(spectrum, bins, dims=-1)
        if bins > 0:
            shifted[..., :bins] = 0
        elif bins < 0:
            shifted[..., bins:] = 0
        return torch.fft.irfft(shifted, n=length, dim=-1)


@dataclass(frozen=True)
class SmoothTimeMask:
    ratio: float

    def __call__(self, signal: Tensor, generator: torch.Generator | None = None) -> Tensor:
        length = signal.shape[-1]
        width = max(1, int(round(length * self.ratio)))
        start = int(torch.randint(0, max(1, length - width + 1), (), generator=generator, device=signal.device).item())
        ramp = min(width // 4, 16)
        mask = torch.ones(length, dtype=signal.dtype, device=signal.device)
        mask[start : start + width] = 0.0
        if ramp:
            mask[start : start + ramp] = torch.linspace(1.0, 0.0, ramp, device=signal.device)
            mask[start + width - ramp : start + width] = torch.linspace(0.0, 1.0, ramp, device=signal.device)
        return signal * mask


@dataclass(frozen=True)
class BaselineWander:
    sample_rate: float
    maximum_hz: float
    magnitude: float

    def __call__(self, signal: Tensor, generator: torch.Generator | None = None) -> Tensor:
        length = signal.shape[-1]
        frequency = self.maximum_hz * float(probability(generator, signal.device).item())
        phase = 2.0 * torch.pi * float(probability(generator, signal.device).item())
        time = torch.arange(length, dtype=signal.dtype, device=signal.device) / self.sample_rate
        wave = torch.sin(2.0 * torch.pi * frequency * time + phase)
        return signal + self.magnitude * wave


@dataclass(frozen=True)
class Quantize:
    levels: int

    def __call__(self, signal: Tensor, generator: torch.Generator | None = None) -> Tensor:
        if self.levels < 2:
            raise ValueError("levels must be at least two")
        minimum = signal.amin(dim=-1, keepdim=True)
        maximum = signal.amax(dim=-1, keepdim=True)
        scale = (maximum - minimum).clamp_min(1e-6) / (self.levels - 1)
        return torch.round((signal - minimum) / scale) * scale + minimum


@dataclass(frozen=True)
class ClampOutliers:
    standard_deviations: float

    def __call__(self, signal: Tensor, generator: torch.Generator | None = None) -> Tensor:
        mean = signal.mean(dim=-1, keepdim=True)
        deviation = signal.std(dim=-1, keepdim=True).clamp_min(1e-6)
        lower = mean - self.standard_deviations * deviation
        upper = mean + self.standard_deviations * deviation
        return torch.maximum(torch.minimum(signal, upper), lower)
