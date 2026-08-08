from __future__ import annotations

import numpy as np
import torch
from scipy.signal import butter, filtfilt, hilbert, iirnotch, sosfiltfilt
from torch import Tensor


def bandpass_numpy(data: np.ndarray, sample_rate: float, low: float, high: float, order: int = 4) -> np.ndarray:
    if data.ndim < 1:
        raise ValueError("signal must have at least one dimension")
    if not 0.0 < low < high < sample_rate / 2.0:
        raise ValueError("band edges must lie inside the Nyquist interval")
    sections = butter(order, (low, high), btype="bandpass", fs=sample_rate, output="sos")
    return np.asarray(sosfiltfilt(sections, data, axis=-1), dtype=np.float32)


def notch_numpy(data: np.ndarray, sample_rate: float, frequency: float, quality: float = 30.0) -> np.ndarray:
    if frequency >= sample_rate / 2.0:
        return np.asarray(data, dtype=np.float32)
    numerator, denominator = iirnotch(frequency, quality, sample_rate)
    return np.asarray(filtfilt(numerator, denominator, data, axis=-1), dtype=np.float32)


def zscore_numpy(data: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    mean = data.mean(axis=-1, keepdims=True)
    deviation = data.std(axis=-1, keepdims=True)
    return np.asarray((data - mean) / np.maximum(deviation, epsilon), dtype=np.float32)


def analytic_envelope_numpy(data: np.ndarray) -> np.ndarray:
    return np.asarray(np.abs(hilbert(data, axis=-1)), dtype=np.float32)


def band_envelope_numpy(data: np.ndarray, sample_rate: float, low: float, high: float) -> np.ndarray:
    return analytic_envelope_numpy(bandpass_numpy(data, sample_rate, low, high))


def rfft_band_envelope(data: Tensor, sample_rate: float, low: float, high: float) -> Tensor:
    length = data.shape[-1]
    spectrum = torch.fft.rfft(data, dim=-1)
    frequencies = torch.fft.rfftfreq(length, d=1.0 / sample_rate).to(data.device)
    keep = (frequencies >= low) & (frequencies <= high)
    filtered = spectrum * keep.to(spectrum.dtype)
    analytic = torch.fft.irfft(filtered, n=length, dim=-1)
    power = analytic.square()
    return torch.sqrt(power + 1e-8)


def dual_band_envelope(data: Tensor, sample_rate: float) -> Tensor:
    mu = rfft_band_envelope(data, sample_rate, 8.0, 13.0)
    beta = rfft_band_envelope(data, sample_rate, 13.0, 30.0)
    return torch.stack((mu, beta), dim=-2)


def preprocess_numpy(data: np.ndarray, sample_rate: float) -> np.ndarray:
    filtered = bandpass_numpy(data, sample_rate, 4.0, 40.0)
    filtered = notch_numpy(filtered, sample_rate, 50.0)
    return zscore_numpy(filtered)
