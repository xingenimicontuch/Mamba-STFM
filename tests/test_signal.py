from __future__ import annotations

import numpy as np
import pytest
import torch
from mamba_stfm.signal.augment import (
    AmplitudeScale,
    BaselineWander,
    ChannelDropout,
    ChannelMix,
    ChannelShuffle,
    CircularShift,
    ClampOutliers,
    Compose,
    DCShift,
    FrequencyMask,
    FrequencyShift,
    GaussianNoise,
    PolarityFlip,
    Quantize,
    RandomApply,
    RelativeNoise,
    SmoothTimeMask,
    TemporalCropResize,
    TimeReverse,
)
from mamba_stfm.signal.filters import (
    analytic_envelope_numpy,
    band_envelope_numpy,
    bandpass_numpy,
    dual_band_envelope,
    notch_numpy,
    preprocess_numpy,
    rfft_band_envelope,
    zscore_numpy,
)
from mamba_stfm.signal.windows import cue_epochs, resample_linear, sliding_windows, window_bounds


def sine(frequency: float, sample_rate: float = 250.0, seconds: float = 4.0) -> np.ndarray:
    time = np.arange(int(sample_rate * seconds)) / sample_rate
    return np.sin(2.0 * np.pi * frequency * time).astype(np.float32)


def test_bandpass_keeps_mu_and_rejects_high_frequency() -> None:
    signal = sine(10.0) + sine(70.0)
    result = bandpass_numpy(signal, 250.0, 8.0, 13.0)
    correlation = np.corrcoef(result, sine(10.0))[0, 1]
    assert correlation > 0.99


def test_bandpass_rejects_invalid_edges() -> None:
    with pytest.raises(ValueError):
        bandpass_numpy(sine(10.0), 250.0, 13.0, 8.0)


def test_notch_skips_frequency_above_nyquist() -> None:
    signal = sine(10.0, 80.0)
    result = notch_numpy(signal, 80.0, 50.0)
    np.testing.assert_allclose(result, signal)


def test_zscore_centers_each_channel() -> None:
    generator = np.random.default_rng(7)
    signal = generator.normal(3.0, 2.0, size=(4, 1000)).astype(np.float32)
    result = zscore_numpy(signal)
    np.testing.assert_allclose(result.mean(-1), 0.0, atol=1e-6)
    np.testing.assert_allclose(result.std(-1), 1.0, atol=1e-6)


def test_analytic_envelope_of_sine_is_one() -> None:
    result = analytic_envelope_numpy(sine(10.0))
    np.testing.assert_allclose(result[20:-20], 1.0, atol=0.02)


def test_band_envelope_returns_original_shape() -> None:
    signal = np.stack((sine(10.0), sine(20.0)))
    result = band_envelope_numpy(signal, 250.0, 8.0, 13.0)
    assert result.shape == signal.shape


def test_rfft_envelope_is_differentiable() -> None:
    signal = torch.randn(2, 3, 1000, requires_grad=True)
    result = rfft_band_envelope(signal, 250.0, 8.0, 13.0)
    result.mean().backward()
    assert signal.grad is not None
    assert torch.isfinite(signal.grad).all()


def test_dual_band_has_band_axis() -> None:
    signal = torch.randn(2, 3, 1000)
    result = dual_band_envelope(signal, 250.0)
    assert result.shape == (2, 3, 2, 1000)


def test_preprocess_is_finite() -> None:
    signal = np.random.default_rng(2).normal(size=(3, 1000)).astype(np.float32)
    result = preprocess_numpy(signal, 250.0)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


def test_window_bounds_matches_protocol() -> None:
    result = list(window_bounds(2000, 1000, 62))
    assert result[0] == (0, 1000)
    assert result[1] == (62, 1062)
    assert result[-1][1] <= 2000


def test_sliding_windows_returns_views() -> None:
    signal = np.arange(20).reshape(2, 10)
    result = list(sliding_windows(signal, 4, 3))
    assert [window.start for window in result] == [0, 3, 6]
    np.testing.assert_array_equal(result[1].values, signal[:, 3:7])


def test_cue_epochs_drops_out_of_range_trials() -> None:
    signal = np.arange(20).reshape(2, 10)
    result = cue_epochs(signal, np.array((-1, 1, 8)), width=4)
    assert result.shape == (1, 2, 4)


def test_resample_linear_changes_length() -> None:
    signal = np.stack((sine(10.0, 250.0), sine(20.0, 250.0)))
    result = resample_linear(signal, 250.0, 160.0)
    assert result.shape == (2, 640)


@pytest.mark.parametrize(
    "transform",
    (
        GaussianNoise(0.1),
        RelativeNoise(0.1),
        AmplitudeScale(0.8, 1.2),
        DCShift(0.2),
        TimeReverse(),
        PolarityFlip(),
        CircularShift(10),
        TemporalCropResize(0.8),
        ChannelDropout(0.25),
        ChannelShuffle(),
        ChannelMix(0.1),
        FrequencyMask(250.0, 4.0, 40.0, 3.0),
        FrequencyShift(250.0, 2.0),
        SmoothTimeMask(0.1),
        BaselineWander(250.0, 1.0, 0.1),
        Quantize(32),
        ClampOutliers(3.0),
    ),
)
def test_augmentation_preserves_shape_and_finiteness(transform: object) -> None:
    signal = torch.randn(2, 4, 1000)
    generator = torch.Generator().manual_seed(3)
    result = transform(signal, generator)
    assert result.shape == signal.shape
    assert torch.isfinite(result).all()


def test_random_apply_zero_is_identity() -> None:
    signal = torch.randn(2, 4, 1000)
    result = RandomApply(PolarityFlip(), 0.0)(signal)
    assert torch.equal(result, signal)


def test_random_apply_one_runs_transform() -> None:
    signal = torch.randn(2, 4, 1000)
    result = RandomApply(PolarityFlip(), 1.0)(signal)
    assert torch.equal(result, -signal)


def test_compose_runs_in_order() -> None:
    signal = torch.ones(2, 4, 100)
    result = Compose((PolarityFlip(), AmplitudeScale(2.0, 2.0)))(signal)
    assert torch.equal(result, -2.0 * signal)


def test_channel_dropout_drops_exact_count() -> None:
    signal = torch.ones(2, 4, 100)
    result = ChannelDropout(0.5)(signal, torch.Generator().manual_seed(1))
    dropped = (result.abs().sum(dim=(0, 2)) == 0).sum()
    assert dropped == 2


def test_quantize_limits_unique_values() -> None:
    signal = torch.linspace(-1.0, 1.0, 100).reshape(1, 1, 100)
    result = Quantize(8)(signal)
    assert torch.unique(result).numel() <= 8
