from __future__ import annotations

import torch
from mamba_stfm.models.network import MambaSTFM
from mamba_stfm.objectives.efm import EFMPretrainer, EnvelopeDecoder, factorized_mask, masked_mean_squared_error
from mamba_stfm.settings import ModelSettings, SignalSettings


def model() -> MambaSTFM:
    return MambaSTFM(
        ModelSettings(
            channels=4,
            classes=2,
            dimension=8,
            state_dimension=8,
            expansion=2,
            convolution_width=3,
            blocks=1,
            clusters=4,
            dropout=0.0,
            bidirectional=True,
            descriptor_min_samples=8,
        ),
        SignalSettings(
            sample_rate=50,
            window_seconds=2.0,
            stride_seconds=0.25,
            patch_samples=10,
        ),
    )


def test_factorized_mask_shapes() -> None:
    mask = factorized_mask(2, 4, 2, 10, 0.5, 0.5, 0.5, torch.device("cpu"))
    assert mask.cluster.shape == (2, 4, 1, 1)
    assert mask.band.shape == (2, 1, 2, 1)
    assert mask.time.shape == (2, 1, 1, 10)
    assert mask.combined.shape == (2, 4, 2, 10)


def test_zero_ratio_yields_empty_mask() -> None:
    mask = factorized_mask(2, 4, 2, 10, 0.0, 0.0, 0.0, torch.device("cpu"))
    assert not mask.combined.any()


def test_one_cluster_ratio_masks_everything() -> None:
    mask = factorized_mask(2, 4, 2, 10, 1.0, 0.0, 0.0, torch.device("cpu"))
    assert mask.combined.all()


def test_masked_error_uses_only_selected_entries() -> None:
    prediction = torch.tensor((1.0, 2.0, 3.0))
    target = torch.zeros(3)
    mask = torch.tensor((True, False, True))
    result = masked_mean_squared_error(prediction, target, mask)
    assert result.item() == 5.0


def test_envelope_decoder_shape() -> None:
    decoder = EnvelopeDecoder(8, 4, 10, 4)
    tokens = torch.randn(2, 40, 8)
    result = decoder(tokens)
    assert result.shape == (2, 4, 2, 100)


def test_pretrainer_target_shape() -> None:
    pretrainer = EFMPretrainer(model(), 4)
    eeg = torch.randn(2, 4, 100)
    target = pretrainer.target(eeg)
    assert target.shape == (2, 4, 2, 100)


def test_pretrainer_forward_is_finite() -> None:
    pretrainer = EFMPretrainer(model(), 4)
    eeg = torch.randn(2, 4, 100)
    context = torch.randn(2, 4, 100)
    mask = factorized_mask(2, 4, 2, 10, 0.5, 0.5, 0.5, torch.device("cpu"))
    loss, prediction, target = pretrainer(eeg, context, mask)
    assert torch.isfinite(loss)
    assert prediction.shape == target.shape


def test_pretrainer_backpropagates() -> None:
    pretrainer = EFMPretrainer(model(), 4)
    eeg = torch.randn(2, 4, 100)
    context = torch.randn(2, 4, 100)
    mask = factorized_mask(2, 4, 2, 10, 0.5, 0.5, 0.5, torch.device("cpu"))
    loss, _, _ = pretrainer(eeg, context, mask)
    loss.backward()
    gradients = [parameter.grad for parameter in pretrainer.parameters() if parameter.requires_grad]
    assert any(gradient is not None for gradient in gradients)
