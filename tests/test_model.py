from __future__ import annotations

import torch
from mamba_stfm.models.descriptor import SubjectDescriptor, channel_covariance, upper_triangle
from mamba_stfm.models.network import LinearProbe, MambaSTFM
from mamba_stfm.models.sasg import BidirectionalSASG, SASGScan, SelectiveBlock
from mamba_stfm.models.tokenizer import EEGStructuredTokenizer, balanced_clusters
from mamba_stfm.settings import ModelSettings, SignalSettings


def small_settings() -> tuple[ModelSettings, SignalSettings]:
    model = ModelSettings(
        channels=4,
        classes=3,
        dimension=8,
        state_dimension=8,
        expansion=2,
        convolution_width=3,
        blocks=2,
        clusters=4,
        dropout=0.0,
        bidirectional=True,
        descriptor_min_samples=8,
    )
    signal = SignalSettings(
        sample_rate=50,
        window_seconds=2.0,
        stride_seconds=0.25,
        patch_samples=10,
    )
    return model, signal


def test_balanced_clusters_partition_channels() -> None:
    clusters = balanced_clusters(10, 4)
    flattened = sorted(index for cluster in clusters for index in cluster)
    assert flattened == list(range(10))


def test_small_montage_degenerates_to_channel_count() -> None:
    clusters = balanced_clusters(3, 4)
    assert clusters == ((0,), (1,), (2,))


def test_tokenizer_scan_order_and_shape() -> None:
    tokenizer = EEGStructuredTokenizer(4, 10, 8)
    signal = torch.randn(2, 4, 100)
    result = tokenizer(signal)
    assert result.shape == (2, 40, 8)
    assert tokenizer.scan_index(2, 3) == 14


def test_tokenizer_pads_partial_patch() -> None:
    tokenizer = EEGStructuredTokenizer(4, 10, 8)
    signal = torch.randn(2, 4, 93)
    result = tokenizer(signal)
    assert result.shape == (2, 40, 8)


def test_tokenizer_rejects_wrong_channel_count() -> None:
    tokenizer = EEGStructuredTokenizer(4, 10, 8)
    signal = torch.randn(2, 3, 100)
    try:
        tokenizer(signal)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong channel count was accepted")


def test_covariance_is_symmetric() -> None:
    context = torch.randn(2, 4, 100)
    result = channel_covariance(context)
    torch.testing.assert_close(result, result.transpose(-1, -2))


def test_upper_triangle_has_expected_size() -> None:
    covariance = torch.randn(2, 4, 4)
    result = upper_triangle(covariance)
    assert result.shape == (2, 10)


def test_descriptor_dimension_matches_definition() -> None:
    descriptor = SubjectDescriptor(4, ((0,), (1,), (2,), (3,)), 50.0, 8)
    assert descriptor.output_dimension == 18


def test_descriptor_falls_back_for_short_context() -> None:
    descriptor = SubjectDescriptor(4, ((0,), (1,), (2,), (3,)), 50.0, 64)
    context = torch.randn(2, 4, 20)
    result = descriptor(context)
    assert torch.equal(result, torch.zeros_like(result))


def test_descriptor_updates_population() -> None:
    descriptor = SubjectDescriptor(4, ((0,), (1,), (2,), (3,)), 50.0, 8)
    context = torch.randn(2, 4, 100)
    result = descriptor(context, update=True)
    assert descriptor.population_count.item() == 2
    torch.testing.assert_close(descriptor.population_mean, result.mean(dim=0))


def test_sasg_scan_shape() -> None:
    scan = SASGScan(8, 18, 8, 3)
    tokens = torch.randn(2, 12, 8)
    descriptor = torch.randn(2, 18)
    result = scan(tokens, descriptor)
    assert result.shape == tokens.shape


def test_bidirectional_scan_shape() -> None:
    scan = BidirectionalSASG(8, 18, 8, 3)
    tokens = torch.randn(2, 12, 8)
    descriptor = torch.randn(2, 18)
    result = scan(tokens, descriptor)
    assert result.shape == tokens.shape


def test_selective_block_backpropagates() -> None:
    block = SelectiveBlock(8, 18, 8, 2, 3, 0.0, True)
    tokens = torch.randn(2, 12, 8, requires_grad=True)
    descriptor = torch.randn(2, 18)
    block(tokens, descriptor).mean().backward()
    assert tokens.grad is not None


def test_network_forward_shape() -> None:
    model_settings, signal_settings = small_settings()
    model = MambaSTFM(model_settings, signal_settings)
    eeg = torch.randn(2, 4, 100)
    context = torch.randn(2, 4, 100)
    result = model(eeg, context)
    assert result.shape == (2, 3)


def test_network_encode_shape() -> None:
    model_settings, signal_settings = small_settings()
    model = MambaSTFM(model_settings, signal_settings)
    eeg = torch.randn(2, 4, 100)
    context = torch.randn(2, 4, 100)
    result = model.encode(eeg, context)
    assert result.shape == (2, 40, 8)


def test_network_parameter_count_is_positive() -> None:
    model_settings, signal_settings = small_settings()
    model = MambaSTFM(model_settings, signal_settings)
    assert model.parameter_count() > 0


def test_linear_probe_freezes_encoder() -> None:
    model_settings, signal_settings = small_settings()
    encoder = MambaSTFM(model_settings, signal_settings)
    probe = LinearProbe(encoder, 3)
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert all(parameter.requires_grad for parameter in probe.classifier.parameters())
