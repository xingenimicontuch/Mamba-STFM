from __future__ import annotations

import pytest
import torch
from mamba_stfm.models.baselines import (
    Chomp1d,
    DeepConvNet,
    EEGConformer,
    EEGNet,
    EEGTCNet,
    MultiHeadSelfAttention,
    SafeLog,
    ShallowConvNet,
    Square,
    TemporalResidual,
    TransformerBlock,
    sinusoidal_position,
)


def test_square() -> None:
    signal = torch.tensor((-2.0, 3.0))
    result = Square()(signal)
    torch.testing.assert_close(
        result,
        torch.tensor((4.0, 9.0)),
    )


def test_safe_log_is_finite_at_zero() -> None:
    result = SafeLog()(torch.zeros(3))
    assert torch.isfinite(result).all()


def test_chomp_removes_right_padding() -> None:
    signal = torch.arange(10).reshape(1, 1, 10)
    result = Chomp1d(3)(signal)
    assert result.shape[-1] == 7


def test_temporal_residual_preserves_length() -> None:
    block = TemporalResidual(
        inputs=4,
        outputs=4,
        kernel=3,
        dilation=2,
        dropout=0.0,
    )
    signal = torch.randn(2, 4, 100)
    result = block(signal)
    assert result.shape == signal.shape


def test_attention_shape() -> None:
    attention = MultiHeadSelfAttention(
        dimension=16,
        heads=4,
        dropout=0.0,
    )
    tokens = torch.randn(2, 20, 16)
    result = attention(tokens)
    assert result.shape == tokens.shape


def test_attention_rejects_invalid_dimension() -> None:
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(
            dimension=15,
            heads=4,
            dropout=0.0,
        )


def test_transformer_block_shape() -> None:
    block = TransformerBlock(
        dimension=16,
        heads=4,
        expansion=2,
        dropout=0.0,
    )
    tokens = torch.randn(2, 20, 16)
    result = block(tokens)
    assert result.shape == tokens.shape


def test_position_encoding_shape() -> None:
    result = sinusoidal_position(
        length=100,
        dimension=16,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert result.shape == (100, 16)


def test_position_encoding_bounded() -> None:
    result = sinusoidal_position(
        length=100,
        dimension=16,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert result.abs().max() <= 1.0


def test_eegnet_output_shape() -> None:
    model = EEGNet(
        channels=4,
        samples=256,
        classes=3,
        temporal_kernel=32,
    )
    result = model(torch.randn(2, 4, 256))
    assert result.shape == (2, 3)


def test_shallow_convnet_output_shape() -> None:
    model = ShallowConvNet(
        channels=4,
        samples=256,
        classes=3,
        pool_kernel=25,
    )
    result = model(torch.randn(2, 4, 256))
    assert result.shape == (2, 3)


def test_deep_convnet_output_shape() -> None:
    model = DeepConvNet(
        channels=4,
        samples=1000,
        classes=3,
    )
    result = model(torch.randn(2, 4, 1000))
    assert result.shape == (2, 3)


def test_eegtcnet_output_shape() -> None:
    model = EEGTCNet(
        channels=4,
        classes=3,
        features=8,
        levels=2,
    )
    result = model(torch.randn(2, 4, 256))
    assert result.shape == (2, 3)


def test_eegconformer_output_shape() -> None:
    model = EEGConformer(
        channels=4,
        classes=3,
        dimension=16,
        temporal_kernel=15,
        pool=25,
        stride=10,
        depth=2,
        heads=4,
        dropout=0.0,
    )
    result = model(torch.randn(2, 4, 256))
    assert result.shape == (2, 3)


@pytest.mark.parametrize(
    "model",
    (
        EEGNet(4, 256, 3, temporal_kernel=32),
        ShallowConvNet(4, 256, 3, pool_kernel=25),
        EEGTCNet(4, 3, features=8, levels=2),
        EEGConformer(4, 3, dimension=16, temporal_kernel=15, pool=25, stride=10, depth=1, heads=4, dropout=0.0),
    ),
)
def test_baseline_backpropagation(model: torch.nn.Module) -> None:
    signal = torch.randn(2, 4, 256)
    loss = model(signal).square().mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
