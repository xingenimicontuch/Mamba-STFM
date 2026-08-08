from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ConstraintConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        max_norm: float,
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.max_norm = max_norm

    def forward(self, signal: Tensor) -> Tensor:
        weight = torch.renorm(
            self.weight,
            p=2,
            dim=0,
            maxnorm=self.max_norm,
        )
        return torch.nn.functional.conv2d(
            signal,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class EEGNet(nn.Module):
    def __init__(
        self,
        channels: int,
        samples: int,
        classes: int,
        temporal_filters: int = 8,
        depth_multiplier: int = 2,
        separable_filters: int = 16,
        temporal_kernel: int = 64,
        separable_kernel: int = 16,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        spatial_filters = temporal_filters * depth_multiplier
        self.temporal = nn.Sequential(
            nn.Conv2d(
                1,
                temporal_filters,
                (1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
                bias=False,
            ),
            nn.BatchNorm2d(
                temporal_filters,
            ),
        )
        self.spatial = nn.Sequential(
            ConstraintConv2d(
                temporal_filters,
                spatial_filters,
                (channels, 1),
                max_norm=1.0,
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(
                spatial_filters,
            ),
            nn.ELU(),
            nn.AvgPool2d(
                (1, 4),
            ),
            nn.Dropout(
                dropout,
            ),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(
                spatial_filters,
                spatial_filters,
                (1, separable_kernel),
                padding=(0, separable_kernel // 2),
                groups=spatial_filters,
                bias=False,
            ),
            nn.Conv2d(
                spatial_filters,
                separable_filters,
                (1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(
                separable_filters,
            ),
            nn.ELU(),
            nn.AvgPool2d(
                (1, 8),
            ),
            nn.Dropout(
                dropout,
            ),
        )
        with torch.no_grad():
            example = torch.zeros(
                1,
                1,
                channels,
                samples,
            )
            features = self.separable(
                self.spatial(
                    self.temporal(
                        example,
                    )
                )
            ).numel()
        self.classifier = nn.Linear(
            features,
            classes,
        )

    def forward(self, eeg: Tensor) -> Tensor:
        signal = eeg.unsqueeze(1)
        signal = self.temporal(
            signal,
        )
        signal = self.spatial(
            signal,
        )
        signal = self.separable(
            signal,
        )
        return self.classifier(
            signal.flatten(1),
        )


class Square(nn.Module):
    def forward(self, signal: Tensor) -> Tensor:
        return signal.square()


class SafeLog(nn.Module):
    def forward(self, signal: Tensor) -> Tensor:
        return torch.log(
            signal.clamp_min(1e-6),
        )


class ShallowConvNet(nn.Module):
    def __init__(
        self,
        channels: int,
        samples: int,
        classes: int,
        temporal_filters: int = 40,
        spatial_filters: int = 40,
        temporal_kernel: int = 25,
        pool_kernel: int = 75,
        pool_stride: int = 15,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                temporal_filters,
                (1, temporal_kernel),
                bias=False,
            ),
            nn.Conv2d(
                temporal_filters,
                spatial_filters,
                (channels, 1),
                bias=False,
            ),
            nn.BatchNorm2d(
                spatial_filters,
            ),
            Square(),
            nn.AvgPool2d(
                (1, pool_kernel),
                stride=(1, pool_stride),
            ),
            SafeLog(),
            nn.Dropout(
                dropout,
            ),
        )
        with torch.no_grad():
            features = self.features(
                torch.zeros(
                    1,
                    1,
                    channels,
                    samples,
                )
            ).numel()
        self.classifier = nn.Linear(
            features,
            classes,
        )

    def forward(self, eeg: Tensor) -> Tensor:
        features = self.features(
            eeg.unsqueeze(1),
        )
        return self.classifier(
            features.flatten(1),
        )


class ConvBlock(nn.Module):
    def __init__(
        self,
        inputs: int,
        outputs: int,
        kernel: tuple[int, int],
        pool: tuple[int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(
                inputs,
                outputs,
                kernel,
                bias=False,
            ),
            nn.BatchNorm2d(
                outputs,
            ),
            nn.ELU(),
            nn.MaxPool2d(
                pool,
            ),
            nn.Dropout(
                dropout,
            ),
        )

    def forward(self, signal: Tensor) -> Tensor:
        return self.network(
            signal,
        )


class DeepConvNet(nn.Module):
    def __init__(
        self,
        channels: int,
        samples: int,
        classes: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(
                1,
                25,
                (1, 10),
                bias=False,
            ),
            nn.Conv2d(
                25,
                25,
                (channels, 1),
                bias=False,
            ),
            nn.BatchNorm2d(
                25,
            ),
            nn.ELU(),
            nn.MaxPool2d(
                (1, 3),
            ),
            nn.Dropout(
                dropout,
            ),
        )
        self.blocks = nn.Sequential(
            ConvBlock(
                25,
                50,
                (1, 10),
                (1, 3),
                dropout,
            ),
            ConvBlock(
                50,
                100,
                (1, 10),
                (1, 3),
                dropout,
            ),
            ConvBlock(
                100,
                200,
                (1, 10),
                (1, 3),
                dropout,
            ),
        )
        with torch.no_grad():
            example = torch.zeros(
                1,
                1,
                channels,
                samples,
            )
            features = self.blocks(
                self.first(
                    example,
                )
            ).numel()
        self.classifier = nn.Linear(
            features,
            classes,
        )

    def forward(self, eeg: Tensor) -> Tensor:
        signal = self.first(
            eeg.unsqueeze(1),
        )
        signal = self.blocks(
            signal,
        )
        return self.classifier(
            signal.flatten(1),
        )


class Chomp1d(nn.Module):
    def __init__(
        self,
        amount: int,
    ) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, signal: Tensor) -> Tensor:
        return signal[..., : -self.amount] if self.amount else signal


class TemporalResidual(nn.Module):
    def __init__(
        self,
        inputs: int,
        outputs: int,
        kernel: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel - 1) * dilation
        self.first = nn.Sequential(
            nn.Conv1d(
                inputs,
                outputs,
                kernel,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            Chomp1d(
                padding,
            ),
            nn.BatchNorm1d(
                outputs,
            ),
            nn.ELU(),
            nn.Dropout(
                dropout,
            ),
        )
        self.second = nn.Sequential(
            nn.Conv1d(
                outputs,
                outputs,
                kernel,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            Chomp1d(
                padding,
            ),
            nn.BatchNorm1d(
                outputs,
            ),
            nn.ELU(),
            nn.Dropout(
                dropout,
            ),
        )
        self.residual = (
            nn.Conv1d(
                inputs,
                outputs,
                1,
            )
            if inputs != outputs
            else nn.Identity()
        )

    def forward(self, signal: Tensor) -> Tensor:
        residual = self.residual(
            signal,
        )
        signal = self.first(
            signal,
        )
        signal = self.second(
            signal,
        )
        return torch.nn.functional.elu(
            signal + residual,
        )


class EEGTCNet(nn.Module):
    def __init__(
        self,
        channels: int,
        classes: int,
        features: int = 16,
        levels: int = 3,
        kernel: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv1d(
                channels,
                features,
                16,
                padding=8,
                bias=False,
            ),
            nn.BatchNorm1d(
                features,
            ),
            nn.ELU(),
        )
        blocks: list[nn.Module] = []
        for index in range(levels):
            blocks.append(
                TemporalResidual(
                    features,
                    features,
                    kernel,
                    2**index,
                    dropout,
                )
            )
        self.temporal = nn.Sequential(
            *blocks,
        )
        self.classifier = nn.Linear(
            features,
            classes,
        )

    def forward(self, eeg: Tensor) -> Tensor:
        signal = self.spatial(
            eeg,
        )
        signal = self.temporal(
            signal,
        )
        return self.classifier(
            signal[..., -1],
        )


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("dimension must be divisible by heads")
        self.dimension = dimension
        self.heads = heads
        self.head_dimension = dimension // heads
        self.scale = self.head_dimension**-0.5
        self.qkv = nn.Linear(
            dimension,
            3 * dimension,
        )
        self.dropout = nn.Dropout(
            dropout,
        )
        self.output = nn.Linear(
            dimension,
            dimension,
        )

    def forward(self, tokens: Tensor) -> Tensor:
        batch, length, _ = tokens.shape
        qkv = (
            self.qkv(
                tokens,
            )
            .reshape(
                batch,
                length,
                3,
                self.heads,
                self.head_dimension,
            )
            .permute(
                2,
                0,
                3,
                1,
                4,
            )
        )
        query, key, value = qkv.unbind(0)
        scores = query @ key.transpose(-1, -2) * self.scale
        weights = self.dropout(
            scores.softmax(dim=-1),
        )
        attended = weights @ value
        attended = attended.transpose(1, 2).reshape(
            batch,
            length,
            self.dimension,
        )
        return self.output(
            attended,
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(
            dimension,
        )
        self.attention = MultiHeadSelfAttention(
            dimension,
            heads,
            dropout,
        )
        self.feed_norm = nn.LayerNorm(
            dimension,
        )
        self.feed = nn.Sequential(
            nn.Linear(
                dimension,
                dimension * expansion,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout,
            ),
            nn.Linear(
                dimension * expansion,
                dimension,
            ),
            nn.Dropout(
                dropout,
            ),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = tokens + self.attention(
            self.attention_norm(
                tokens,
            )
        )
        return tokens + self.feed(
            self.feed_norm(
                tokens,
            )
        )


class EEGConformer(nn.Module):
    def __init__(
        self,
        channels: int,
        classes: int,
        dimension: int = 40,
        temporal_kernel: int = 25,
        pool: int = 75,
        stride: int = 15,
        depth: int = 6,
        heads: int = 10,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv2d(
                1,
                dimension,
                (1, temporal_kernel),
                bias=False,
            ),
            nn.Conv2d(
                dimension,
                dimension,
                (channels, 1),
                bias=False,
            ),
            nn.BatchNorm2d(
                dimension,
            ),
            nn.ELU(),
            nn.AvgPool2d(
                (1, pool),
                stride=(1, stride),
            ),
            nn.Dropout(
                dropout,
            ),
        )
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    dimension,
                    heads,
                    4,
                    dropout,
                )
                for _ in range(depth)
            ],
        )
        self.norm = nn.LayerNorm(
            dimension,
        )
        self.classifier = nn.Linear(
            dimension,
            classes,
        )

    def forward(self, eeg: Tensor) -> Tensor:
        features = self.patch(
            eeg.unsqueeze(1),
        )
        tokens = features.squeeze(2).transpose(1, 2)
        tokens = self.blocks(
            tokens,
        )
        return self.classifier(
            self.norm(
                tokens,
            ).mean(dim=1),
        )


def sinusoidal_position(length: int, dimension: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    position = torch.arange(
        length,
        device=device,
        dtype=dtype,
    ).unsqueeze(1)
    divisor = torch.exp(
        torch.arange(
            0,
            dimension,
            2,
            device=device,
            dtype=dtype,
        )
        * (-math.log(10000.0) / dimension)
    )
    encoding = torch.zeros(
        length,
        dimension,
        device=device,
        dtype=dtype,
    )
    encoding[:, 0::2] = torch.sin(
        position * divisor,
    )
    encoding[:, 1::2] = torch.cos(
        position * divisor[: encoding[:, 1::2].shape[1]],
    )
    return encoding
