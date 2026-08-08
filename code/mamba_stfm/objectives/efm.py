from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from mamba_stfm.models.network import MambaSTFM
from mamba_stfm.signal.filters import dual_band_envelope


@dataclass(frozen=True)
class FactorizedMask:
    cluster: Tensor
    band: Tensor
    time: Tensor
    combined: Tensor


def factorized_mask(batch: int, clusters: int, bands: int, times: int, cluster_ratio: float, band_ratio: float, time_ratio: float, device: torch.device) -> FactorizedMask:
    cluster = torch.rand(batch, clusters, 1, 1, device=device) < cluster_ratio
    band = torch.rand(batch, 1, bands, 1, device=device) < band_ratio
    time = torch.rand(batch, 1, 1, times, device=device) < time_ratio
    combined = cluster | band | time
    return FactorizedMask(cluster, band, time, combined)


def masked_mean_squared_error(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    expanded = torch.broadcast_to(mask, target.shape)
    denominator = expanded.sum().clamp_min(1)
    return ((prediction - target).square() * expanded).sum() / denominator


class EnvelopeDecoder(nn.Module):
    def __init__(self, dimension: int, clusters: int, patch_samples: int, hidden: int = 64) -> None:
        super().__init__()
        self.clusters = clusters
        self.patch_samples = patch_samples
        self.network = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(), nn.Linear(hidden, 2 * patch_samples))

    def forward(self, tokens: Tensor) -> Tensor:
        batch, length, _ = tokens.shape
        if length % self.clusters != 0:
            raise ValueError("token count must be divisible by cluster count")
        patches = length // self.clusters
        decoded = self.network(tokens).reshape(batch, patches, self.clusters, 2, self.patch_samples)
        return decoded.permute(0, 2, 3, 1, 4).reshape(batch, self.clusters, 2, patches * self.patch_samples)


class EFMPretrainer(nn.Module):
    def __init__(self, encoder: MambaSTFM, hidden: int = 64) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = EnvelopeDecoder(encoder.model_settings.dimension, len(encoder.tokenizer.clusters), encoder.signal_settings.patch_samples, hidden)

    def cluster_signal(self, eeg: Tensor) -> Tensor:
        return torch.stack([eeg[:, indices, :].mean(dim=1) for indices in self.encoder.tokenizer.clusters], dim=1)

    def target(self, eeg: Tensor) -> Tensor:
        return dual_band_envelope(self.cluster_signal(eeg), self.encoder.signal_settings.sample_rate)

    def forward(self, eeg: Tensor, context: Tensor, mask: FactorizedMask) -> tuple[Tensor, Tensor, Tensor]:
        cluster_mask = mask.combined.any(dim=2).to(eeg.dtype)
        samples = eeg.shape[-1]
        expanded = torch.nn.functional.interpolate(cluster_mask, size=samples, mode="nearest")
        visible = eeg.clone()
        for index, channels in enumerate(self.encoder.tokenizer.clusters):
            visible[:, channels, :] = visible[:, channels, :] * (1.0 - expanded[:, index : index + 1, :])
        encoded = self.encoder.encode(visible, context)
        prediction = self.decoder(encoded)[..., :samples]
        target = self.target(eeg)
        target_mask = torch.nn.functional.interpolate(mask.combined.to(eeg.dtype).flatten(1, 2), size=samples, mode="nearest").reshape_as(target)
        loss = masked_mean_squared_error(prediction, target, target_mask)
        return loss, prediction, target
