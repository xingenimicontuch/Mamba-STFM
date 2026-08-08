from __future__ import annotations

import torch
from torch import Tensor, nn

from mamba_stfm.signal.filters import rfft_band_envelope


def upper_triangle(covariance: Tensor) -> Tensor:
    channels = covariance.shape[-1]
    row, column = torch.triu_indices(channels, channels, device=covariance.device)
    return covariance[..., row, column]


def channel_covariance(context: Tensor, epsilon: float = 1e-5) -> Tensor:
    centered = context - context.mean(dim=-1, keepdim=True)
    denominator = max(context.shape[-1] - 1, 1)
    covariance = centered @ centered.transpose(-1, -2) / denominator
    identity = torch.eye(context.shape[-2], device=context.device, dtype=context.dtype)
    return covariance + epsilon * identity


class SubjectDescriptor(nn.Module):
    def __init__(self, channels: int, clusters: tuple[tuple[int, ...], ...], sample_rate: float, minimum_samples: int = 64) -> None:
        super().__init__()
        self.channels = channels
        self.clusters = clusters
        self.sample_rate = sample_rate
        self.minimum_samples = minimum_samples
        size = 2 * len(clusters) + channels * (channels + 1) // 2
        self.population_mean: Tensor
        self.population_count: Tensor
        self.register_buffer("population_mean", torch.zeros(size))
        self.register_buffer("population_count", torch.zeros((), dtype=torch.long))

    @property
    def output_dimension(self) -> int:
        return self.population_mean.numel()

    def informative(self, context: Tensor) -> Tensor:
        finite = torch.isfinite(context).all(dim=-1).all(dim=-1)
        variable = context.std(dim=-1).mean(dim=-1) > 1e-5
        enough = torch.full_like(finite, context.shape[-1] >= self.minimum_samples)
        return finite & variable & enough

    def estimate(self, context: Tensor) -> Tensor:
        if context.ndim != 3 or context.shape[1] != self.channels:
            raise ValueError("context must have batch, channel, time shape")
        safe = torch.nan_to_num(context)
        powers: list[Tensor] = []
        for indices in self.clusters:
            cluster = safe[:, indices, :].mean(dim=1)
            mu = rfft_band_envelope(cluster, self.sample_rate, 8.0, 13.0).square().mean(dim=-1)
            beta = rfft_band_envelope(cluster, self.sample_rate, 13.0, 30.0).square().mean(dim=-1)
            powers.extend((mu, beta))
        covariance = upper_triangle(channel_covariance(safe))
        descriptor = torch.cat((*[item.unsqueeze(-1) for item in powers], covariance), dim=-1)
        valid = self.informative(context).unsqueeze(-1)
        fallback = self.population_mean.to(descriptor).expand_as(descriptor)
        return torch.where(valid, descriptor, fallback)

    @torch.no_grad()
    def update_population(self, descriptor: Tensor) -> None:
        count = descriptor.shape[0]
        if count == 0:
            return
        old_count = int(self.population_count.item())
        total = old_count + count
        combined = (self.population_mean * old_count + descriptor.detach().sum(dim=0)) / total
        self.population_mean.copy_(combined)
        self.population_count.fill_(total)

    def forward(self, context: Tensor, update: bool = False) -> Tensor:
        descriptor = self.estimate(context)
        if update:
            self.update_population(descriptor)
        return descriptor
