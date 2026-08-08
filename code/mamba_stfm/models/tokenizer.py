from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def balanced_clusters(channels: int, groups: int = 4) -> tuple[tuple[int, ...], ...]:
    count = min(channels, groups)
    return tuple(tuple(range(index, channels, count)) for index in range(count))


class EEGStructuredTokenizer(nn.Module):
    def __init__(self, channels: int, patch_samples: int, dimension: int, clusters: Sequence[Sequence[int]] | None = None) -> None:
        super().__init__()
        self.channels = channels
        self.patch_samples = patch_samples
        self.dimension = dimension
        chosen = balanced_clusters(channels) if clusters is None else tuple(tuple(group) for group in clusters)
        if not chosen or any(not group for group in chosen):
            raise ValueError("clusters must be nonempty")
        covered = sorted(index for group in chosen for index in group)
        if covered != list(range(channels)):
            raise ValueError("clusters must partition all channels")
        self.clusters = chosen
        self.projections = nn.ModuleList(nn.Linear(len(group) * patch_samples, dimension) for group in chosen)
        self.cluster_embedding = nn.Parameter(torch.empty(len(chosen), dimension))
        nn.init.normal_(self.cluster_embedding, std=0.02)

    def forward(self, eeg: Tensor) -> Tensor:
        if eeg.ndim != 3 or eeg.shape[1] != self.channels:
            raise ValueError("expected batch, channel, time EEG tensor")
        batch, _, samples = eeg.shape
        patches = (samples + self.patch_samples - 1) // self.patch_samples
        padded_samples = patches * self.patch_samples
        if padded_samples != samples:
            eeg = torch.nn.functional.pad(eeg, (0, padded_samples - samples))
        outputs: list[Tensor] = []
        for patch in range(patches):
            segment = eeg[:, :, patch * self.patch_samples : (patch + 1) * self.patch_samples]
            for cluster_index, channels in enumerate(self.clusters):
                selected = segment[:, channels, :].reshape(batch, -1)
                token = self.projections[cluster_index](selected)
                outputs.append(token + self.cluster_embedding[cluster_index])
        return torch.stack(outputs, dim=1)

    def scan_index(self, cluster: int, patch: int) -> int:
        return patch * len(self.clusters) + cluster

    def token_count(self, samples: int) -> int:
        return len(self.clusters) * ((samples + self.patch_samples - 1) // self.patch_samples)
