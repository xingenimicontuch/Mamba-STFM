from __future__ import annotations

import torch
from torch import Tensor, nn

from mamba_stfm.models.descriptor import SubjectDescriptor
from mamba_stfm.models.sasg import SelectiveBlock
from mamba_stfm.models.tokenizer import EEGStructuredTokenizer
from mamba_stfm.settings import ModelSettings, SignalSettings


class MambaSTFM(nn.Module):
    def __init__(self, model: ModelSettings, signal: SignalSettings) -> None:
        super().__init__()
        self.model_settings = model
        self.signal_settings = signal
        self.tokenizer = EEGStructuredTokenizer(model.channels, signal.patch_samples, model.dimension)
        self.descriptor = SubjectDescriptor(model.channels, self.tokenizer.clusters, signal.sample_rate, model.descriptor_min_samples)
        self.blocks = nn.ModuleList(
            SelectiveBlock(
                model.dimension,
                self.descriptor.output_dimension,
                model.state_dimension,
                model.expansion,
                model.convolution_width,
                model.dropout,
                model.bidirectional,
            )
            for _ in range(model.blocks)
        )
        self.final_norm = nn.LayerNorm(model.dimension)
        self.head = nn.Linear(model.dimension, model.classes)

    def encode(self, eeg: Tensor, context: Tensor, update_descriptor: bool = False) -> Tensor:
        tokens = self.tokenizer(eeg)
        descriptor = self.descriptor(context, update_descriptor)
        for block in self.blocks:
            tokens = block(tokens, descriptor)
        return self.final_norm(tokens)

    def forward(self, eeg: Tensor, context: Tensor, update_descriptor: bool = False) -> Tensor:
        encoded = self.encode(eeg, context, update_descriptor)
        return self.head(encoded.mean(dim=1))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class LinearProbe(nn.Module):
    def __init__(self, encoder: MambaSTFM, classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.classifier = nn.Linear(encoder.model_settings.dimension, classes)

    def forward(self, eeg: Tensor, context: Tensor) -> Tensor:
        with torch.no_grad():
            representation = self.encoder.encode(eeg, context).mean(dim=1)
        return self.classifier(representation)
