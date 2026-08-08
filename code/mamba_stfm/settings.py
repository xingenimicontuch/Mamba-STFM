from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SignalSettings:
    sample_rate: int = 250
    window_seconds: float = 4.0
    stride_seconds: float = 0.25
    low_hz: float = 4.0
    high_hz: float = 40.0
    notch_hz: float = 50.0
    patch_samples: int = 25
    mu_band: tuple[float, float] = (8.0, 13.0)
    beta_band: tuple[float, float] = (13.0, 30.0)


@dataclass(frozen=True)
class ModelSettings:
    channels: int = 22
    classes: int = 4
    dimension: int = 128
    state_dimension: int = 128
    expansion: int = 2
    convolution_width: int = 4
    blocks: int = 6
    clusters: int = 4
    dropout: float = 0.1
    bidirectional: bool = True
    descriptor_min_samples: int = 64


@dataclass(frozen=True)
class MaskSettings:
    cluster_ratio: float = 0.5
    band_ratio: float = 0.5
    time_ratio: float = 0.5
    decoder_dimension: int = 64


@dataclass(frozen=True)
class OptimizerSettings:
    name: str = "adamw"
    learning_rate: float = 5e-4
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    epochs: int = 100
    batch_size: int = 64
    gradient_accumulation: int = 1
    gradient_clip: float = 1.0
    warmup_epochs: int = 0
    scheduler: str = "cosine"


@dataclass(frozen=True)
class RuntimeSettings:
    seeds: tuple[int, ...] = tuple(range(20))
    workers: int = 8
    precision: str = "fp32"
    distributed_backend: str = "nccl"
    output_directory: str = "runs"
    device: str = "auto"


@dataclass(frozen=True)
class ExperimentSettings:
    stage: str = "finetune"
    dataset: str = "BNCI2014_001"
    signal: SignalSettings = field(default_factory=SignalSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    mask: MaskSettings = field(default_factory=MaskSettings)
    optimizer: OptimizerSettings = field(default_factory=OptimizerSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ExperimentSettings:
        return cls(
            stage=str(raw.get("stage", "finetune")),
            dataset=str(raw.get("dataset", "BNCI2014_001")),
            signal=SignalSettings(**raw.get("signal", {})),
            model=ModelSettings(**raw.get("model", {})),
            mask=MaskSettings(**raw.get("mask", {})),
            optimizer=OptimizerSettings(**raw.get("optimizer", {})),
            runtime=RuntimeSettings(**raw.get("runtime", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> ExperimentSettings:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
        if not isinstance(value, dict):
            raise ValueError("configuration root must be a mapping")
        return cls.from_mapping(value)
