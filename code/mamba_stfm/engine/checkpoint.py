from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mamba_stfm.settings import ExperimentSettings


def checkpoint_payload(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, step: int, seed: int, settings: ExperimentSettings) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "seed": seed,
        "settings": asdict(settings),
        "rng_cpu": torch.random.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def atomic_save(payload: dict[str, Any], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    torch.random.set_rng_state(payload["rng_cpu"])
    if torch.cuda.is_available() and payload["rng_cuda"]:
        torch.cuda.set_rng_state_all(payload["rng_cuda"])
    return payload
