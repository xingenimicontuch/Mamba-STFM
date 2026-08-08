from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def distributed_ready() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def rank() -> int:
    return torch.distributed.get_rank() if distributed_ready() else 0


def world_size() -> int:
    return torch.distributed.get_world_size() if distributed_ready() else 1


def reduce_mean(value: Tensor) -> Tensor:
    if not distributed_ready():
        return value
    result = value.detach().clone()
    torch.distributed.all_reduce(result)
    return result / world_size()


def initialize_distributed(backend: str) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not distributed_ready():
        torch.distributed.init_process_group(backend=backend, init_method="env://")


@dataclass
class ExponentialAverage:
    decay: float
    value: float | None = None

    def update(self, current: float) -> float:
        self.value = current if self.value is None else self.decay * self.value + (1.0 - self.decay) * current
        return self.value


def gradient_norm(module: nn.Module) -> float:
    squares = [parameter.grad.detach().square().sum() for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.sqrt(torch.stack(squares).sum()).item()) if squares else 0.0
