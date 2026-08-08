from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from mamba_stfm.engine.runtime import reduce_mean

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Batch:
    eeg: Tensor
    context: Tensor
    target: Tensor


@dataclass(frozen=True)
class EpochResult:
    loss: float
    accuracy: float
    examples: int


def make_optimizer(module: nn.Module, learning_rate: float, weight_decay: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> torch.optim.AdamW:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            destination = no_decay if parameter.ndim < 2 or name.endswith("bias") else decay
            destination.append(parameter)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=learning_rate, betas=(beta1, beta2), eps=epsilon)


def cosine_schedule(optimizer: torch.optim.Optimizer, epochs: int, steps_per_epoch: int, warmup_epochs: int = 0) -> torch.optim.lr_scheduler.LambdaLR:
    total = max(1, epochs * steps_per_epoch)
    warmup = warmup_epochs * steps_per_epoch

    def multiplier(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def unpack_batch(raw: tuple[Tensor, Tensor, Tensor, object], device: torch.device) -> Batch:
    eeg, context, target, _ = raw
    return Batch(eeg.to(device), context.to(device), target.to(device))


class SupervisedTrainer:
    def __init__(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, gradient_clip: float = 1.0, accumulation: int = 1, precision: str = "fp32"
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.gradient_clip = gradient_clip
        self.accumulation = accumulation
        self.precision = precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16" and device.type == "cuda")
        self.loss_function = nn.CrossEntropyLoss()

    def autocast(self) -> torch.amp.autocast:
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        enabled = self.precision in {"fp16", "bf16"} and self.device.type == "cuda"
        return torch.amp.autocast("cuda", dtype=dtype, enabled=enabled)

    def train_epoch(self, loader: Iterable[tuple[Tensor, Tensor, Tensor, object]]) -> EpochResult:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for step, raw in enumerate(loader):
            batch = unpack_batch(raw, self.device)
            with self.autocast():
                logits = self.model(batch.eeg, batch.context)
                loss = self.loss_function(logits, batch.target) / self.accumulation
            self.scaler.scale(loss).backward()
            if (step + 1) % self.accumulation == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().item()) * self.accumulation * batch.target.shape[0]
            total_correct += int((logits.argmax(dim=-1) == batch.target).sum().item())
            total_examples += batch.target.shape[0]
        if total_examples == 0:
            raise ValueError("training loader is empty")
        summary = torch.tensor((total_loss, total_correct, total_examples), device=self.device, dtype=torch.float64)
        summary = reduce_mean(summary)
        return EpochResult(float(summary[0] / summary[2]), float(summary[1] / summary[2]), int(summary[2].item()))

    @torch.no_grad()
    def evaluate(self, loader: Iterable[tuple[Tensor, Tensor, Tensor, object]]) -> EpochResult:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for raw in loader:
            batch = unpack_batch(raw, self.device)
            logits = self.model(batch.eeg, batch.context)
            total_loss += float(self.loss_function(logits, batch.target).item()) * batch.target.shape[0]
            total_correct += int((logits.argmax(dim=-1) == batch.target).sum().item())
            total_examples += batch.target.shape[0]
        if total_examples == 0:
            raise ValueError("evaluation loader is empty")
        return EpochResult(total_loss / total_examples, total_correct / total_examples, total_examples)
