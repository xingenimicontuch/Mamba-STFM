from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class StreamingResult:
    logits: Tensor
    starts: Tensor
    latency_ms: Tensor


@torch.no_grad()
def simulated_online(model: nn.Module, signal: Tensor, context: Tensor, window_samples: int, stride_samples: int) -> StreamingResult:
    if signal.ndim != 3:
        raise ValueError("signal must have batch, channel, time shape")
    logits: list[Tensor] = []
    starts: list[int] = []
    latencies: list[float] = []
    model.eval()
    for start in range(0, signal.shape[-1] - window_samples + 1, stride_samples):
        window = signal[..., start : start + window_samples]
        if window.device.type == "cuda":
            torch.cuda.synchronize(window.device)
        before = time.perf_counter_ns()
        output = model(window, context[..., : start + window_samples], True)
        if window.device.type == "cuda":
            torch.cuda.synchronize(window.device)
        after = time.perf_counter_ns()
        logits.append(output)
        starts.append(start)
        latencies.append((after - before) / 1_000_000.0)
    if not logits:
        raise ValueError("stream is shorter than one window")
    return StreamingResult(torch.stack(logits, dim=1), torch.tensor(starts), torch.tensor(latencies))


def latency_percentiles(latencies: Tensor) -> dict[str, float]:
    return {
        "mean": float(latencies.mean().item()),
        "p50": float(torch.quantile(latencies, 0.50).item()),
        "p95": float(torch.quantile(latencies, 0.95).item()),
        "p99": float(torch.quantile(latencies, 0.99).item()),
        "maximum": float(latencies.max().item()),
    }
