from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mamba_stfm.commands.train import records_from_manifest
from mamba_stfm.data.records import EEGTrialDataset
from mamba_stfm.engine.checkpoint import restore
from mamba_stfm.engine.runtime import choose_device
from mamba_stfm.metrics.classification import classification_metrics
from mamba_stfm.models.network import MambaSTFM
from mamba_stfm.settings import ExperimentSettings


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="mamba-stfm-evaluate")
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--weights", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


@torch.no_grad()
def main() -> None:
    arguments = parser().parse_args()
    settings = ExperimentSettings.load(arguments.config)
    device = choose_device(settings.runtime.device)
    dataset = EEGTrialDataset(records_from_manifest(arguments.manifest), settings.model.channels, int(settings.signal.sample_rate * settings.signal.window_seconds))
    loader = DataLoader(dataset, batch_size=settings.optimizer.batch_size, num_workers=settings.runtime.workers)
    model = MambaSTFM(settings.model, settings.signal).to(device)
    restore(arguments.weights, model, map_location=device)
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for eeg, context, target, _ in loader:
        logits = model(eeg.to(device), context.to(device))
        predictions.append(logits.argmax(dim=-1).cpu().numpy())
        targets.append(target.numpy())
    metrics = classification_metrics(np.concatenate(targets), np.concatenate(predictions))
    arguments.output.write_text(json.dumps({"accuracy": metrics.accuracy, "kappa": metrics.kappa, "macro_f1": metrics.macro_f1}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
