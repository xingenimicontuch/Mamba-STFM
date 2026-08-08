from __future__ import annotations

import argparse
import logging
from pathlib import Path

from torch.utils.data import DataLoader

from mamba_stfm.data.records import EEGTrialDataset, TrialRecord
from mamba_stfm.engine.checkpoint import atomic_save, checkpoint_payload
from mamba_stfm.engine.runtime import choose_device, set_seed
from mamba_stfm.engine.trainer import SupervisedTrainer, make_optimizer
from mamba_stfm.models.network import MambaSTFM
from mamba_stfm.settings import ExperimentSettings


def records_from_manifest(path: Path) -> list[TrialRecord]:
    records: list[TrialRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        signal, subject, session, label, context = line.split("\t")
        records.append(TrialRecord(Path(signal), subject, session, int(label), Path(context) if context else None))
    return records


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="mamba-stfm-train")
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> None:
    arguments = parser().parse_args()
    settings = ExperimentSettings.load(arguments.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    device = choose_device(settings.runtime.device)
    for seed in settings.runtime.seeds:
        set_seed(seed)
        dataset = EEGTrialDataset(records_from_manifest(arguments.manifest), settings.model.channels, int(settings.signal.sample_rate * settings.signal.window_seconds))
        loader = DataLoader(dataset, batch_size=settings.optimizer.batch_size, shuffle=True, num_workers=settings.runtime.workers)
        model = MambaSTFM(settings.model, settings.signal).to(device)
        optimizer = make_optimizer(model, settings.optimizer.learning_rate, settings.optimizer.weight_decay)
        trainer = SupervisedTrainer(model, optimizer, device, settings.optimizer.gradient_clip, settings.optimizer.gradient_accumulation, settings.runtime.precision)
        for epoch in range(settings.optimizer.epochs):
            result = trainer.train_epoch(loader)
            logging.info("seed=%d epoch=%d loss=%.6f accuracy=%.4f", seed, epoch + 1, result.loss, result.accuracy)
        payload = checkpoint_payload(model, optimizer, settings.optimizer.epochs, len(loader) * settings.optimizer.epochs, seed, settings)
        atomic_save(payload, arguments.output / f"seed-{seed}.pt")


if __name__ == "__main__":
    main()
