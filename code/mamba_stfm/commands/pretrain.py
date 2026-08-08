from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mamba_stfm.commands.train import records_from_manifest
from mamba_stfm.data.records import EEGTrialDataset
from mamba_stfm.engine.checkpoint import atomic_save, checkpoint_payload
from mamba_stfm.engine.runtime import choose_device, set_seed
from mamba_stfm.engine.trainer import make_optimizer
from mamba_stfm.models.network import MambaSTFM
from mamba_stfm.objectives.efm import EFMPretrainer, factorized_mask
from mamba_stfm.settings import ExperimentSettings

LOGGER = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="mamba-stfm-pretrain")
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> None:
    arguments = parser().parse_args()
    settings = ExperimentSettings.load(arguments.config)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    device = choose_device(settings.runtime.device)
    set_seed(settings.runtime.seeds[0])
    samples = int(settings.signal.sample_rate * settings.signal.window_seconds)
    dataset = EEGTrialDataset(
        records_from_manifest(arguments.manifest),
        settings.model.channels,
        samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=settings.optimizer.batch_size,
        shuffle=True,
        num_workers=settings.runtime.workers,
    )
    encoder = MambaSTFM(
        settings.model,
        settings.signal,
    ).to(device)
    model = EFMPretrainer(
        encoder,
        settings.mask.decoder_dimension,
    ).to(device)
    optimizer = make_optimizer(
        model,
        settings.optimizer.learning_rate,
        settings.optimizer.weight_decay,
        settings.optimizer.beta1,
        settings.optimizer.beta2,
        settings.optimizer.epsilon,
    )
    for epoch in range(settings.optimizer.epochs):
        model.train()
        total = 0.0
        examples = 0
        for eeg, context, _, _ in loader:
            eeg = eeg.to(device)
            context = context.to(device)
            patches = encoder.tokenizer.token_count(eeg.shape[-1]) // len(encoder.tokenizer.clusters)
            mask = factorized_mask(
                eeg.shape[0],
                len(encoder.tokenizer.clusters),
                2,
                patches,
                settings.mask.cluster_ratio,
                settings.mask.band_ratio,
                settings.mask.time_ratio,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = model(
                eeg,
                context,
                mask,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                settings.optimizer.gradient_clip,
            )
            optimizer.step()
            total += float(loss.detach().item()) * eeg.shape[0]
            examples += eeg.shape[0]
        LOGGER.info(
            "epoch=%d loss=%.6f",
            epoch + 1,
            total / examples,
        )
    payload = checkpoint_payload(
        model,
        optimizer,
        settings.optimizer.epochs,
        len(loader) * settings.optimizer.epochs,
        settings.runtime.seeds[0],
        settings,
    )
    atomic_save(
        payload,
        arguments.output / "pretrained.pt",
    )


if __name__ == "__main__":
    main()
