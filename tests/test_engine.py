from __future__ import annotations

from pathlib import Path

import torch
from mamba_stfm.analysis.streaming import latency_percentiles, simulated_online
from mamba_stfm.data.records import TensorTrials
from mamba_stfm.engine.checkpoint import atomic_save, checkpoint_payload, restore
from mamba_stfm.engine.runtime import ExponentialAverage, choose_device, gradient_norm, set_seed
from mamba_stfm.engine.trainer import SupervisedTrainer, cosine_schedule, make_optimizer
from mamba_stfm.models.network import MambaSTFM
from mamba_stfm.settings import ExperimentSettings, ModelSettings, OptimizerSettings, RuntimeSettings, SignalSettings
from torch import nn
from torch.utils.data import DataLoader


def settings() -> ExperimentSettings:
    return ExperimentSettings(
        stage="finetune",
        dataset="synthetic",
        signal=SignalSettings(
            sample_rate=20,
            window_seconds=2.0,
            stride_seconds=0.5,
            patch_samples=10,
        ),
        model=ModelSettings(
            channels=2,
            classes=2,
            dimension=4,
            state_dimension=4,
            expansion=2,
            convolution_width=2,
            blocks=1,
            clusters=2,
            dropout=0.0,
            bidirectional=False,
            descriptor_min_samples=4,
        ),
        optimizer=OptimizerSettings(
            learning_rate=0.01,
            weight_decay=0.0,
            epochs=2,
            batch_size=4,
            gradient_clip=1.0,
        ),
        runtime=RuntimeSettings(
            seeds=(0,),
            workers=0,
            precision="fp32",
            device="cpu",
        ),
    )


def dataset() -> TensorTrials:
    generator = torch.Generator().manual_seed(7)
    signals = torch.randn(8, 2, 40, generator=generator)
    labels = (signals[:, 0].mean(dim=-1) > 0.0).long()
    contexts = torch.randn(8, 2, 40, generator=generator)
    subjects = [f"s{index // 2}" for index in range(8)]
    return TensorTrials(signals, contexts, labels, subjects)


def test_seed_repeats_torch_randomness() -> None:
    set_seed(9)
    first = torch.randn(10)
    set_seed(9)
    second = torch.randn(10)
    assert torch.equal(first, second)


def test_choose_device_cpu() -> None:
    assert choose_device("cpu") == torch.device("cpu")


def test_exponential_average() -> None:
    average = ExponentialAverage(0.5)
    assert average.update(2.0) == 2.0
    assert average.update(4.0) == 3.0


def test_gradient_norm_without_gradients_is_zero() -> None:
    assert gradient_norm(nn.Linear(2, 2)) == 0.0


def test_gradient_norm_after_backward_is_positive() -> None:
    model = nn.Linear(2, 2)
    model(torch.randn(3, 2)).sum().backward()
    assert gradient_norm(model) > 0.0


def test_optimizer_separates_decay_groups() -> None:
    optimizer = make_optimizer(nn.Linear(2, 2), 0.01, 0.05)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == 0.05
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


def test_cosine_schedule_starts_high_without_warmup() -> None:
    model = nn.Linear(2, 2)
    optimizer = make_optimizer(model, 0.01, 0.0)
    scheduler = cosine_schedule(optimizer, 2, 5)
    values = []
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        values.append(optimizer.param_groups[0]["lr"])
    assert values[-1] <= values[0]


def test_atomic_checkpoint_roundtrip(tmp_path: Path) -> None:
    experiment = settings()
    model = MambaSTFM(experiment.model, experiment.signal)
    optimizer = make_optimizer(model, 0.01, 0.0)
    payload = checkpoint_payload(model, optimizer, 1, 2, 3, experiment)
    destination = tmp_path / "state.pt"
    atomic_save(payload, destination)
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    for parameter in model.parameters():
        parameter.data.zero_()
    restored = restore(destination, model, optimizer)
    assert restored["seed"] == 3
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, original[name])


def test_training_epoch_returns_finite_metrics() -> None:
    experiment = settings()
    model = MambaSTFM(experiment.model, experiment.signal)
    optimizer = make_optimizer(model, 0.01, 0.0)
    trainer = SupervisedTrainer(model, optimizer, torch.device("cpu"), 1.0, 1, "fp32")
    loader = DataLoader(dataset(), batch_size=4, shuffle=False)
    result = trainer.train_epoch(loader)
    assert result.loss > 0.0
    assert 0.0 <= result.accuracy <= 1.0
    assert result.examples == 8


def test_evaluation_epoch_returns_all_examples() -> None:
    experiment = settings()
    model = MambaSTFM(experiment.model, experiment.signal)
    optimizer = make_optimizer(model, 0.01, 0.0)
    trainer = SupervisedTrainer(model, optimizer, torch.device("cpu"), 1.0, 1, "fp32")
    loader = DataLoader(dataset(), batch_size=4, shuffle=False)
    result = trainer.evaluate(loader)
    assert result.examples == 8


def test_two_training_updates_change_parameters() -> None:
    experiment = settings()
    model = MambaSTFM(experiment.model, experiment.signal)
    optimizer = make_optimizer(model, 0.01, 0.0)
    trainer = SupervisedTrainer(model, optimizer, torch.device("cpu"), 1.0, 1, "fp32")
    loader = DataLoader(dataset(), batch_size=4, shuffle=False)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    trainer.train_epoch(loader)
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters())


def test_simulated_online_window_count() -> None:
    experiment = settings()
    model = MambaSTFM(experiment.model, experiment.signal)
    signal = torch.randn(1, 2, 80)
    context = torch.randn(1, 2, 80)
    result = simulated_online(model, signal, context, 40, 10)
    assert result.logits.shape == (1, 5, 2)
    assert result.starts.tolist() == [0, 10, 20, 30, 40]


def test_latency_percentiles_are_ordered() -> None:
    values = torch.tensor((1.0, 2.0, 3.0, 4.0, 5.0))
    result = latency_percentiles(values)
    assert result["mean"] == 3.0
    assert result["p50"] <= result["p95"] <= result["p99"] <= result["maximum"]
