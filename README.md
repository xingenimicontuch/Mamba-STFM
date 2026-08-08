# Mamba-Driven Spatio-Temporal Foundation Model for Cross-Subject Motor Imagery EEG Pattern Recognition

This package contains the Mamba ST-FM training and evaluation stack for zero-calibration, cross-subject motor-imagery EEG decoding. It combines channel-then-time EEG tokenization, ERD/ERS-aligned factorized masked pretraining, subject-adaptive selective-state gating, strict leave-one-subject-out evaluation, and simulated online sliding-window measurement.

## Installation

Python 3.10 or 3.11 is supported.

```bash
python -m venv .venv
.venv/bin/pip install -e .
```

Conda users can create the pinned CUDA environment with:

```bash
conda env create -f environment.yml
conda activate mamba-stfm
```

The container entry point runs the training command:

```bash
docker build -t mamba-stfm .
```

## Data

Accessible canonical data entry points and access terms are collected in `dataset_links.txt`. The main experiments use BCI Competition IV 2a, BCI Competition IV 2b, and OpenBMI. PhysioNet EEGMMIDB, SHU, and High-Gamma supply secondary transfer and pretraining evidence. The High-Gamma entry is omitted from the link list because its repository did not pass the access check. SHU is CC BY 4.0 but its archive password must be requested from its maintainer.

All input is band-pass filtered from 4 to 40 Hz, notch filtered at 50 Hz when that frequency lies below Nyquist, channel-wise z-scored, resampled to the experiment rate, and divided into 4-second task windows. A manifest is tab-separated with signal path, subject identifier, session identifier, integer label, and an optional unlabeled baseline-context path. Array files must be NumPy `.npy` or `.npz` with channel-by-time layout.

## Architecture

The main configuration uses six bidirectional selective-state blocks at dimension 128. Temporal patches are traversed in channel-cluster order. The label-free subject descriptor concatenates mu and beta band power per anatomical cluster with the upper triangle of the channel covariance matrix. Invalid or short context falls back to the running population descriptor.

EFM samples a coherent cluster-by-band-by-time mask at ratio 0.5 and reconstructs the mu/beta analytic envelope only at masked positions. Fine-tuning uses cross-entropy on source subjects while the held-out subject contributes no target labels.

## Pretraining

The reported pretraining schedule is AdamW for 200 epochs at learning rate 0.001 and weight decay 0.05. The combined public MI pool contains approximately 220 harmonized subjects.

```bash
python -m mamba_stfm.commands.pretrain --config configs/pretrain.yaml --manifest manifests/pretrain.tsv --output runs/pretrain
```

## Training

The reported cross-subject schedule is AdamW for 100 epochs at learning rate 0.0005, weight decay 0.05, batch size 64, and 20 independent seeds. Each held-out subject is evaluated by a separately fitted fold.

```bash
mamba-stfm-train --config configs/main.yaml --manifest manifests/fold-01-train.tsv --output runs/fold-01
```

Expected strict zero-calibration LOSO results are 65.0 ± 1.4% accuracy and 0.53 Cohen's kappa on IV-2a, 81.5 ± 0.5% accuracy on IV-2b, and 82.8 ± 4.8% accuracy on the eight-fold OpenBMI protocol. These are reference outcomes over the reported 20-seed protocol, not guarantees for altered preprocessing or data partitions.

## Evaluation

```bash
mamba-stfm-evaluate --config configs/main.yaml --manifest manifests/fold-01-test.tsv --weights runs/fold-01/seed-0.pt --output runs/fold-01/seed-0.json
```

Primary metrics are accuracy and Cohen's kappa for four-class IV-2a, and accuracy with macro-F1 for two-class datasets. Paired comparisons use two-sided Wilcoxon signed-rank tests with Holm-Bonferroni correction. The effect size is the paired standardized mean difference.

## Simulated online protocol

Inference advances a 4-second window every 250 ms while updating the descriptor only from unlabeled context. The reported 1.7M-parameter configuration requires about 41M MACs per window, reaches 38 ms per window on the reported edge-class GPU, and uses 312 MB peak inference memory at 40,000 timesteps. The paper does not identify the GPU model, VRAM capacity, training wall-clock, or storage footprint, so those quantities cannot be stated more precisely without hardware records.

## Experiment variants

The ablation configurations cover gate removal, training from scratch, random token masking, flattened scanning, joint gate and pretraining removal, joint gate and scan removal, joint pretraining and scan removal, raw-waveform reconstruction, frozen linear probing, pretraining-corpus scale, and label-efficiency fractions. Each variant changes only the named scientific factor.

## Tests

```bash
pytest -q
ruff check .
mypy --strict code/mamba_stfm
```

The tests exercise signal processing, scan ordering, descriptor fallback, the selective recurrence, factorized masking, LOSO separation, statistics, atomic persistence, and a two-update training integration path.
