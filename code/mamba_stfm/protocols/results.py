from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Score:
    mean: float
    deviation: float | None


@dataclass(frozen=True)
class MainResult:
    method: str
    family: str
    iv2a_accuracy: Score | None
    iv2a_kappa: float | None
    iv2b_accuracy: Score | None
    openbmi_accuracy: Score | None
    protocol: str


@dataclass(frozen=True)
class AblationResult:
    configuration: str
    accuracy: Score
    kappa: float
    delta: float
    factors: tuple[str, ...]


@dataclass(frozen=True)
class ProbeResult:
    setting: str
    finetune: Score
    probe: Score | None


@dataclass(frozen=True)
class ScaleResult:
    subjects: int
    sources: tuple[str, ...]
    accuracy: Score


@dataclass(frozen=True)
class LabelResult:
    fraction: float
    pretrained: float
    scratch: float


@dataclass(frozen=True)
class EfficiencyResult:
    method: str
    parameters: int | None
    macs_millions: float | None
    memory_40k_mb: float | None
    latency_ms: float | None
    scaling: str


MAIN_RESULTS = (
    MainResult(
        method="EEGNet",
        family="compact_cnn",
        iv2a_accuracy=Score(
            mean=52.0,
            deviation=1.2,
        ),
        iv2a_kappa=0.36,
        iv2b_accuracy=Score(
            mean=77.7,
            deviation=1.1,
        ),
        openbmi_accuracy=Score(
            mean=74.9,
            deviation=4.5,
        ),
        protocol="zero_calibration_loso",
    ),
    MainResult(
        method="ShallowConvNet",
        family="cnn",
        iv2a_accuracy=Score(
            mean=48.8,
            deviation=1.0,
        ),
        iv2a_kappa=0.32,
        iv2b_accuracy=Score(
            mean=74.5,
            deviation=0.9,
        ),
        openbmi_accuracy=Score(
            mean=75.2,
            deviation=5.4,
        ),
        protocol="zero_calibration_loso",
    ),
    MainResult(
        method="DeepConvNet",
        family="cnn",
        iv2a_accuracy=None,
        iv2a_kappa=None,
        iv2b_accuracy=None,
        openbmi_accuracy=Score(
            mean=76.9,
            deviation=3.5,
        ),
        protocol="zero_calibration_loso",
    ),
    MainResult(
        method="EEG-Conformer",
        family="cnn_attention",
        iv2a_accuracy=Score(
            mean=45.4,
            deviation=1.0,
        ),
        iv2a_kappa=0.27,
        iv2b_accuracy=Score(
            mean=73.4,
            deviation=0.9,
        ),
        openbmi_accuracy=Score(
            mean=77.9,
            deviation=5.3,
        ),
        protocol="zero_calibration_loso",
    ),
    MainResult(
        method="EEG-TCNet",
        family="tcn",
        iv2a_accuracy=Score(
            mean=55.1,
            deviation=1.1,
        ),
        iv2a_kappa=0.40,
        iv2b_accuracy=Score(
            mean=78.8,
            deviation=0.5,
        ),
        openbmi_accuracy=None,
        protocol="zero_calibration_loso",
    ),
    MainResult(
        method="CTNet",
        family="convolution_transformer",
        iv2a_accuracy=Score(
            mean=59.7,
            deviation=2.0,
        ),
        iv2a_kappa=0.46,
        iv2b_accuracy=Score(
            mean=79.4,
            deviation=0.6,
        ),
        openbmi_accuracy=None,
        protocol="zero_calibration_loso",
    ),
    MainResult(
        method="ATCNet",
        family="attention_convolution",
        iv2a_accuracy=Score(
            mean=60.1,
            deviation=1.9,
        ),
        iv2a_kappa=0.47,
        iv2b_accuracy=Score(
            mean=80.3,
            deviation=0.4,
        ),
        openbmi_accuracy=None,
        protocol="zero_calibration_loso",
    ),
    MainResult(
        method="S-Mamba",
        family="generic_mamba",
        iv2a_accuracy=None,
        iv2a_kappa=None,
        iv2b_accuracy=None,
        openbmi_accuracy=Score(
            mean=72.1,
            deviation=4.2,
        ),
        protocol="zero_calibration_loso",
    ),
    MainResult(
        method="TCFormer",
        family="tcn_transformer",
        iv2a_accuracy=Score(
            mean=62.4,
            deviation=1.4,
        ),
        iv2a_kappa=0.50,
        iv2b_accuracy=Score(
            mean=79.7,
            deviation=0.5,
        ),
        openbmi_accuracy=None,
        protocol="zero_calibration_loso",
    ),
    MainResult(
        method="Cortical-SSM",
        family="eeg_structured_ssm",
        iv2a_accuracy=None,
        iv2a_kappa=None,
        iv2b_accuracy=None,
        openbmi_accuracy=Score(
            mean=81.6,
            deviation=5.2,
        ),
        protocol="zero_calibration_loso",
    ),
    MainResult(
        method="Mamba ST-FM",
        family="selective_ssm_ssl",
        iv2a_accuracy=Score(
            mean=65.0,
            deviation=1.4,
        ),
        iv2a_kappa=0.53,
        iv2b_accuracy=Score(
            mean=81.5,
            deviation=0.5,
        ),
        openbmi_accuracy=Score(
            mean=82.8,
            deviation=4.8,
        ),
        protocol="zero_calibration_loso",
    ),
    MainResult(
        method="MIRepNet",
        family="mi_foundation_model",
        iv2a_accuracy=Score(
            mean=64.1,
            deviation=None,
        ),
        iv2a_kappa=None,
        iv2b_accuracy=None,
        openbmi_accuracy=None,
        protocol="few_shot_calibration",
    ),
)


ABLATIONS = (
    AblationResult(
        configuration="full",
        accuracy=Score(
            mean=65.0,
            deviation=1.4,
        ),
        kappa=0.53,
        delta=0.0,
        factors=(),
    ),
    AblationResult(
        configuration="without_sasg",
        accuracy=Score(
            mean=60.1,
            deviation=1.7,
        ),
        kappa=0.47,
        delta=-4.9,
        factors=("sasg",),
    ),
    AblationResult(
        configuration="from_scratch",
        accuracy=Score(
            mean=58.9,
            deviation=1.6,
        ),
        kappa=0.45,
        delta=-6.1,
        factors=("efm",),
    ),
    AblationResult(
        configuration="random_masking",
        accuracy=Score(
            mean=62.0,
            deviation=1.5,
        ),
        kappa=0.49,
        delta=-3.0,
        factors=("factorized_mask",),
    ),
    AblationResult(
        configuration="flattened_scan",
        accuracy=Score(
            mean=62.5,
            deviation=1.5,
        ),
        kappa=0.50,
        delta=-2.5,
        factors=("structured_scan",),
    ),
    AblationResult(
        configuration="without_sasg_and_ssl",
        accuracy=Score(
            mean=52.5,
            deviation=1.8,
        ),
        kappa=0.37,
        delta=-12.5,
        factors=("sasg", "efm"),
    ),
    AblationResult(
        configuration="without_sasg_and_scan",
        accuracy=Score(
            mean=56.7,
            deviation=1.7,
        ),
        kappa=0.42,
        delta=-8.3,
        factors=("sasg", "structured_scan"),
    ),
    AblationResult(
        configuration="without_ssl_and_scan",
        accuracy=Score(
            mean=56.3,
            deviation=1.6,
        ),
        kappa=0.42,
        delta=-8.7,
        factors=("efm", "structured_scan"),
    ),
)


PROBES = (
    ProbeResult(
        setting="from_scratch",
        finetune=Score(
            mean=58.9,
            deviation=1.6,
        ),
        probe=None,
    ),
    ProbeResult(
        setting="raw_waveform_target",
        finetune=Score(
            mean=61.7,
            deviation=1.6,
        ),
        probe=Score(
            mean=28.4,
            deviation=2.2,
        ),
    ),
    ProbeResult(
        setting="random_token_mask",
        finetune=Score(
            mean=62.0,
            deviation=1.5,
        ),
        probe=Score(
            mean=29.1,
            deviation=2.4,
        ),
    ),
    ProbeResult(
        setting="factorized_envelope",
        finetune=Score(
            mean=65.0,
            deviation=1.4,
        ),
        probe=Score(
            mean=41.8,
            deviation=2.3,
        ),
    ),
)


SCALES = (
    ScaleResult(
        subjects=0,
        sources=(),
        accuracy=Score(
            mean=58.9,
            deviation=1.6,
        ),
    ),
    ScaleResult(
        subjects=54,
        sources=("OpenBMI",),
        accuracy=Score(
            mean=62.1,
            deviation=1.5,
        ),
    ),
    ScaleResult(
        subjects=109,
        sources=("OpenBMI", "PhysioNet"),
        accuracy=Score(
            mean=63.8,
            deviation=1.5,
        ),
    ),
    ScaleResult(
        subjects=168,
        sources=("OpenBMI", "PhysioNet", "SHU", "HighGamma"),
        accuracy=Score(
            mean=63.6,
            deviation=1.5,
        ),
    ),
    ScaleResult(
        subjects=220,
        sources=("OpenBMI", "PhysioNet", "SHU", "HighGamma", "BCI_IV"),
        accuracy=Score(
            mean=65.0,
            deviation=1.4,
        ),
    ),
)


LABEL_EFFICIENCY = (
    LabelResult(
        fraction=0.10,
        pretrained=58.7,
        scratch=49.8,
    ),
    LabelResult(
        fraction=0.25,
        pretrained=61.9,
        scratch=54.1,
    ),
    LabelResult(
        fraction=0.50,
        pretrained=63.5,
        scratch=57.2,
    ),
    LabelResult(
        fraction=1.00,
        pretrained=65.0,
        scratch=58.9,
    ),
)


EFFICIENCY = (
    EfficiencyResult(
        method="EEGNet",
        parameters=2500,
        macs_millions=12.0,
        memory_40k_mb=None,
        latency_ms=4.0,
        scaling="linear",
    ),
    EfficiencyResult(
        method="EEG-TCNet",
        parameters=4272,
        macs_millions=6.8,
        memory_40k_mb=None,
        latency_ms=5.0,
        scaling="linear",
    ),
    EfficiencyResult(
        method="ATCNet",
        parameters=115000,
        macs_millions=29.0,
        memory_40k_mb=None,
        latency_ms=19.0,
        scaling="quadratic",
    ),
    EfficiencyResult(
        method="EEG-Conformer",
        parameters=789000,
        macs_millions=63.0,
        memory_40k_mb=None,
        latency_ms=31.0,
        scaling="quadratic",
    ),
    EfficiencyResult(
        method="Cortical-SSM",
        parameters=1600000,
        macs_millions=38.0,
        memory_40k_mb=298.0,
        latency_ms=35.0,
        scaling="linear",
    ),
    EfficiencyResult(
        method="EEGPT-Giant",
        parameters=1090000000,
        macs_millions=None,
        memory_40k_mb=None,
        latency_ms=None,
        scaling="quadratic",
    ),
    EfficiencyResult(
        method="Mamba ST-FM",
        parameters=1700000,
        macs_millions=41.0,
        memory_40k_mb=312.0,
        latency_ms=38.0,
        scaling="linear",
    ),
)


MEMORY_CURVE = {
    1000: {
        "mamba_stfm_mb": 22.0,
        "attention_mb": 41.0,
    },
    5000: {
        "mamba_stfm_mb": 58.0,
        "attention_mb": 980.0,
    },
    10000: {
        "mamba_stfm_mb": 102.0,
        "attention_mb": None,
    },
    40000: {
        "mamba_stfm_mb": 312.0,
        "attention_mb": None,
    },
}


def main_result(method: str) -> MainResult:
    for result in MAIN_RESULTS:
        if result.method == method:
            return result
    raise ValueError(f"unknown method {method}")


def ablation(configuration: str) -> AblationResult:
    for result in ABLATIONS:
        if result.configuration == configuration:
            return result
    raise ValueError(f"unknown ablation {configuration}")
