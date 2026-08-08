from __future__ import annotations

import numpy as np
import pytest
from mamba_stfm.data.catalog import DATASETS, channel_indices, common_channels, get_dataset
from mamba_stfm.data.splits import leave_one_subject_out, openbmi_eight_folds, stratified_fraction
from mamba_stfm.metrics.classification import (
    classification_metrics,
    holm_bonferroni,
    interaction_ratio,
    mean_standard_deviation,
    paired_standardized_mean_difference,
    paired_wilcoxon,
)
from mamba_stfm.protocols.results import ABLATIONS, EFFICIENCY, LABEL_EFFICIENCY, MAIN_RESULTS, MEMORY_CURVE, PROBES, SCALES, ablation, main_result


def test_catalog_contains_six_datasets() -> None:
    assert len(DATASETS) == 6


@pytest.mark.parametrize(
    ("code", "subjects", "rate", "classes"),
    (
        ("BNCI2014_001", 9, 250, 4),
        ("BNCI2014_004", 9, 250, 2),
        ("Lee2019_MI", 54, 1000, 2),
        ("PhysionetMI", 109, 160, 4),
        ("Shu2022", 25, 250, 2),
        ("Schirrmeister2017", 14, 500, 4),
    ),
)
def test_catalog_values(code: str, subjects: int, rate: int, classes: int) -> None:
    specification = get_dataset(code)
    assert specification.subjects == subjects
    assert specification.sample_rate == rate
    assert len(specification.classes) == classes


def test_unknown_dataset_is_rejected() -> None:
    with pytest.raises(ValueError):
        get_dataset("missing")


def test_common_channels_preserves_first_order() -> None:
    result = common_channels(("BNCI2014_001", "Lee2019_MI"))
    assert result[0] == "Fz"
    assert "C3" in result
    assert "C4" in result


def test_channel_indices_are_case_insensitive() -> None:
    result = channel_indices(("FZ", "C3", "cz"), ("Fz", "Cz"))
    assert result == (0, 2)


def test_channel_indices_reject_missing_channel() -> None:
    with pytest.raises(ValueError):
        channel_indices(("Fz", "Cz"), ("C3",))


def test_loso_produces_one_fold_per_subject() -> None:
    subjects = ["a", "a", "b", "b", "c", "c"]
    folds = list(leave_one_subject_out(subjects))
    assert len(folds) == 3
    assert {fold.held_out for fold in folds} == {"a", "b", "c"}


def test_loso_never_leaks_held_out_subject() -> None:
    subjects = np.asarray(["a", "a", "b", "b", "c", "c"])
    for fold in leave_one_subject_out(subjects.tolist()):
        assert np.all(subjects[fold.train_indices] != fold.held_out)
        assert np.all(subjects[fold.test_indices] == fold.held_out)


def test_openbmi_has_eight_folds() -> None:
    subjects = [f"s{index:02d}" for index in range(54) for _ in range(2)]
    folds = openbmi_eight_folds(subjects, seed=3)
    assert len(folds) == 8
    assert sum(test.size for _, test in folds) == len(subjects)


def test_openbmi_subjects_do_not_cross_partition() -> None:
    subjects = np.asarray([f"s{index:02d}" for index in range(54) for _ in range(2)])
    for train, test in openbmi_eight_folds(subjects.tolist(), seed=3):
        assert set(subjects[train]).isdisjoint(set(subjects[test]))


@pytest.mark.parametrize("fraction", (0.1, 0.25, 0.5, 1.0))
def test_stratified_fraction_keeps_every_class(fraction: float) -> None:
    labels = np.repeat(np.arange(4), 20)
    selected = stratified_fraction(labels, fraction, seed=5)
    assert set(labels[selected]) == {0, 1, 2, 3}


def test_stratified_fraction_is_deterministic() -> None:
    labels = np.repeat(np.arange(4), 20)
    first = stratified_fraction(labels, 0.25, seed=5)
    second = stratified_fraction(labels, 0.25, seed=5)
    np.testing.assert_array_equal(first, second)


def test_classification_metrics_perfect_prediction() -> None:
    target = np.array((0, 1, 2, 3))
    metrics = classification_metrics(target, target)
    assert metrics.accuracy == 1.0
    assert metrics.kappa == 1.0
    assert metrics.macro_f1 == 1.0


def test_mean_standard_deviation_uses_sample_deviation() -> None:
    mean, deviation = mean_standard_deviation(np.array((1.0, 2.0, 3.0)))
    assert mean == 2.0
    assert deviation == 1.0


def test_paired_effect_size() -> None:
    first = np.array((3.0, 5.0, 8.0, 10.0))
    second = np.array((1.0, 2.0, 4.0, 5.0))
    result = paired_standardized_mean_difference(first, second)
    assert result > 0.0


def test_holm_bonferroni_is_monotonic_in_sorted_order() -> None:
    values = np.array((0.001, 0.02, 0.04, 0.20))
    adjusted = holm_bonferroni(values)
    assert np.all(np.diff(adjusted) >= 0.0)
    assert np.all(adjusted >= values)


def test_paired_wilcoxon_returns_probability() -> None:
    first = np.arange(1.0, 11.0)
    second = first - 1.0
    result = paired_wilcoxon(first, second)
    assert 0.0 <= result <= 1.0


def test_interaction_ratio_matches_reported_gate_ssl_value() -> None:
    result = interaction_ratio(65.0, 52.5, 60.1, 58.9)
    assert result == pytest.approx(1.13636, rel=1e-4)


def test_main_result_lookup() -> None:
    result = main_result("Mamba ST-FM")
    assert result.iv2a_accuracy is not None
    assert result.iv2a_accuracy.mean == 65.0


def test_ablation_lookup() -> None:
    result = ablation("without_sasg")
    assert result.delta == -4.9


def test_result_table_sizes() -> None:
    assert len(MAIN_RESULTS) == 12
    assert len(ABLATIONS) == 8
    assert len(PROBES) == 4
    assert len(SCALES) == 5
    assert len(LABEL_EFFICIENCY) == 4
    assert len(EFFICIENCY) == 7
    assert len(MEMORY_CURVE) == 4


def test_attention_memory_exhausts_at_ten_thousand() -> None:
    assert MEMORY_CURVE[10000]["attention_mb"] is None
    assert MEMORY_CURVE[40000]["attention_mb"] is None


def test_mamba_memory_is_monotonic() -> None:
    values = [entry["mamba_stfm_mb"] for entry in MEMORY_CURVE.values()]
    assert values == sorted(values)


def test_full_ablation_is_highest_accuracy() -> None:
    full = ablation("full")
    assert all(full.accuracy.mean >= result.accuracy.mean for result in ABLATIONS)


def test_scale_finishes_at_full_result() -> None:
    assert SCALES[-1].subjects == 220
    assert SCALES[-1].accuracy.mean == 65.0


def test_pretraining_improves_all_label_fractions() -> None:
    assert all(result.pretrained > result.scratch for result in LABEL_EFFICIENCY)


def test_efficiency_entry_matches_memory_curve() -> None:
    model = next(result for result in EFFICIENCY if result.method == "Mamba ST-FM")
    assert model.memory_40k_mb == MEMORY_CURVE[40000]["mamba_stfm_mb"]
