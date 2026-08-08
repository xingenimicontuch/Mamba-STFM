from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    kappa: float
    macro_f1: float
    confusion: np.ndarray


def classification_metrics(target: np.ndarray, prediction: np.ndarray, labels: np.ndarray | None = None) -> ClassificationMetrics:
    return ClassificationMetrics(
        float(accuracy_score(target, prediction)),
        float(cohen_kappa_score(target, prediction)),
        float(f1_score(target, prediction, average="macro", zero_division=0)),
        confusion_matrix(target, prediction, labels=labels),
    )


def mean_standard_deviation(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1))


def paired_standardized_mean_difference(first: np.ndarray, second: np.ndarray) -> float:
    difference = first - second
    deviation = difference.std(ddof=1)
    return float(difference.mean() / deviation) if deviation > 0.0 else float("inf")


def holm_bonferroni(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues, dtype=np.float64)
    running = 0.0
    count = pvalues.size
    for rank, index in enumerate(order):
        value = min(1.0, float(pvalues[index]) * (count - rank))
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def paired_wilcoxon(first: np.ndarray, second: np.ndarray) -> float:
    return float(wilcoxon(first, second, alternative="two-sided", method="auto").pvalue)


def interaction_ratio(full: float, joint: float, single_a: float, single_b: float) -> float:
    denominator = (full - single_a) + (full - single_b)
    return (full - joint) / denominator if denominator else float("nan")
