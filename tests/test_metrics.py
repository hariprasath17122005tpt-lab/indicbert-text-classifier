"""
Tests for classification metrics and per-language evaluation functions.
"""

import pandas as pd
import pytest

from src.metrics import (
    compute_classification_metrics,
    compute_per_language_metrics,
    plot_confusion_matrix,
    plot_low_resource_curve,
)


def test_compute_classification_metrics_perfect():
    y_true = ["positive", "negative", "positive", "negative"]
    y_pred = ["positive", "negative", "positive", "negative"]

    metrics = compute_classification_metrics(y_true, y_pred, class_names=["negative", "positive"])
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_compute_classification_metrics_integer_labels():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 0]

    metrics = compute_classification_metrics(y_true, y_pred, class_names=["negative", "positive"])
    assert metrics["accuracy"] == 0.75
    assert 0.0 < metrics["macro_f1"] < 1.0


def test_compute_classification_metrics_empty_raises():
    with pytest.raises(ValueError):
        compute_classification_metrics([], [])


def test_compute_per_language_metrics():
    df = pd.DataFrame({
        "language": ["Tamil", "Tamil", "Hindi", "Hindi", "Malayalam", "Malayalam"],
        "label": ["positive", "negative", "positive", "negative", "positive", "negative"],
        "predicted_label": ["positive", "negative", "positive", "positive", "negative", "negative"],
    })

    lang_metrics = compute_per_language_metrics(df)
    assert len(lang_metrics) == 3
    assert set(lang_metrics["language"].tolist()) == {"Tamil", "Hindi", "Malayalam"}

    tamil_row = lang_metrics[lang_metrics["language"] == "Tamil"].iloc[0]
    assert tamil_row["accuracy"] == 1.0


def test_plot_confusion_matrix(tmp_path):
    cm = [[10, 2], [3, 15]]
    out = tmp_path / "test_cm.png"
    fig = plot_confusion_matrix(cm, ["negative", "positive"], output_path=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_low_resource_curve(tmp_path):
    fractions = [0.10, 0.25, 0.50, 1.00]
    f1_scores = [0.65, 0.74, 0.81, 0.86]
    out = tmp_path / "test_curve.png"
    fig = plot_low_resource_curve(fractions, f1_scores, output_path=out)
    assert out.exists()
    assert out.stat().st_size > 0
