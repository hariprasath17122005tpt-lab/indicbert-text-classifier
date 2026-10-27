"""
Evaluation metrics computation and visualization module.
Calculates Accuracy, Precision, Recall, Macro F1, Per-Class metrics,
Confusion Matrix, and Per-Language Performance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils import ensure_dir, save_json


def compute_classification_metrics(
    y_true: List[int | str],
    y_pred: List[int | str],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    Handles binary and multiclass sentiment tasks gracefully.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Cannot compute metrics on empty ground truth or predictions.")

    # Harmonize types: if class_names (strings) is provided and inputs are integer IDs, map to string names
    if class_names is not None and len(y_true) > 0 and isinstance(y_true[0], (int, np.integer)):
        y_true = [class_names[int(i)] if 0 <= int(i) < len(class_names) else str(i) for i in y_true]
        y_pred = [class_names[int(i)] if 0 <= int(i) < len(class_names) else str(i) for i in y_pred]
        unique_labels = class_names
    elif class_names is not None:
        unique_labels = class_names
    else:
        unique_labels = sorted(list(set(y_true) | set(y_pred)))

    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class detailed dictionary
    clf_rep_dict = classification_report(
        y_true,
        y_pred,
        labels=unique_labels,
        target_names=[str(c) for c in unique_labels],
        output_dict=True,
        zero_division=0,
    )

    # Format text classification report
    clf_rep_text = classification_report(
        y_true,
        y_pred,
        labels=unique_labels,
        target_names=[str(c) for c in unique_labels],
        digits=4,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "sample_count": len(y_true),
        "class_names": [str(c) for c in unique_labels],
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": clf_rep_dict,
        "classification_report_text": clf_rep_text,
    }


def compute_per_language_metrics(
    df: pd.DataFrame,
    text_col: str = "text",
    lang_col: str = "language",
    true_label_col: str = "label",
    pred_label_col: str = "predicted_label",
) -> pd.DataFrame:
    """
    Compute classification performance broken down by language.
    Essential for evaluating low-resource multilingual fairness.
    """
    required = [lang_col, true_label_col, pred_label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns for per-language metrics: {missing}")

    records = []
    for lang, group in df.groupby(lang_col):
        y_true = group[true_label_col].tolist()
        y_pred = group[pred_label_col].tolist()

        if len(y_true) == 0:
            continue

        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

        records.append({
            "language": str(lang),
            "sample_count": len(group),
            "accuracy": round(acc, 4),
            "macro_precision": round(prec, 4),
            "macro_recall": round(rec, 4),
            "macro_f1": round(macro_f1, 4),
        })

    return pd.DataFrame(records)


def plot_confusion_matrix(
    cm: Union[List[List[int]], np.ndarray],
    class_names: List[str],
    output_path: Optional[Path | str] = None,
    title: str = "IndicBERT Sentiment Confusion Matrix",
) -> plt.Figure:
    """
    Render a polished Seaborn confusion matrix heatmap and optionally save to disk.
    """
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax.set_title(title, fontsize=14, weight="bold", pad=12)
    ax.set_ylabel("True Sentiment Label", fontsize=11, labelpad=8)
    ax.set_xlabel("Predicted Sentiment Label", fontsize=11, labelpad=8)
    plt.tight_layout()

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")

    return fig


def plot_low_resource_curve(
    fractions: List[float],
    f1_scores: List[float],
    sample_sizes: Optional[List[int]] = None,
    output_path: Optional[Path | str] = None,
    title: str = "Low-Resource Learning Efficiency: Training Data vs Macro F1",
) -> plt.Figure:
    """
    Plot training data fraction vs Test Macro F1 score to demonstrate
    transfer learning capability with low-resource Indic languages.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    percentages = [f * 100 for f in fractions]
    ax.plot(percentages, f1_scores, marker="o", color="#2563EB", linewidth=2.5, markersize=8)

    # Annotate points
    for idx, (p, f1) in enumerate(zip(percentages, f1_scores)):
        label = f"{f1:.3f}"
        if sample_sizes and idx < len(sample_sizes):
            label += f"\n(N={sample_sizes[idx]})"
        ax.annotate(
            label,
            (p, f1),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            weight="bold",
        )

    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Training Data Percentage (%)", fontsize=11, labelpad=8)
    ax.set_ylabel("Test Set Macro F1-Score", fontsize=11, labelpad=8)
    ax.set_ylim([max(0.0, min(f1_scores) - 0.1), min(1.05, max(f1_scores) + 0.1)])
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")

    return fig
