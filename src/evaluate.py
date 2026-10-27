"""
Model evaluation module.
Evaluates fine-tuned IndicBERT on held-out test set, generating metrics.json,
classification_report.txt, confusion_matrix.png, and per_language_metrics.csv.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.config import ProjectConfig, load_config
from src.metrics import (
    compute_classification_metrics,
    compute_per_language_metrics,
    plot_confusion_matrix,
)
from src.model import get_model_and_tokenizer
from src.utils import ensure_dir, get_device, save_json, setup_logger

logger = setup_logger("Evaluator")


def evaluate_test_set(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_df: pd.DataFrame,
    config: Optional[ProjectConfig] = None,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Run complete evaluation of the trained model on test_df.
    Computes global metrics, classification report, confusion matrix plot,
    and per-language breakdown.
    """
    if config is None:
        config = load_config()

    if device is None:
        device = get_device()

    model.to(device)
    model.eval()

    label2id = config.label2id
    id2label = config.id2label
    max_length = config.max_length

    texts = test_df["text"].tolist()
    true_labels = [label2id[str(lbl).lower()] for lbl in test_df["label"].tolist()]
    languages = test_df["language"].tolist()

    all_pred_ids = []
    all_pred_labels = []
    all_confidences = []
    all_probs = []

    logger.info(f"Evaluating model on {len(texts)} test samples...")

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            pred_ids = probs.argmax(axis=-1).tolist()
            confidences = probs.max(axis=-1).tolist()

            all_pred_ids.extend(pred_ids)
            all_confidences.extend(confidences)
            all_probs.extend(probs.tolist())
            all_pred_labels.extend([id2label[pid] for pid in pred_ids])

    # 1. Global classification metrics
    class_names = [id2label[i] for i in sorted(id2label.keys())]
    metrics = compute_classification_metrics(
        y_true=true_labels,
        y_pred=all_pred_ids,
        class_names=class_names,
    )

    # 2. Per-language breakdown
    test_eval_df = test_df.copy()
    test_eval_df["predicted_label"] = all_pred_labels
    test_eval_df["confidence"] = all_confidences

    per_lang_df = compute_per_language_metrics(
        test_eval_df,
        text_col="text",
        lang_col="language",
        true_label_col="label",
        pred_label_col="predicted_label",
    )

    # 3. Save all evaluation artifacts
    results_dir = ensure_dir(config.results_dir)

    metrics_to_save = {
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "sample_count": metrics["sample_count"],
        "class_names": metrics["class_names"],
        "confusion_matrix": metrics["confusion_matrix"],
        "per_class_metrics": metrics["per_class_metrics"],
    }

    if save_artifacts:
        # A. Save metrics.json
        save_json(metrics_to_save, config.metrics_file)

        # B. Save classification_report.txt
        with open(config.report_file, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("INDICBERT SENTIMENT CLASSIFICATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(metrics["classification_report_text"] + "\n\n")
            f.write("Overall Metrics Summary:\n")
            f.write(f"- Test Accuracy:        {metrics['accuracy'] * 100:.2f}%\n")
            f.write(f"- Test Macro F1-Score:  {metrics['macro_f1']:.4f}\n")
            f.write(f"- Test Macro Precision: {metrics['macro_precision']:.4f}\n")
            f.write(f"- Test Macro Recall:    {metrics['macro_recall']:.4f}\n")
            f.write(f"- Total Test Samples:   {metrics['sample_count']}\n")

        # C. Save confusion matrix plot
        plot_confusion_matrix(
            cm=metrics["confusion_matrix"],
            class_names=class_names,
            output_path=config.confusion_matrix_plot,
            title="IndicBERT Confusion Matrix (Held-Out Test Set)",
        )

        # D. Save per-language CSV
        per_lang_df.to_csv(config.per_language_metrics_file, index=False, encoding="utf-8")

        logger.info(
            f"Evaluation complete! Test Accuracy: {metrics['accuracy']:.4f}, "
            f"Macro F1: {metrics['macro_f1']:.4f}. Artifacts saved in {results_dir}"
        )
    else:
        logger.info(
            f"Evaluation complete! Test Accuracy: {metrics['accuracy']:.4f}, "
            f"Macro F1: {metrics['macro_f1']:.4f} (in-memory mode, artifacts not overwritten)."
        )

    return {
        "metrics": metrics_to_save,
        "per_language_df": per_lang_df,
        "classification_report_text": metrics["classification_report_text"],
        "evaluation_df": test_eval_df,
    }


def run_evaluation_from_saved_model(
    config: Optional[ProjectConfig] = None,
    model_dir: Optional[Path | str] = None,
    test_csv: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Load fine-tuned model from disk and evaluate on test CSV.
    """
    if config is None:
        config = load_config()

    m_dir = Path(model_dir) if model_dir else config.local_model_dir
    t_csv = Path(test_csv) if test_csv else config.test_file

    if not m_dir.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {m_dir.resolve()}.\n"
            "Please run `python scripts/train_model.py` before evaluation."
        )

    if not t_csv.exists():
        raise FileNotFoundError(
            f"Test dataset not found at: {t_csv.resolve()}.\n"
            "Please run `python scripts/prepare_dataset.py` first."
        )

    test_df = pd.read_csv(t_csv, encoding="utf-8")
    model, tokenizer = get_model_and_tokenizer(
        model_name_or_path=m_dir,
        num_labels=config.num_labels,
        id2label=config.id2label,
        label2id=config.label2id,
        memory_mapped=True,
    )

    return evaluate_test_set(
        model=model,
        tokenizer=tokenizer,
        test_df=test_df,
        config=config,
    )
