"""
Model evaluation script.
Evaluates the trained IndicBERT sentiment classifier on the held-out test set.
Generates metrics.json, classification_report.txt, confusion_matrix.png, and per_language_metrics.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.evaluate import run_evaluation_from_saved_model
from src.utils import setup_logger

logger = setup_logger("EvaluateScript")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned IndicBERT on test set.")
    parser.add_argument("--model-dir", type=str, default=None, help="Path to trained model directory.")
    parser.add_argument("--test-csv", type=str, default=None, help="Path to test CSV dataset.")
    args = parser.parse_args()

    config = load_config()

    print("\n" + "=" * 60)
    print("INDICBERT MODEL EVALUATION ON HELD-OUT TEST SET")
    print("=" * 60)
    print(f"Model Directory: {args.model_dir or config.local_model_dir}")
    print(f"Test Set:        {args.test_csv or config.test_file}")
    print("=" * 60 + "\n")

    result = run_evaluation_from_saved_model(
        config=config,
        model_dir=args.model_dir,
        test_csv=args.test_csv,
    )

    metrics = result["metrics"]
    per_lang_df = result["per_language_df"]

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Test Accuracy:         {metrics['accuracy'] * 100:.2f}%")
    print(f"Test Macro F1-Score:   {metrics['macro_f1']:.4f}")
    print(f"Test Macro Precision:  {metrics['macro_precision']:.4f}")
    print(f"Test Macro Recall:     {metrics['macro_recall']:.4f}")
    print(f"Total Test Samples:    {metrics['sample_count']}")

    print("\n--- Per-Language Performance Breakdown ---")
    print(per_lang_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("Generated Artifacts:")
    print(f"  - Metrics JSON:          {config.metrics_file}")
    print(f"  - Classification Report: {config.report_file}")
    print(f"  - Confusion Matrix Plot: {config.confusion_matrix_plot}")
    print(f"  - Per-Language Metrics:  {config.per_language_metrics_file}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
