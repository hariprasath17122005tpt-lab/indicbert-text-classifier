"""
Low-Resource Indic NLP Transfer Learning Experiment.
Compares classification performance across different proportions of training data
(e.g., 10%, 25%, 50%, 100% or quick mode: 25%, 100%) on an identical held-out test set.
Saves results to results/low_resource_comparison.csv and results/low_resource_comparison.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.evaluate import evaluate_test_set
from src.metrics import plot_low_resource_curve
from src.train import train_model
from src.utils import ensure_dir, set_seed, setup_logger

logger = setup_logger("LowResourceExperiment")


def stratified_subset(df: pd.DataFrame, fraction: float, seed: int = 42) -> pd.DataFrame:
    """
    Extract a stratified subset preserving language and sentiment balance.
    """
    if fraction >= 1.0:
        return df.copy().reset_index(drop=True)

    df_copy = df.copy().reset_index(drop=True)
    df_copy["strata"] = df_copy["language"].astype(str) + "_" + df_copy["label"].astype(str)

    # Check minimum stratum size
    counts = df_copy["strata"].value_counts()
    if counts.min() < 2:
        # Fallback to simple random sample if strata are too small
        return df_copy.sample(frac=fraction, random_state=seed).drop(columns=["strata"]).reset_index(drop=True)

    sub_df, _ = train_test_split(
        df_copy,
        train_size=fraction,
        random_state=seed,
        stratify=df_copy["strata"],
    )
    return sub_df.drop(columns=["strata"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Run low-resource Indic learning experiment.")
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="Quick mode (25%, 100% with fewer epochs) vs Full mode (10%, 25%, 50%, 100%).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Epochs per fraction (default: 1 in quick, 2 in full).")
    args = parser.parse_args()

    config = load_config()
    set_seed(config.seed)

    # Check that datasets exist
    if not config.train_file.exists() or not config.test_file.exists():
        logger.error("Dataset splits not found! Please run `python scripts/prepare_dataset.py` first.")
        sys.exit(1)

    train_df = pd.read_csv(config.train_file, encoding="utf-8")
    val_df = pd.read_csv(config.val_file, encoding="utf-8")
    test_df = pd.read_csv(config.test_file, encoding="utf-8")

    if args.mode == "quick":
        fractions = config.quick_experiment_fractions
        epochs_per_fraction = args.epochs if args.epochs else 1
        batch_size = 8
        logger.info(f"Running QUICK mode experiment on fractions: {fractions} with {epochs_per_fraction} epoch(s)")
    else:
        fractions = config.experiment_fractions
        epochs_per_fraction = args.epochs if args.epochs else 2
        batch_size = config.batch_size
        logger.info(f"Running FULL mode experiment on fractions: {fractions} with {epochs_per_fraction} epoch(s)")

    results_dir = ensure_dir(config.results_dir)
    experiment_results = []

    print("\n" + "=" * 65)
    print("LOW-RESOURCE INDIC NLP TRANSFER LEARNING EXPERIMENT")
    print("=" * 65)
    print(f"Mode:              {args.mode.upper()}")
    print(f"Fractions to test: {[f'{int(f*100)}%' for f in fractions]}")
    print(f"Total Train Pool:  {len(train_df)} samples")
    print(f"Validation Set:    {len(val_df)} samples")
    print(f"Held-out Test Set: {len(test_df)} samples (constant across all runs)")
    print("=" * 65 + "\n")

    for frac in fractions:
        pct = int(frac * 100)
        subset_df = stratified_subset(train_df, frac, seed=config.seed)
        logger.info(f"--> Training with {pct}% of training data ({len(subset_df)} samples)...")

        # Language distribution in this subset
        lang_dist = subset_df["language"].value_counts().to_dict()
        lang_dist_str = ", ".join([f"{k}: {v}" for k, v in lang_dist.items()])
        logger.info(f"    Language distribution: {lang_dist_str}")

        # Temporary model output dir for this fraction
        frac_model_dir = config.local_model_dir.parent / f"indicbert_fraction_{pct}pct"

        # Override config epochs & batch_size for experiment
        exp_config = load_config()
        exp_config._raw.setdefault("training", {})["num_epochs"] = epochs_per_fraction
        exp_config._raw.setdefault("training", {})["batch_size"] = batch_size

        start_time = time.time()
        # Use a representative validation slice for faster epoch evaluation on CPU
        exp_val_df = val_df.head(100) if len(val_df) > 100 else val_df

        train_res = train_model(
            config=exp_config,
            smoke_test=False,
            custom_train_df=subset_df,
            custom_val_df=exp_val_df,
            output_dir=frac_model_dir,
            save_checkpoint=False,
        )
        duration = round(time.time() - start_time, 2)

        # Evaluate the trained model on the identical held-out test set (all 519 samples)
        eval_res = evaluate_test_set(
            model=train_res["model"],
            tokenizer=train_res["tokenizer"],
            test_df=test_df,
            config=exp_config,
            save_artifacts=False,
        )

        test_metrics = eval_res["metrics"]
        record = {
            "fraction": frac,
            "training_percentage": f"{pct}%",
            "train_samples": len(subset_df),
            "samples_per_language": str(lang_dist),
            "training_duration_sec": duration,
            "best_val_macro_f1": train_res["best_val_macro_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_precision": test_metrics["macro_precision"],
            "test_macro_recall": test_metrics["macro_recall"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
        }
        experiment_results.append(record)

        # Free memory before next fraction
        del train_res
        del eval_res
        import gc
        gc.collect()

        logger.info(
            f"Result for {pct}% data: Test Acc = {test_metrics['accuracy']:.4f}, "
            f"Test Macro F1 = {test_metrics['macro_f1']:.4f} (took {duration:.1f}s)"
        )

    # Save results to CSV
    results_df = pd.DataFrame(experiment_results)
    results_csv_path = config.experiment_results_csv
    results_df.to_csv(results_csv_path, index=False, encoding="utf-8")
    logger.info(f"Saved experiment comparison table to: {results_csv_path}")

    # Generate comparison plot
    plot_low_resource_curve(
        fractions=[r["fraction"] for r in experiment_results],
        f1_scores=[r["test_macro_f1"] for r in experiment_results],
        sample_sizes=[r["train_samples"] for r in experiment_results],
        output_path=config.experiment_plot_path,
        title=f"IndicBERT Transfer Learning vs Data Quantity ({args.mode.capitalize()} Experiment)",
    )
    logger.info(f"Saved experiment comparison plot to: {config.experiment_plot_path}")

    print("\n" + "=" * 65)
    print("LOW-RESOURCE EXPERIMENT SUMMARY")
    print("=" * 65)
    print(results_df[["training_percentage", "train_samples", "best_val_macro_f1", "test_accuracy", "test_macro_f1"]].to_string(index=False))
    print("\nFiles saved:")
    print(f"  - CSV table: {results_csv_path}")
    print(f"  - Curve plot: {config.experiment_plot_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
