"""
Model training execution script for IndicBERT transfer learning.
Fine-tunes IndicBERT on multilingual Indic text (Tamil, Hindi, Malayalam).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.train import train_model
from src.utils import setup_logger

logger = setup_logger("TrainScript")


def main():
    parser = argparse.ArgumentParser(description="Train / fine-tune IndicBERT on Indic sentiment data.")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 1-epoch smoke test with small sample size.")
    parser.add_argument("--full-finetune", action="store_true", help="Unfreeze entire model for full fine-tuning (recommended for GPU).")
    parser.add_argument("--freeze-base", action="store_true", help="Freeze base transformer and train classification head (recommended for CPU).")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of training samples to use.")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory for model.")
    args = parser.parse_args()

    config = load_config()

    # Override config values if passed via CLI
    if args.full_finetune:
        config._raw.setdefault("training", {})["freeze_base_model"] = False
        config._raw.setdefault("training", {})["learning_rate"] = config._raw.get("training", {}).get("full_finetune_learning_rate", 2e-5)
    elif args.freeze_base:
        config._raw.setdefault("training", {})["freeze_base_model"] = True
        config._raw.setdefault("training", {})["learning_rate"] = config._raw.get("training", {}).get("head_learning_rate", 1e-3)

    if args.epochs:
        config._raw.setdefault("training", {})["num_epochs"] = args.epochs
    if args.batch_size:
        config._raw.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr:
        config._raw.setdefault("training", {})["learning_rate"] = args.lr

    print("\n" + "=" * 60)
    print("INDICBERT MULTILINGUAL TRANSFER LEARNING")
    print("=" * 60)
    print(f"Base Pretrained Model: {config.model_name_or_path}")
    print(f"Target Output Dir:     {args.output_dir or config.local_model_dir}")
    print(f"Smoke Test Mode:       {args.smoke_test}")
    print(f"Epochs:                {args.epochs or (config.smoke_test_epochs if args.smoke_test else config.num_epochs)}")
    print(f"Batch Size:            {args.batch_size or (config.smoke_test_batch_size if args.smoke_test else config.batch_size)}")
    print(f"Learning Rate:         {args.lr or config.learning_rate}")
    print("=" * 60 + "\n")

    result = train_model(
        config=config,
        smoke_test=args.smoke_test,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("TRAINING PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Best Validation Macro F1: {result['best_val_macro_f1']:.4f}")
    print(f"Model saved at:           {args.output_dir or config.local_model_dir}")
    print(f"Training History JSON:    {config.training_history_file}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
