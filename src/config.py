"""
Configuration management module.
Loads, validates, and provides easy access to settings from config.yaml.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

# Resolve the project root dynamically (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(Exception):
    """Raised when configuration is missing, invalid, or unreadable."""
    pass


class ProjectConfig:
    """Wrapper around the parsed YAML configuration."""

    def __init__(self, config_dict: Dict[str, Any], config_path: Path):
        self._raw = config_dict
        self.config_path = config_path

    # --- Model settings ---
    @property
    def model_name_or_path(self) -> str:
        return self._raw.get("model", {}).get(
            "pretrained_model_name_or_path", "ai4bharat/IndicBERTv2-MLM-only"
        )

    @property
    def local_model_dir(self) -> Path:
        rel = self._raw.get("model", {}).get("local_model_dir", "models/indicbert_sentiment")
        return PROJECT_ROOT / rel

    @property
    def hf_repo_id(self) -> str:
        return self._raw.get("model", {}).get("hf_repo_id", "")

    @property
    def num_labels(self) -> int:
        return int(self._raw.get("model", {}).get("num_labels", 2))

    @property
    def id2label(self) -> Dict[int, str]:
        raw_map = self._raw.get("model", {}).get(
            "id2label", {0: "negative", 1: "positive"}
        )
        return {int(k): str(v) for k, v in raw_map.items()}

    @property
    def label2id(self) -> Dict[str, int]:
        raw_map = self._raw.get("model", {}).get(
            "label2id", {"negative": 0, "positive": 1}
        )
        return {str(k): int(v) for k, v in raw_map.items()}

    # --- Data settings ---
    @property
    def supported_languages(self) -> List[str]:
        return list(
            self._raw.get("data", {}).get(
                "supported_languages", ["Tamil", "Hindi", "Malayalam"]
            )
        )

    @property
    def language_codes(self) -> Dict[str, str]:
        return dict(
            self._raw.get("data", {}).get(
                "language_codes", {"Tamil": "ta", "Hindi": "hi", "Malayalam": "ml"}
            )
        )

    @property
    def raw_dir(self) -> Path:
        return PROJECT_ROOT / self._raw.get("data", {}).get("raw_dir", "data/raw")

    @property
    def processed_dir(self) -> Path:
        return PROJECT_ROOT / self._raw.get("data", {}).get("processed_dir", "data/processed")

    @property
    def raw_file(self) -> Path:
        return PROJECT_ROOT / self._raw.get("data", {}).get(
            "raw_file", "data/raw/indic_sentiment_raw.csv"
        )

    @property
    def train_file(self) -> Path:
        return PROJECT_ROOT / self._raw.get("data", {}).get(
            "train_file", "data/processed/train.csv"
        )

    @property
    def val_file(self) -> Path:
        return PROJECT_ROOT / self._raw.get("data", {}).get(
            "val_file", "data/processed/val.csv"
        )

    @property
    def test_file(self) -> Path:
        return PROJECT_ROOT / self._raw.get("data", {}).get(
            "test_file", "data/processed/test.csv"
        )

    @property
    def dataset_metadata_file(self) -> Path:
        return PROJECT_ROOT / self._raw.get("data", {}).get(
            "dataset_metadata_file", "data/processed/dataset_summary.json"
        )

    @property
    def history_file(self) -> Path:
        return PROJECT_ROOT / self._raw.get("data", {}).get(
            "history_file", "data/prediction_history.csv"
        )

    # --- Training settings ---
    @property
    def max_length(self) -> int:
        return int(self._raw.get("training", {}).get("max_length", 128))

    @property
    def batch_size(self) -> int:
        return int(self._raw.get("training", {}).get("batch_size", 16))

    @property
    def learning_rate(self) -> float:
        return float(self._raw.get("training", {}).get("learning_rate", 2e-5))

    @property
    def num_epochs(self) -> int:
        return int(self._raw.get("training", {}).get("num_epochs", 3))

    @property
    def weight_decay(self) -> float:
        return float(self._raw.get("training", {}).get("weight_decay", 0.01))

    @property
    def warmup_ratio(self) -> float:
        return float(self._raw.get("training", {}).get("warmup_ratio", 0.1))

    @property
    def freeze_base_model(self) -> bool:
        return bool(self._raw.get("training", {}).get("freeze_base_model", True))

    @property
    def trainable_layers(self) -> str:
        return str(self._raw.get("training", {}).get("trainable_layers", "head"))

    @property
    def early_stopping_patience(self) -> int:
        return int(self._raw.get("training", {}).get("early_stopping_patience", 2))

    @property
    def seed(self) -> int:
        return int(self._raw.get("training", {}).get("seed", 42))

    # --- Smoke test settings ---
    @property
    def smoke_test_max_samples(self) -> int:
        return int(self._raw.get("smoke_test", {}).get("max_samples", 30))

    @property
    def smoke_test_batch_size(self) -> int:
        return int(self._raw.get("smoke_test", {}).get("batch_size", 4))

    @property
    def smoke_test_epochs(self) -> int:
        return int(self._raw.get("smoke_test", {}).get("num_epochs", 1))

    # --- Low-resource experiment ---
    @property
    def experiment_fractions(self) -> List[float]:
        return list(
            self._raw.get("low_resource_experiment", {}).get(
                "fractions", [0.10, 0.25, 0.50, 1.00]
            )
        )

    @property
    def quick_experiment_fractions(self) -> List[float]:
        return list(
            self._raw.get("low_resource_experiment", {}).get(
                "quick_fractions", [0.25, 1.00]
            )
        )

    @property
    def experiment_results_csv(self) -> Path:
        return PROJECT_ROOT / self._raw.get("low_resource_experiment", {}).get(
            "results_csv", "results/low_resource_comparison.csv"
        )

    @property
    def experiment_plot_path(self) -> Path:
        return PROJECT_ROOT / self._raw.get("low_resource_experiment", {}).get(
            "plot_path", "results/low_resource_comparison.png"
        )

    # --- Results files ---
    @property
    def results_dir(self) -> Path:
        return PROJECT_ROOT / self._raw.get("results", {}).get("dir", "results")

    @property
    def metrics_file(self) -> Path:
        return PROJECT_ROOT / self._raw.get("results", {}).get(
            "metrics_file", "results/metrics.json"
        )

    @property
    def report_file(self) -> Path:
        return PROJECT_ROOT / self._raw.get("results", {}).get(
            "report_file", "results/classification_report.txt"
        )

    @property
    def confusion_matrix_plot(self) -> Path:
        return PROJECT_ROOT / self._raw.get("results", {}).get(
            "confusion_matrix_plot", "results/confusion_matrix.png"
        )

    @property
    def per_language_metrics_file(self) -> Path:
        return PROJECT_ROOT / self._raw.get("results", {}).get(
            "per_language_metrics", "results/per_language_metrics.csv"
        )

    @property
    def training_history_file(self) -> Path:
        return PROJECT_ROOT / self._raw.get("results", {}).get(
            "training_history_file", "results/training_history.json"
        )

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._raw)


def load_config(config_path: Optional[str | Path] = None) -> ProjectConfig:
    """
    Load configuration from yaml file.
    If path is not provided, defaults to config.yaml in project root.
    """
    if config_path is None:
        target_path = PROJECT_ROOT / "config.yaml"
    else:
        target_path = Path(config_path)
        if not target_path.is_absolute():
            target_path = PROJECT_ROOT / target_path

    if not target_path.exists():
        raise ConfigError(
            f"Configuration file not found at: {target_path}.\n"
            "Ensure 'config.yaml' exists in the project root."
        )

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            if not isinstance(cfg, dict):
                raise ConfigError(f"Config file {target_path} is empty or not a valid dictionary.")
            return ProjectConfig(cfg, target_path)
    except yaml.YAMLError as e:
        raise ConfigError(f"Error parsing YAML in {target_path}: {e}") from e
