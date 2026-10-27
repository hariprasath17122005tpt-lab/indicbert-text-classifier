"""
Dataset loading, schema validation, cleaning, and stratified splitting module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import clean_indic_text, get_text_statistics
from src.utils import ensure_dir, save_json, setup_logger

logger = setup_logger("DataLoader")

REQUIRED_COLUMNS = ["text", "language", "label"]

# Normalization mapping for language names
LANGUAGE_NORMALIZATION = {
    "tamil": "Tamil",
    "ta": "Tamil",
    "தமிழ்": "Tamil",
    "hindi": "Hindi",
    "hi": "Hindi",
    "हिन्दी": "Hindi",
    "हिंदी": "Hindi",
    "malayalam": "Malayalam",
    "ml": "Malayalam",
    "മലയാളം": "Malayalam",
    "telugu": "Telugu",
    "te": "Telugu",
    "kannada": "Kannada",
    "kn": "Kannada",
    "bengali": "Bengali",
    "bn": "Bengali",
}

# Normalization mapping for sentiment labels
LABEL_NORMALIZATION = {
    "positive": "positive",
    "pos": "positive",
    "1": "positive",
    "1.0": "positive",
    "negative": "negative",
    "neg": "negative",
    "0": "negative",
    "0.0": "negative",
    "-1": "negative",
}


class DatasetValidationError(Exception):
    """Raised when dataset fails schema, column, or integrity validation."""
    pass


def load_dataset_csv(file_path: Path | str) -> pd.DataFrame:
    """
    Load a CSV file with UTF-8 encoding.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset CSV not found at: {p.resolve()}")

    try:
        df = pd.read_csv(p, encoding="utf-8")
        return df
    except UnicodeDecodeError:
        # Fallback to utf-8-sig if BOM exists
        df = pd.read_csv(p, encoding="utf-8-sig")
        return df


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate that the DataFrame contains all required columns: text, language, label.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DatasetValidationError(
            f"Dataset is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}. Required: {REQUIRED_COLUMNS}"
        )
    if len(df) == 0:
        raise DatasetValidationError("Dataset CSV is empty (0 rows).")


def normalize_language_name(lang: Any) -> str:
    """
    Normalize language names to capitalized English names (e.g. 'ta' -> 'Tamil').
    """
    if pd.isna(lang):
        return ""
    key = str(lang).strip().lower()
    return LANGUAGE_NORMALIZATION.get(key, str(lang).strip().capitalize())


def normalize_sentiment_label(label: Any) -> str:
    """
    Normalize sentiment labels to 'positive' or 'negative'.
    """
    if pd.isna(label):
        return ""
    key = str(label).strip().lower()
    return LABEL_NORMALIZATION.get(key, "")


def clean_and_validate_dataset(
    df: pd.DataFrame,
    supported_languages: Optional[List[str]] = None,
    supported_labels: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Comprehensive dataset cleaning and auditing:
    1. Validates schema.
    2. Drops nulls and empty texts.
    3. Cleans native text while preserving Indic scripts.
    4. Normalizes language and label columns.
    5. Filters to supported languages and labels.
    6. Identifies duplicates and detects contradictory duplicate labels.
    7. Returns the cleaned DataFrame and an audit report dictionary.
    """
    validate_schema(df)
    initial_count = len(df)
    audit: Dict[str, Any] = {"initial_sample_count": initial_count}

    df = df.copy()

    # Drop rows where text or label or language is null
    null_counts = df[REQUIRED_COLUMNS].isnull().sum().to_dict()
    audit["missing_values"] = {k: int(v) for k, v in null_counts.items()}
    df = df.dropna(subset=REQUIRED_COLUMNS)

    # Clean text
    df["text"] = df["text"].apply(clean_indic_text)
    # Filter out empty texts
    empty_mask = df["text"] == ""
    audit["empty_text_dropped"] = int(empty_mask.sum())
    df = df[~empty_mask]

    # Normalize language and label
    df["language"] = df["language"].apply(normalize_language_name)
    df["label"] = df["label"].apply(normalize_sentiment_label)

    # Filter invalid labels
    invalid_label_mask = df["label"] == ""
    audit["invalid_label_dropped"] = int(invalid_label_mask.sum())
    df = df[~invalid_label_mask]

    # Filter supported languages if specified
    if supported_languages:
        sup_lang_set = set(supported_languages)
        lang_mask = df["language"].isin(sup_lang_set)
        audit["unsupported_languages_dropped"] = int((~lang_mask).sum())
        df = df[lang_mask]

    # Filter supported labels if specified
    if supported_labels:
        sup_lbl_set = set(supported_labels)
        lbl_mask = df["label"].isin(sup_lbl_set)
        audit["unsupported_labels_dropped"] = int((~lbl_mask).sum())
        df = df[lbl_mask]

    # Check for contradictory duplicates (same text with conflicting labels)
    text_label_grouped = df.groupby("text")["label"].nunique()
    conflicting_texts = text_label_grouped[text_label_grouped > 1].index.tolist()
    audit["conflicting_duplicate_count"] = len(conflicting_texts)
    if conflicting_texts:
        logger.warning(
            f"Found {len(conflicting_texts)} text entries with conflicting labels! Removing them to prevent data corruption."
        )
        df = df[~df["text"].isin(conflicting_texts)]

    # Check for identical duplicate rows
    duplicate_mask = df.duplicated(subset=["text", "language", "label"])
    audit["exact_duplicates_removed"] = int(duplicate_mask.sum())
    df = df.drop_duplicates(subset=["text", "language", "label"]).reset_index(drop=True)

    audit["final_clean_sample_count"] = len(df)
    audit["language_distribution"] = df["language"].value_counts().to_dict()
    audit["label_distribution"] = df["label"].value_counts().to_dict()

    # Language x Label cross-tabulation
    cross_tab = pd.crosstab(df["language"], df["label"]).to_dict()
    audit["language_label_crosstab"] = cross_tab

    # Text length stats
    audit["text_statistics"] = get_text_statistics(df["text"].tolist())

    return df, audit


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified train / val / test splitting.
    Stratifies on composite 'language_label' key to preserve both class balance
    and language distribution across splits, while preventing leakage.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        raise ValueError(f"Split ratios must sum to 1.0, got: {total_ratio}")

    df = df.copy().reset_index(drop=True)
    df["strata"] = df["language"].astype(str) + "_" + df["label"].astype(str)

    # Check minimum samples per stratum for stratification
    strata_counts = df["strata"].value_counts()
    min_stratum_size = strata_counts.min()

    use_stratify = True
    if min_stratum_size < 3:
        logger.warning(
            f"Smallest stratum has {min_stratum_size} samples (< 3). Falling back to non-stratified random split."
        )
        use_stratify = False

    test_size = test_ratio
    stratify_col = df["strata"] if use_stratify else None

    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify_col,
    )

    # Second split: separate train and validation from remaining
    val_relative_ratio = val_ratio / (train_ratio + val_ratio)
    stratify_train_val = train_val_df["strata"] if use_stratify else None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_ratio,
        random_state=random_seed,
        stratify=stratify_train_val,
    )

    # Clean up auxiliary strata column
    train_df = train_df.drop(columns=["strata"]).reset_index(drop=True)
    val_df = val_df.drop(columns=["strata"]).reset_index(drop=True)
    test_df = test_df.drop(columns=["strata"]).reset_index(drop=True)

    return train_df, val_df, test_df


def save_processed_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path | str,
    audit_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """
    Save train.csv, val.csv, test.csv and dataset_summary.json into output_dir.
    """
    out = ensure_dir(output_dir)
    train_path = out / "train.csv"
    val_path = out / "val.csv"
    test_path = out / "test.csv"
    summary_path = out / "dataset_summary.json"

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    meta: Dict[str, Any] = {
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "total_samples": len(train_df) + len(val_df) + len(test_df),
        "splits": {
            "train": {
                "languages": train_df["language"].value_counts().to_dict(),
                "labels": train_df["label"].value_counts().to_dict(),
            },
            "val": {
                "languages": val_df["language"].value_counts().to_dict(),
                "labels": val_df["label"].value_counts().to_dict(),
            },
            "test": {
                "languages": test_df["language"].value_counts().to_dict(),
                "labels": test_df["label"].value_counts().to_dict(),
            },
        },
    }

    if audit_summary:
        meta["audit_summary"] = audit_summary

    save_json(meta, summary_path)

    logger.info(
        f"Saved splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)} to {out}"
    )

    return {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "summary": summary_path,
    }
