"""
Tests for data loader, schema validation, cleaning, and stratified splitting.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.data_loader import (
    DatasetValidationError,
    clean_and_validate_dataset,
    load_dataset_csv,
    normalize_language_name,
    normalize_sentiment_label,
    split_dataset,
    validate_schema,
)


def test_validate_schema_valid():
    df = pd.DataFrame({
        "text": ["வணக்கம்"],
        "language": ["Tamil"],
        "label": ["positive"],
    })
    # Should not raise
    validate_schema(df)


def test_validate_schema_missing_column():
    df = pd.DataFrame({
        "text": ["வணக்கம்"],
        "language": ["Tamil"],
    })
    with pytest.raises(DatasetValidationError) as exc_info:
        validate_schema(df)
    assert "label" in str(exc_info.value)


def test_validate_schema_empty():
    df = pd.DataFrame(columns=["text", "language", "label"])
    with pytest.raises(DatasetValidationError):
        validate_schema(df)


def test_language_normalization():
    assert normalize_language_name("ta") == "Tamil"
    assert normalize_language_name("tamil") == "Tamil"
    assert normalize_language_name("hi") == "Hindi"
    assert normalize_language_name("hindi") == "Hindi"
    assert normalize_language_name("ml") == "Malayalam"
    assert normalize_language_name("malayalam") == "Malayalam"


def test_label_normalization():
    assert normalize_sentiment_label("Positive") == "positive"
    assert normalize_sentiment_label("POS") == "positive"
    assert normalize_sentiment_label("1") == "positive"
    assert normalize_sentiment_label("Negative") == "negative"
    assert normalize_sentiment_label("NEG") == "negative"
    assert normalize_sentiment_label("0") == "negative"
    assert normalize_sentiment_label("unknown") == ""


def test_clean_and_validate_dataset(tmp_path):
    raw_data = {
        "text": [
            "இந்த படம் அருமை",
            "   ",  # empty text
            "यह फिल्म बहुत अच्छी है",
            "यह फिल्म बहुत खराब है",
            "ഈ സിനിമ നല്ലതാണ്",
            "ഈ സിനിമ നല്ലതാണ്",  # duplicate row
            "இந்த படம் அருமை",  # duplicate with contradictory label
            None,  # null text
        ],
        "language": ["ta", "Tamil", "Hindi", "hi", "ml", "Malayalam", "Tamil", "Hindi"],
        "label": ["positive", "positive", "1", "0", "pos", "pos", "negative", "pos"],
    }
    df = pd.DataFrame(raw_data)

    cleaned_df, audit = clean_and_validate_dataset(
        df,
        supported_languages=["Tamil", "Hindi", "Malayalam"],
        supported_labels=["positive", "negative"],
    )

    # Empty and null rows should be removed
    assert len(cleaned_df) > 0
    # Conflicting texts should be removed
    assert "இந்த படம் அருமை" not in cleaned_df["text"].values
    assert audit["conflicting_duplicate_count"] >= 1
    # Check that labels are properly normalized
    assert set(cleaned_df["label"].unique()).issubset({"positive", "negative"})
    # Check that languages are properly normalized
    assert set(cleaned_df["language"].unique()).issubset({"Tamil", "Hindi", "Malayalam"})


def test_stratified_split():
    # Construct balanced synthetic dataset
    records = []
    for lang in ["Tamil", "Hindi", "Malayalam"]:
        for lbl in ["positive", "negative"]:
            for i in range(20):
                records.append({
                    "text": f"Review {lang} {lbl} {i}",
                    "language": lang,
                    "label": lbl,
                })

    df = pd.DataFrame(records)
    train_df, val_df, test_df = split_dataset(
        df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42
    )

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert len(train_df) > len(val_df)
    assert len(val_df) > 0
    assert len(test_df) > 0

    # Ensure all languages exist in all splits
    for lang in ["Tamil", "Hindi", "Malayalam"]:
        assert lang in train_df["language"].values
        assert lang in val_df["language"].values
        assert lang in test_df["language"].values
