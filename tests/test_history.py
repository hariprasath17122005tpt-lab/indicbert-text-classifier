"""
Tests for local CSV prediction history persistence and filtering.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.history import (
    append_prediction_record,
    clear_prediction_history,
    export_history_csv,
    filter_history,
    init_history_file,
    load_prediction_history,
)


def test_init_history_file(tmp_path):
    history_file = tmp_path / "test_history.csv"
    p = init_history_file(history_file)
    assert p.exists()
    df = pd.read_csv(p)
    assert list(df.columns) == [
        "timestamp",
        "text",
        "language",
        "predicted_sentiment",
        "confidence",
        "model_identifier",
    ]


def test_append_and_load_record_indic_unicode(tmp_path):
    history_file = tmp_path / "test_history.csv"
    record1 = append_prediction_record(
        text="இந்த படம் மிகவும் அருமை",
        language="Tamil",
        predicted_sentiment="positive",
        confidence=0.985,
        model_identifier="IndicBERT-Test",
        history_file_path=history_file,
    )

    record2 = append_prediction_record(
        text="यह फिल्म बहुत खराब है",
        language="Hindi",
        predicted_sentiment="negative",
        confidence=0.942,
        model_identifier="IndicBERT-Test",
        history_file_path=history_file,
    )

    df = load_prediction_history(history_file)
    assert len(df) == 2
    # Newest first
    assert df.iloc[0]["language"] == "Hindi"
    assert df.iloc[1]["language"] == "Tamil"

    # Verify UTF-8 Tamil and Hindi characters preserved
    assert "இந்த" in df.iloc[1]["text"]
    assert "फिल्म" in df.iloc[0]["text"]


def test_filter_history(tmp_path):
    history_file = tmp_path / "test_history.csv"
    append_prediction_record(
        text="இந்த படம் அருமை",
        language="Tamil",
        predicted_sentiment="positive",
        confidence=0.95,
        history_file_path=history_file,
    )
    append_prediction_record(
        text="ഈ സിനിമ മോശമാണ്",
        language="Malayalam",
        predicted_sentiment="negative",
        confidence=0.91,
        history_file_path=history_file,
    )

    df = load_prediction_history(history_file)

    # Filter by language
    tamil_only = filter_history(df, language="Tamil")
    assert len(tamil_only) == 1
    assert tamil_only.iloc[0]["language"] == "Tamil"

    # Filter by sentiment
    neg_only = filter_history(df, sentiment="negative")
    assert len(neg_only) == 1
    assert neg_only.iloc[0]["predicted_sentiment"] == "negative"

    # Filter by search
    searched = filter_history(df, search_query="സിനിമ")
    assert len(searched) == 1
    assert searched.iloc[0]["language"] == "Malayalam"


def test_clear_history(tmp_path):
    history_file = tmp_path / "test_history.csv"
    append_prediction_record(
        text="परीक्षण",
        language="Hindi",
        predicted_sentiment="positive",
        confidence=0.89,
        history_file_path=history_file,
    )

    df_before = load_prediction_history(history_file)
    assert len(df_before) == 1

    success = clear_prediction_history(history_file)
    assert success is True

    df_after = load_prediction_history(history_file)
    assert len(df_after) == 0


def test_export_history_csv(tmp_path):
    df = pd.DataFrame([{"text": "தமிழ்", "language": "Tamil", "predicted_sentiment": "positive"}])
    csv_bytes = export_history_csv(df)
    assert isinstance(csv_bytes, bytes)
    assert "தமிழ்".encode("utf-8") in csv_bytes
