"""
Local CSV-based prediction history persistence module.
Stores inference results locally without any external database.
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.utils import ensure_dir, setup_logger

logger = setup_logger("PredictionHistory")

HISTORY_COLUMNS = [
    "timestamp",
    "text",
    "language",
    "predicted_sentiment",
    "confidence",
    "model_identifier",
]


def init_history_file(history_file_path: Path | str) -> Path:
    """
    Ensure the CSV prediction history file exists with correct columns.
    Creates parent directories and header if not present.
    """
    p = Path(history_file_path)
    ensure_dir(p.parent)

    if not p.exists() or p.stat().st_size == 0:
        empty_df = pd.DataFrame(columns=HISTORY_COLUMNS)
        empty_df.to_csv(p, index=False, encoding="utf-8")
        logger.info(f"Initialized new prediction history file at: {p}")

    return p


def append_prediction_record(
    text: str,
    language: str,
    predicted_sentiment: str,
    confidence: float,
    model_identifier: str = "IndicBERT-Sentiment",
    history_file_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Append a single prediction record to local CSV history with UTF-8 encoding.
    """
    from src.config import load_config
    if history_file_path is None:
        cfg = load_config()
        p = cfg.history_file
    else:
        p = Path(history_file_path)

    init_history_file(p)

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": str(text).strip(),
        "language": str(language).strip(),
        "predicted_sentiment": str(predicted_sentiment).strip().lower(),
        "confidence": round(float(confidence), 4),
        "model_identifier": str(model_identifier).strip(),
    }

    df_record = pd.DataFrame([record])
    # Append mode with header=False
    try:
        df_record.to_csv(p, mode="a", header=False, index=False, encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to append prediction to history file {p}: {e}")
        raise IOError(f"Could not persist prediction to history CSV: {e}") from e

    return record


def load_prediction_history(history_file_path: Optional[Path | str] = None) -> pd.DataFrame:
    """
    Load prediction history from CSV, sorting newest-first.
    Returns empty DataFrame with standard columns if file doesn't exist or is empty.
    """
    if history_file_path is None:
        from src.config import load_config
        cfg = load_config()
        p = cfg.history_file
    else:
        p = Path(history_file_path)

    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    try:
        df = pd.read_csv(p, encoding="utf-8")
        # Ensure all columns exist
        for col in HISTORY_COLUMNS:
            if col not in df.columns:
                df[col] = None

        df = df[HISTORY_COLUMNS]
        # Return newest records first (reversing CSV append order)
        if not df.empty:
            df = df.iloc[::-1].reset_index(drop=True)

        return df

    except Exception as e:
        logger.warning(f"Error reading history CSV from {p}: {e}. Returning empty DataFrame.")
        return pd.DataFrame(columns=HISTORY_COLUMNS)


def filter_history(
    df: pd.DataFrame,
    search_query: Optional[str] = None,
    language: Optional[str] = None,
    sentiment: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filter prediction history by text substring search, language, or sentiment.
    """
    if df.empty:
        return df.copy()

    filtered = df.copy()

    # Search filter
    if search_query and search_query.strip():
        q = search_query.strip().lower()
        filtered = filtered[filtered["text"].astype(str).str.lower().str.contains(q, na=False)]

    # Language filter
    if language and language != "All":
        filtered = filtered[filtered["language"].astype(str).str.lower() == language.lower()]

    # Sentiment filter
    if sentiment and sentiment != "All":
        filtered = filtered[filtered["predicted_sentiment"].astype(str).str.lower() == sentiment.lower()]

    return filtered.reset_index(drop=True)


def clear_prediction_history(history_file_path: Optional[Path | str] = None) -> bool:
    """
    Safely reset the prediction history file to an empty state with columns.
    """
    if history_file_path is None:
        from src.config import load_config
        cfg = load_config()
        p = cfg.history_file
    else:
        p = Path(history_file_path)

    try:
        empty_df = pd.DataFrame(columns=HISTORY_COLUMNS)
        empty_df.to_csv(p, index=False, encoding="utf-8")
        logger.info(f"Cleared prediction history at: {p}")
        return True
    except Exception as e:
        logger.error(f"Failed to clear prediction history at {p}: {e}")
        return False


def export_history_csv(df: pd.DataFrame) -> bytes:
    """
    Export DataFrame to UTF-8 encoded bytes for Streamlit file download.
    """
    return df.to_csv(index=False, encoding="utf-8").encode("utf-8")
