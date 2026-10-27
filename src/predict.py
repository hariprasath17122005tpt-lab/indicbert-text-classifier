"""
Inference module for single and batch predictions using fine-tuned IndicBERT.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.config import ProjectConfig, load_config
from src.history import append_prediction_record
from src.model import check_fine_tuned_model_exists, get_model_and_tokenizer
from src.preprocessing import clean_indic_text, detect_indic_script
from src.utils import get_device, setup_logger

logger = setup_logger("Predictor")

# Global in-memory cache for singleton model & tokenizer loading
_LOADED_MODEL: Optional[PreTrainedModel] = None
_LOADED_TOKENIZER: Optional[PreTrainedTokenizerBase] = None
_LOADED_CONFIG: Optional[ProjectConfig] = None


class ModelNotAvailableError(Exception):
    """Raised when no trained fine-tuned model checkpoint is found."""
    pass


def get_cached_model(
    config: Optional[ProjectConfig] = None,
    force_reload: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, ProjectConfig]:
    """
    Get or load the fine-tuned IndicBERT model and tokenizer singleton.
    """
    global _LOADED_MODEL, _LOADED_TOKENIZER, _LOADED_CONFIG

    if config is None:
        config = load_config()

    if not force_reload and _LOADED_MODEL is not None and _LOADED_TOKENIZER is not None:
        return _LOADED_MODEL, _LOADED_TOKENIZER, _LOADED_CONFIG  # type: ignore

    model_dir = config.local_model_dir

    if not check_fine_tuned_model_exists(model_dir):
        # Check if HF repo ID was provided
        if config.hf_repo_id and config.hf_repo_id.strip():
            logger.info(f"Local model not found. Attempting to load from HF Hub repo: {config.hf_repo_id}")
            load_target = config.hf_repo_id.strip()
        else:
            raise ModelNotAvailableError(
                f"No fine-tuned model found in '{model_dir}'.\n"
                "Please train the model first by running:\n"
                "  python scripts/train_model.py\n"
                "Or smoke-test training via:\n"
                "  python scripts/train_model.py --smoke-test"
            )
    else:
        load_target = str(model_dir)

    device = get_device()
    logger.info(f"Loading fine-tuned model from '{load_target}' onto device: {device}")

    model, tokenizer = get_model_and_tokenizer(
        model_name_or_path=load_target,
        num_labels=config.num_labels,
        id2label=config.id2label,
        label2id=config.label2id,
        memory_mapped=True,
    )

    model.to(device)
    model.eval()

    _LOADED_MODEL = model
    _LOADED_TOKENIZER = tokenizer
    _LOADED_CONFIG = config

    return _LOADED_MODEL, _LOADED_TOKENIZER, _LOADED_CONFIG


def predict_sentiment(
    text: str,
    language: Optional[str] = None,
    config: Optional[ProjectConfig] = None,
    log_to_history: bool = True,
) -> Dict[str, Any]:
    """
    Perform sentiment inference on a single text.
    Returns predicted sentiment, confidence, and probability distribution.
    """
    cleaned = clean_indic_text(text)
    if not cleaned:
        raise ValueError("Input text is empty or contains only whitespace.")

    model, tokenizer, cfg = get_cached_model(config)
    device = get_device()

    detected_lang = detect_indic_script(cleaned)
    final_lang = language if language and language != "Auto-Detect" else (detected_lang or "Unknown")

    raw_inputs = tokenizer(
        cleaned,
        padding=True,
        truncation=True,
        max_length=cfg.max_length,
        return_tensors="pt",
    )
    if hasattr(raw_inputs, "to"):
        inputs = raw_inputs.to(device)
    else:
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in raw_inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()

    best_idx = int(probs.argmax())
    confidence = float(probs[best_idx])
    predicted_label = cfg.id2label[best_idx]

    probabilities = {
        cfg.id2label[i]: round(float(probs[i]), 4)
        for i in range(len(probs))
    }

    result = {
        "text": cleaned,
        "language": final_lang,
        "predicted_sentiment": predicted_label,
        "predicted_label": predicted_label,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
        "model_name": cfg.model_name_or_path,
    }

    if log_to_history:
        try:
            append_prediction_record(
                text=cleaned,
                language=final_lang,
                predicted_sentiment=predicted_label,
                confidence=confidence,
                model_identifier=f"IndicBERT ({cfg.model_name_or_path.split('/')[-1]})",
                history_file_path=cfg.history_file,
            )
        except Exception as e:
            logger.warning(f"Could not log prediction to history: {e}")

    return result


def predict_batch(
    df: pd.DataFrame,
    text_column: str = "text",
    language_column: Optional[str] = "language",
    batch_size: int = 16,
    config: Optional[ProjectConfig] = None,
    log_to_history: bool = False,
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Run sentiment prediction across an entire DataFrame in batches.
    Preserves original row order and returns DataFrame with predictions.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in uploaded DataFrame.")

    model, tokenizer, cfg = get_cached_model(config, force_reload=force_reload)
    device = get_device()

    result_df = df.copy()

    # Preprocess texts
    raw_texts = result_df[text_column].fillna("").astype(str).tolist()
    cleaned_texts = [clean_indic_text(t) for t in raw_texts]

    preds = []
    confs = []
    prob_pos = []
    prob_neg = []
    detected_languages = []

    pos_id = cfg.label2id.get("positive", 1)
    neg_id = cfg.label2id.get("negative", 0)

    for i in range(0, len(cleaned_texts), batch_size):
        batch_slice = cleaned_texts[i : i + batch_size]
        # Handle empty strings in slice safely
        safe_batch = [t if t else " " for t in batch_slice]

        raw_inputs = tokenizer(
            safe_batch,
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        if hasattr(raw_inputs, "to"):
            inputs = raw_inputs.to(device)
        else:
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in raw_inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        for j, p in enumerate(probs):
            original_idx = i + j
            if not cleaned_texts[original_idx]:
                # If text was empty
                preds.append("invalid/empty")
                confs.append(0.0)
                prob_pos.append(0.0)
                prob_neg.append(0.0)
                detected_languages.append("Unknown")
            else:
                best_idx = int(p.argmax())
                pred_label = cfg.id2label[best_idx]
                conf = float(p[best_idx])
                preds.append(pred_label)
                confs.append(round(conf, 4))
                prob_pos.append(round(float(p[pos_id]), 4))
                prob_neg.append(round(float(p[neg_id]), 4))

                # Language detection / assignment
                if language_column and language_column in result_df.columns:
                    given_lang = str(result_df[language_column].iloc[original_idx]).strip()
                    lang_val = given_lang if given_lang and given_lang.lower() != "nan" else (detect_indic_script(cleaned_texts[original_idx]) or "Unknown")
                else:
                    lang_val = detect_indic_script(cleaned_texts[original_idx]) or "Unknown"
                detected_languages.append(lang_val)

    result_df["predicted_sentiment"] = preds
    result_df["predicted_label"] = preds
    result_df["confidence"] = confs
    result_df["prob_positive"] = prob_pos
    result_df["prob_negative"] = prob_neg
    if language_column not in result_df.columns or language_column is None:
        result_df["detected_language"] = detected_languages

    if log_to_history:
        for idx, row in result_df.iterrows():
            if row["predicted_sentiment"] != "invalid/empty":
                try:
                    lang = row.get("language") or row.get("detected_language") or "Unknown"
                    append_prediction_record(
                        text=row[text_column],
                        language=str(lang),
                        predicted_sentiment=row["predicted_sentiment"],
                        confidence=row["confidence"],
                        model_identifier=f"IndicBERT-Batch ({cfg.model_name_or_path.split('/')[-1]})",
                        history_file_path=cfg.history_file,
                    )
                except Exception as e:
                    logger.warning(f"Error logging batch row {idx}: {e}")

    return result_df
