"""
Tests for prediction logic, inference formatting, and missing model handling.
"""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
import torch

from src.predict import ModelNotAvailableError, predict_batch, predict_sentiment


def test_missing_model_raises_error():
    with patch("src.predict.check_fine_tuned_model_exists", return_value=False):
        with pytest.raises(ModelNotAvailableError):
            predict_sentiment("இந்த படம் அருமை")


def test_predict_sentiment_mocked():
    # Mock model and tokenizer
    mock_model = MagicMock()
    # Logits: [0.2, 1.8] -> softmax gives high probability for class 1 (positive)
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[0.2, 1.8]])
    mock_model.return_value = mock_outputs
    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }

    with patch("src.predict.check_fine_tuned_model_exists", return_value=True), \
         patch("src.predict.get_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)), \
         patch("src.predict.append_prediction_record", return_value={}):

        result = predict_sentiment(
            text="இந்த படம் மிகவும் அருமையாக உள்ளது",
            language="Tamil",
            log_to_history=False,
        )

        assert result["text"] == "இந்த படம் மிகவும் அருமையாக உள்ளது"
        assert result["language"] == "Tamil"
        assert result["predicted_sentiment"] in ["positive", "negative"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert "positive" in result["probabilities"]
        assert "negative" in result["probabilities"]

        auto_result = predict_sentiment(
            text="నమస్కారం ప్రపంచం",
            language="Auto-Detect",
            log_to_history=False,
        )
        assert auto_result["language"] == "Telugu"
        assert auto_result["predicted_sentiment"] in ["positive", "negative"]


def test_predict_batch_mocked():
    mock_model = MagicMock()
    # 2 samples in batch
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[0.1, 1.5], [1.9, 0.2]])
    mock_model.return_value = mock_outputs
    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.tensor([[1, 1], [1, 1]]),
    }

    df_input = pd.DataFrame({
        "text": ["यह बहुत अच्छा है", "यह बहुत खराब है"],
        "language": ["Hindi", "Hindi"],
    })

    import src.predict
    src.predict._LOADED_MODEL = None
    src.predict._LOADED_TOKENIZER = None

    with patch("src.predict.check_fine_tuned_model_exists", return_value=True), \
         patch("src.predict.get_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)):

        result_df = predict_batch(
            df_input,
            text_column="text",
            language_column="language",
            log_to_history=False,
            force_reload=True,
        )

        assert len(result_df) == 2
        assert "predicted_sentiment" in result_df.columns
        assert "confidence" in result_df.columns
        assert "prob_positive" in result_df.columns
        assert "prob_negative" in result_df.columns
