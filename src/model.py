"""
Model and tokenizer management module.
Loads, configures, and saves IndicBERT checkpoints and fine-tuned classifiers.
Uses mean-pooling transfer learning to handle MLM-only pretraining representations.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import struct
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from src.utils import ensure_dir, setup_logger

logger = setup_logger("ModelManager")


class ModelLoadError(Exception):
    """Raised when IndicBERT model or tokenizer cannot be loaded."""
    pass


class IndicBERTForSequenceClassification(BertPreTrainedModel):
    """
    Multilingual IndicBERT sequence classification model.
    Uses mean-pooling across active token representations with attention masking
    to aggregate rich sentence semantics for Indian languages (Tamil, Hindi, Malayalam).
    """

    _keys_to_ignore_on_load_missing = [r"position_ids", r"bert.pooler.dense.weight", r"bert.pooler.dense.bias"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if getattr(config, "classifier_dropout", None) is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, config.num_labels),
        )
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # Mean pooling over active tokens using attention_mask
        if attention_mask is not None:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = sequence_output.mean(dim=1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def resolve_model_path(model_name_or_path: str | Path) -> str:
    """
    Check if a model exists in local cache snapshot or local directory.
    Avoids unnecessary network queries and gated checks if cached locally.
    """
    model_str = str(model_name_or_path)
    # 1. Direct local path check
    p = Path(model_str)
    if p.exists() and p.is_dir() and (p / "config.json").exists():
        return str(p.resolve())

    # 2. Hugging Face local cache snapshot check
    cache_hub = Path.home() / ".cache" / "huggingface" / "hub"
    repo_folder = f"models--{model_str.replace('/', '--')}"
    snapshot_dir = cache_hub / repo_folder / "snapshots"
    if snapshot_dir.exists():
        snapshots = [d for d in snapshot_dir.iterdir() if d.is_dir() and (d / "config.json").exists()]
        if snapshots:
            logger.info(f"Using locally cached snapshot for '{model_str}': {snapshots[0]}")
            return str(snapshots[0].resolve())

    return model_str


def load_tokenizer(
    model_name_or_path: str | Path,
    use_fast: bool = True,
) -> PreTrainedTokenizerBase:
    """
    Load tokenizer from local path or Hugging Face repository.
    Includes friendly diagnostics for gated repositories or missing files.
    """
    target = resolve_model_path(model_name_or_path)
    logger.info(f"Loading tokenizer from: {target}")

    try:
        tokenizer_config_path = Path(target) / "tokenizer_config.json"
        tokenizer_path = Path(target) / "tokenizer.json"
        if tokenizer_config_path.is_file() and tokenizer_path.is_file():
            with tokenizer_config_path.open(encoding="utf-8") as tokenizer_config_file:
                tokenizer_config = json.load(tokenizer_config_file)

            if tokenizer_config.get("tokenizer_class") == "TokenizersBackend":
                special_tokens = {
                    token_name: tokenizer_config[token_name]
                    for token_name in (
                        "unk_token",
                        "cls_token",
                        "sep_token",
                        "pad_token",
                        "mask_token",
                        "model_max_length",
                    )
                    if token_name in tokenizer_config
                }
                return PreTrainedTokenizerFast(
                    tokenizer_file=str(tokenizer_path),
                    **special_tokens,
                )

        tokenizer = AutoTokenizer.from_pretrained(target, use_fast=use_fast)
        return tokenizer
    except Exception as e:
        err_msg = str(e)
        if "gated repo" in err_msg.lower() or "401" in err_msg:
            raise ModelLoadError(
                f"Repository '{model_name_or_path}' is gated or requires authentication.\n"
                "To resolve:\n"
                "1. If using 'ai4bharat/indic-bert', consider using the ungated 'ai4bharat/IndicBERTv2-MLM-only' checkpoint instead.\n"
                "2. Or run `huggingface-cli login` in your terminal with your Hugging Face access token.\n"
                f"Underlying error: {e}"
            ) from e
        raise ModelLoadError(f"Failed to load tokenizer from '{model_name_or_path}': {e}") from e


def get_model_and_tokenizer(
    model_name_or_path: str | Path,
    num_labels: int = 2,
    id2label: Optional[Dict[int, str]] = None,
    label2id: Optional[Dict[str, int]] = None,
    memory_mapped: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load both tokenizer and sequence classification model.
    Instantiates IndicBERTForSequenceClassification with mean-pooling head.
    Preserves exact label mappings.
    """
    target = resolve_model_path(model_name_or_path)
    logger.info(f"Loading model and tokenizer from: {target} (num_labels={num_labels})")

    tokenizer = load_tokenizer(target)

    # Standardize label mappings if not provided
    if id2label is None:
        id2label = {0: "negative", 1: "positive"}
    if label2id is None:
        label2id = {"negative": 0, "positive": 1}

    try:
        config = AutoConfig.from_pretrained(
            target,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        local_weights = Path(target) / "model.safetensors"
        if local_weights.is_file() and memory_mapped:
            with torch.device("meta"):
                model = IndicBERTForSequenceClassification(config)
            model = _load_safetensors_weights_into_model(model, local_weights)
            model.requires_grad_(False)
        elif local_weights.is_file():
            try:
                model = IndicBERTForSequenceClassification.from_pretrained(
                    target,
                    config=config,
                )
            except Exception as e_custom:
                logger.warning(
                    "Could not load as IndicBERTForSequenceClassification: "
                    f"{e_custom}. Falling back to AutoModel."
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    target,
                    config=config,
                )
        else:
            try:
                model = IndicBERTForSequenceClassification.from_pretrained(
                    target,
                    config=config,
                )
            except Exception as e_custom:
                logger.warning(
                    "Could not load as IndicBERTForSequenceClassification: "
                    f"{e_custom}. Falling back to AutoModel."
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    target,
                    config=config,
                )
        return model, tokenizer

    except Exception as e:
        err_msg = str(e)
        if "gated repo" in err_msg.lower() or "401" in err_msg:
            raise ModelLoadError(
                f"Model repository '{model_name_or_path}' requires authentication or is gated.\n"
                "Please use 'ai4bharat/IndicBERTv2-MLM-only' or log in via `huggingface-cli login`.\n"
                f"Details: {e}"
            ) from e
        raise ModelLoadError(f"Failed to load classification model from '{model_name_or_path}': {e}") from e


def _load_safetensors_weights_into_model(
    model: PreTrainedModel,
    weights_path: Path,
) -> PreTrainedModel:
    """Attach local checkpoint tensors through a copy-on-write memory map."""
    dtype_map = {
        "F64": np.float64,
        "F32": np.float32,
        "F16": np.float16,
        "BF16": np.uint16,
        "I64": np.int64,
        "I32": np.int32,
        "I16": np.int16,
        "I8": np.int8,
        "U8": np.uint8,
        "BOOL": np.bool_,
    }

    file_size = weights_path.stat().st_size
    with weights_path.open("rb") as checkpoint_file:
        header_size_bytes = checkpoint_file.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError("Checkpoint is missing its Safetensors header.")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        if header_size > file_size - 8:
            raise ValueError("Checkpoint Safetensors header exceeds the file size.")
        header = json.loads(checkpoint_file.read(header_size))

    model_keys = set(model.state_dict())
    checkpoint_keys = set(header) - {"__metadata__"}
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    if missing_keys or unexpected_keys:
        raise ValueError(
            "Checkpoint keys do not match the classifier model "
            f"(missing: {sorted(missing_keys)}, unexpected: {sorted(unexpected_keys)})."
        )

    data_start = 8 + header_size
    data_size = file_size - data_start
    backing = np.memmap(weights_path, dtype=np.uint8, mode="c")
    for name, tensor_info in header.items():
        if name == "__metadata__":
            continue
        try:
            dtype = np.dtype(dtype_map[tensor_info["dtype"]])
        except KeyError as e:
            raise ValueError(
                f"Unsupported Safetensors dtype for '{name}': {tensor_info['dtype']}"
            ) from e

        start, end = tensor_info["data_offsets"]
        shape = tuple(tensor_info["shape"])
        expected_size = math.prod(shape) * dtype.itemsize
        if start < 0 or end < start or end > data_size or end - start != expected_size:
            raise ValueError(f"Invalid tensor data range in checkpoint for '{name}'.")

        array = np.ndarray(
            shape,
            dtype=dtype,
            buffer=backing,
            offset=data_start + start,
        )
        tensor = torch.from_numpy(array)
        if tensor_info["dtype"] == "BF16":
            tensor = tensor.view(torch.bfloat16)
        model.load_state_dict({name: tensor}, strict=False, assign=True)

    unloaded_keys = [
        name for name, parameter in model.named_parameters() if parameter.is_meta
    ]
    if unloaded_keys:
        raise ValueError(
            f"Checkpoint left model parameters unloaded: {', '.join(unloaded_keys)}"
        )

    for module in model.modules():
        for name, buffer in module.named_buffers(recurse=False):
            if buffer.is_meta:
                if name == "position_ids":
                    value = torch.arange(buffer.shape[-1], dtype=buffer.dtype).expand(
                        buffer.shape
                    )
                elif name == "token_type_ids":
                    value = torch.zeros(buffer.shape, dtype=buffer.dtype)
                else:
                    raise ValueError(f"Unsupported unloaded model buffer: '{name}'.")
                setattr(module, name, value)

    model._safetensors_mmap = backing
    return model


def save_fine_tuned_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Path | str,
    training_info: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save fine-tuned model weights, tokenizer vocab/files, config, and metadata.
    Saves in standard precision to prevent CPU float16 allocation overhead.
    """
    out_path = ensure_dir(output_dir)
    logger.info(f"Saving fine-tuned model and tokenizer to: {out_path}")

    # Remove existing weight files in target dir before saving new weights
    for weight_name in ["model.safetensors", "pytorch_model.bin"]:
        existing = out_path / weight_name
        if existing.exists():
            try:
                existing.unlink()
                logger.info(f"Removed prior {weight_name} before saving new weights.")
            except Exception as e:
                logger.warning(f"Could not remove {existing}: {e}")

    # Save model and tokenizer
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)

    if training_info:
        from src.utils import save_json
        save_json(training_info, out_path / "model_training_meta.json")

    logger.info(f"Model successfully saved at: {out_path}")
    return out_path


def check_fine_tuned_model_exists(model_dir: Path | str) -> bool:
    """
    Verify if a valid fine-tuned model exists in the directory.
    Checks for config.json and model weights (safetensors or bin).
    """
    p = Path(model_dir)
    if not p.exists() or not p.is_dir():
        return False

    has_config = (p / "config.json").exists()
    has_weights = (
        (p / "model.safetensors").exists()
        or (p / "pytorch_model.bin").exists()
    )
    return has_config and has_weights
