"""
IndicBERT transfer learning and fine-tuning module.
Fine-tunes IndicBERT on multilingual Indic sentiment data with validation tracking,
early stopping, and best checkpoint selection based on validation Macro F1.
Supports mean-pooling contextual transfer learning for MLM-only IndicBERT checkpoints.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer, get_linear_schedule_with_warmup

from src.config import ProjectConfig, load_config
from src.metrics import compute_classification_metrics
from src.model import get_model_and_tokenizer, save_fine_tuned_model
from src.utils import ensure_dir, get_device, save_json, set_seed, setup_logger

logger = setup_logger("Trainer")


class IndicSentimentDataset(Dataset):
    """
    PyTorch Dataset for text classification.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        languages: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.languages = languages
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        return item


def prepare_dataloader(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    label2id: Dict[str, int],
    max_length: int = 128,
    batch_size: int = 16,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a PyTorch DataLoader from a DataFrame.
    """
    texts = df["text"].tolist()
    labels = [label2id[str(lbl).lower()] for lbl in df["label"].tolist()]
    languages = df["language"].tolist()

    ds = IndicSentimentDataset(
        texts=texts,
        labels=labels,
        languages=languages,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def extract_pooled_features(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    batch_size: int = 32,
    max_length: int = 128,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Extract mean-pooled sentence representations from frozen IndicBERT backbone.
    Enables orders-of-magnitude faster CPU transfer learning without redundant forward passes.
    """
    model.eval()
    base_bert = getattr(model, "bert", model)
    all_features = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_slice = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_slice,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = base_bert(**inputs)
            sequence_output = outputs[0]

            mask = inputs["attention_mask"].unsqueeze(-1).expand(sequence_output.size()).float()
            sum_embeddings = torch.sum(sequence_output * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask

            all_features.append(pooled.cpu())

    return torch.cat(all_features, dim=0)


def evaluate_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    id2label: Dict[int, str],
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate model on a dataloader. Returns average loss and classification metrics.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

            outputs = model(**kwargs)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(dataloader))
    class_names = [id2label[i] for i in sorted(id2label.keys())]
    metrics = compute_classification_metrics(all_labels, all_preds, class_names=class_names)
    metrics["loss"] = round(avg_loss, 4)

    return avg_loss, metrics


def train_model(
    config: Optional[ProjectConfig] = None,
    smoke_test: bool = False,
    max_samples: Optional[int] = None,
    custom_train_df: Optional[pd.DataFrame] = None,
    custom_val_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path | str] = None,
    save_checkpoint: bool = True,
) -> Dict[str, Any]:
    """
    Execute transfer learning / fine-tuning training loop for IndicBERT.
    
    Supports:
    - Fast, memory-safe transfer learning via frozen backbone & mean-pooled classifier head
    - Full end-to-end backpropagation when freeze_base_model is False
    - Low-resource fractional evaluation
    - CI smoke-testing
    """
    if config is None:
        config = load_config()

    set_seed(config.seed)
    device = get_device()
    logger.info(f"Using device for training: {device}")

    # Resolve paths
    target_output_dir = Path(output_dir) if output_dir else config.local_model_dir

    # Hyperparameters
    if smoke_test:
        logger.info("--- Running in SMOKE-TEST mode (fast integration check) ---")
        num_epochs = config.smoke_test_epochs
        batch_size = config.smoke_test_batch_size
        sample_limit = config.smoke_test_max_samples
        max_length = 64
    else:
        num_epochs = config.num_epochs
        batch_size = config.batch_size
        sample_limit = max_samples
        max_length = config.max_length

    lr = config.learning_rate
    weight_decay = config.weight_decay
    label2id = config.label2id
    id2label = config.id2label

    # 1. Load data
    if custom_train_df is not None:
        train_df = custom_train_df.copy()
    else:
        train_df = pd.read_csv(config.train_file, encoding="utf-8")

    if custom_val_df is not None:
        val_df = custom_val_df.copy()
    else:
        val_df = pd.read_csv(config.val_file, encoding="utf-8")

    if sample_limit:
        train_df = train_df.head(sample_limit)
        val_df = val_df.head(min(sample_limit, len(val_df)))

    logger.info(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    # 2. Load model and matching tokenizer
    model, tokenizer = get_model_and_tokenizer(
        model_name_or_path=config.model_name_or_path,
        num_labels=config.num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    freeze_base = config.freeze_base_model
    trainable_mode = config.trainable_layers

    if freeze_base:
        base_encoder = getattr(model, "bert", getattr(model, "albert", None))
        if base_encoder is not None:
            for param in base_encoder.parameters():
                param.requires_grad = False

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable parameters: {trainable_count:,} / {total_count:,} ({trainable_count / total_count * 100:.3f}%)"
    )

    model.to(device)

    # -------------------------------------------------------------------------
    # PATH A: FAST FEATURE-CACHED TRANSFER LEARNING (when base is frozen)
    # -------------------------------------------------------------------------
    if freeze_base and hasattr(model, "classifier"):
        logger.info("Executing optimized representation transfer learning...")
        t_start = time.time()

        train_texts = train_df["text"].tolist()
        val_texts = val_df["text"].tolist()

        train_labels_tensor = torch.tensor(
            [label2id[str(lbl).lower()] for lbl in train_df["label"].tolist()],
            dtype=torch.long,
        )
        val_labels_tensor = torch.tensor(
            [label2id[str(lbl).lower()] for lbl in val_df["label"].tolist()],
            dtype=torch.long,
        )

        logger.info(f"Extracting sentence embeddings for {len(train_texts)} train & {len(val_texts)} val texts...")
        train_features = extract_pooled_features(
            model=model,
            tokenizer=tokenizer,
            texts=train_texts,
            batch_size=batch_size * 2,
            max_length=max_length,
            device=device,
        )
        val_features = extract_pooled_features(
            model=model,
            tokenizer=tokenizer,
            texts=val_texts,
            batch_size=batch_size * 2,
            max_length=max_length,
            device=device,
        )

        optimizer = AdamW(model.classifier.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        history: List[Dict[str, Any]] = []
        best_val_macro_f1 = -1.0
        best_classifier_weights = None
        best_epoch = -1
        early_stopping_counter = 0

        train_epochs = 20 if (not smoke_test and num_epochs < 5) else num_epochs

        for epoch in range(1, train_epochs + 1):
            epoch_start = time.time()
            model.classifier.train()
            perm = torch.randperm(len(train_features))
            total_train_loss = 0.0

            for i in range(0, len(train_features), batch_size):
                idx = perm[i : i + batch_size]
                batch_feat = train_features[idx].to(device)
                batch_y = train_labels_tensor[idx].to(device)

                optimizer.zero_grad()
                logits = model.classifier(model.dropout(batch_feat))
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / max(1, (len(train_features) // batch_size))

            # Validation evaluation
            model.classifier.eval()
            with torch.no_grad():
                val_feat_dev = val_features.to(device)
                val_y_dev = val_labels_tensor.to(device)
                val_logits = model.classifier(val_feat_dev)
                val_loss_val = loss_fn(val_logits, val_y_dev).item()
                val_preds = val_logits.argmax(dim=-1).cpu().numpy().tolist()
                val_true = val_labels_tensor.numpy().tolist()

            class_names = [id2label[i] for i in sorted(id2label.keys())]
            val_metrics = compute_classification_metrics(val_true, val_preds, class_names=class_names)
            val_macro_f1 = val_metrics["macro_f1"]
            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch:2d}/{train_epochs} [{epoch_time:.2f}s] - "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss_val:.4f} | "
                f"Val Macro F1: {val_macro_f1:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
            )

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 4),
                "val_loss": round(val_loss_val, 4),
                "val_macro_f1": val_macro_f1,
                "val_accuracy": val_metrics["accuracy"],
                "epoch_duration_sec": round(epoch_time, 2),
            }
            history.append(epoch_record)

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                best_epoch = epoch
                best_classifier_weights = copy.deepcopy(model.classifier.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= config.early_stopping_patience and not smoke_test:
                    logger.info(f"Early stopping at epoch {epoch}.")
                    break

            if smoke_test:
                break

        if best_classifier_weights is not None:
            model.classifier.load_state_dict(best_classifier_weights)

        total_duration = time.time() - t_start

    # -------------------------------------------------------------------------
    # PATH B: STANDARD END-TO-END FINE-TUNING LOOP
    # -------------------------------------------------------------------------
    else:
        train_loader = prepare_dataloader(
            train_df, tokenizer, label2id, max_length=max_length, batch_size=batch_size, shuffle=True
        )
        val_loader = prepare_dataloader(
            val_df, tokenizer, label2id, max_length=max_length, batch_size=batch_size, shuffle=False
        )

        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g["params"]]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

        total_training_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_training_steps * config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        history = []
        best_val_macro_f1 = -1.0
        best_model_weights = None
        best_epoch = -1
        early_stopping_counter = 0
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            model.train()
            train_loss = 0.0

            for step, batch in enumerate(train_loader, 1):
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if "token_type_ids" in batch:
                    kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

                outputs = model(**kwargs)
                logits = outputs.logits
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / max(1, len(train_loader))
            val_loss, val_metrics = evaluate_epoch(model, val_loader, device, id2label)
            val_macro_f1 = val_metrics["macro_f1"]
            epoch_time = time.time() - epoch_start

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_macro_f1": val_macro_f1,
                "val_accuracy": val_metrics["accuracy"],
                "epoch_duration_sec": round(epoch_time, 2),
            }
            history.append(epoch_record)

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                best_epoch = epoch
                best_model_weights = copy.deepcopy(model.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= config.early_stopping_patience and not smoke_test:
                    break

        total_duration = time.time() - start_time
        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)

    # 3. Save fine-tuned model and metadata
    training_summary = {
        "base_model": config.model_name_or_path,
        "num_epochs": len(history),
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_macro_f1, 4),
        "total_duration_seconds": round(total_duration, 2),
        "history": history,
        "smoke_test": smoke_test,
        "num_train_samples": len(train_df),
        "num_val_samples": len(val_df),
        "device": str(device),
    }

    if save_checkpoint:
        save_fine_tuned_model(
            model=model,
            tokenizer=tokenizer,
            output_dir=target_output_dir,
            training_info=training_summary,
        )
        save_json(training_summary, config.training_history_file)
    else:
        logger.info("Skipping model checkpoint serialization to disk (in-memory mode).")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "summary": training_summary,
        "history": history,
        "best_val_macro_f1": best_val_macro_f1,
    }
