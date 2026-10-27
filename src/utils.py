"""
General utilities module.
Seed control, device management, logging, and file helpers.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# On Windows, ensure standard output handles UTF-8 Indic characters without charmap errors
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds across random, numpy, and torch for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Automatically detect and return GPU device if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_logger(
    name: str = "IndicClassifier",
    log_file: Optional[Path | str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Setup a clean, informative logger for console and optional file logging.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_dir(path: Path | str) -> Path:
    """
    Ensure the parent or directory itself exists.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Dict[str, Any], file_path: Path | str) -> None:
    """
    Save dictionary to JSON with utf-8 encoding and indented formatting.
    """
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(file_path: Path | str) -> Dict[str, Any]:
    """
    Load JSON file safely with utf-8 encoding.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file does not exist: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
