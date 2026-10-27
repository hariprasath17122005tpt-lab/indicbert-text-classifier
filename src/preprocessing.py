"""
Text preprocessing and script analysis module.
Preserves native Indic scripts and cleans text.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any, Dict, List, Optional
import torch


# The 22 scheduled Indic languages. This is separate from the training dataset's
# supported_languages setting, which describes the languages in its annotations.
SCHEDULED_INDIC_LANGUAGES = (
    "Assamese",
    "Bengali",
    "Bodo",
    "Dogri",
    "Gujarati",
    "Hindi",
    "Kannada",
    "Kashmiri",
    "Konkani",
    "Maithili",
    "Malayalam",
    "Manipuri",
    "Marathi",
    "Nepali",
    "Odia",
    "Punjabi",
    "Sanskrit",
    "Santali",
    "Sindhi",
    "Tamil",
    "Telugu",
    "Urdu",
)

# Unicode blocks map to a representative language when multiple languages
# share a script. Assamese has distinct letters in the Bengali block.
INDIC_UNICODE_RANGES = {
    "Hindi": (0x0900, 0x097F),
    "Punjabi": (0x0A00, 0x0A7F),
    "Gujarati": (0x0A80, 0x0AFF),
    "Odia": (0x0B00, 0x0B7F),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
    "Bengali": (0x0980, 0x09FF),
    "Urdu": (0x0600, 0x06FF),
    "Manipuri": (0xABC0, 0xABFF),
    "Santali": (0x1C50, 0x1C7F),
}
ASSAMESE_UNIQUE_CHARACTERS = frozenset("\u09f0\u09f1")


def clean_indic_text(text: Any) -> str:
    """
    Clean and normalize Indic text while strictly preserving native scripts.
    
    Operations:
    1. Converts input to string; handles None or NaN.
    2. Unescapes HTML entities (&amp;, &lt;, &gt;, etc.).
    3. Removes HTML tags (<br>, <p>, etc.).
    4. Normalizes Unicode representations to NFC format.
    5. Normalizes multiple whitespace characters, tabs, and newlines into a single space.
    6. Strips leading and trailing whitespace.
    
    IMPORTANT: Native characters, punctuation, and numerals are never deleted.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # 1. Unescape HTML entities
    text = html.unescape(text)

    # 2. Strip HTML tags if present
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. Unicode NFC normalization (ensures consistent diacritic / matra composition)
    text = unicodedata.normalize("NFC", text)

    # 4. Collapse whitespace
    text = re.sub(r"[\s\t\r\n]+", " ", text)

    # 5. Trim ends
    return text.strip()


def detect_indic_script(text: str) -> Optional[str]:
    """
    Detect a language from the predominant Indic script in the input text.

    Script detection cannot distinguish languages that share a script (for
    example Hindi and Marathi). In such cases this returns a representative
    language; callers should allow the user to specify the language.
    """
    if not text:
        return None

    counts = {lang: 0 for lang in INDIC_UNICODE_RANGES}
    has_assamese_character = False
    for char in text:
        code = ord(char)
        if char in ASSAMESE_UNIQUE_CHARACTERS:
            has_assamese_character = True
        for lang, (start, end) in INDIC_UNICODE_RANGES.items():
            if start <= code <= end:
                counts[lang] += 1
                break

    best_lang, best_count = max(counts.items(), key=lambda item: item[1])
    if best_count > 0:
        if best_lang == "Bengali" and has_assamese_character:
            return "Assamese"
        return best_lang
    return None


def get_text_statistics(texts: List[str]) -> Dict[str, Any]:
    """
    Compute length statistics (word count, character count) across a collection of texts.
    """
    if not texts:
        return {
            "total_texts": 0,
            "char_len_mean": 0.0,
            "char_len_min": 0,
            "char_len_max": 0,
            "word_len_mean": 0.0,
            "word_len_min": 0,
            "word_len_max": 0,
        }

    char_lens = [len(t) for t in texts]
    word_lens = [len(t.split()) for t in texts]

    return {
        "total_texts": len(texts),
        "char_len_mean": round(float(sum(char_lens) / len(char_lens)), 2),
        "char_len_min": int(min(char_lens)),
        "char_len_max": int(max(char_lens)),
        "word_len_mean": round(float(sum(word_lens) / len(word_lens)), 2),
        "word_len_min": int(min(word_lens)),
        "word_len_max": int(max(word_lens)),
    }


def tokenize_texts(
    texts: List[str],
    tokenizer: Any,
    max_length: int = 128,
    return_tensors: Optional[str] = "pt",
) -> Dict[str, torch.Tensor]:
    """
    Tokenize a batch of texts using the given tokenizer with truncation and padding.
    """
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors=return_tensors,
    )
