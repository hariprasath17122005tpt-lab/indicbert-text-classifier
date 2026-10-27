"""
Tests for text preprocessing and script detection.
Verifies native Indic character preservation and Unicode handling.
"""

import pytest
from src.preprocessing import clean_indic_text, detect_indic_script, get_text_statistics


def test_clean_indic_text_preserves_tamil():
    sample = "இந்த   படம் \t மிகவும் \n அருமையாக உள்ளது!"
    cleaned = clean_indic_text(sample)
    assert cleaned == "இந்த படம் மிகவும் அருமையாக உள்ளது!"
    # Ensure native Tamil Unicode characters are strictly intact
    assert "இந்த" in cleaned
    assert "அருமையாக" in cleaned


def test_clean_indic_text_preserves_hindi():
    sample = "यह  फिल्म   बहुत <br> अच्छी है। &amp; शानदार!"
    cleaned = clean_indic_text(sample)
    assert "यह फिल्म बहुत अच्छी है।" in cleaned
    assert "&" in cleaned
    assert "<br>" not in cleaned


def test_clean_indic_text_preserves_malayalam():
    sample = "ഈ സിനിമ   വളരെ നല്ലതാണ്...   "
    cleaned = clean_indic_text(sample)
    assert cleaned == "ഈ സിനിമ വളരെ നല്ലതാണ്..."
    assert "സിനിമ" in cleaned


def test_clean_indic_text_handles_empty_and_null():
    assert clean_indic_text(None) == ""
    assert clean_indic_text("") == ""
    assert clean_indic_text("   \t  \n ") == ""


def test_detect_indic_script():
    tamil_text = "வணக்கம் நண்பா"
    hindi_text = "नमस्ते दुनिया"
    malayalam_text = "ഹലോ സുഹൃത്തേ"
    english_text = "Hello world"

    assert detect_indic_script(tamil_text) == "Tamil"
    assert detect_indic_script(hindi_text) == "Hindi"
    assert detect_indic_script(malayalam_text) == "Malayalam"
    assert detect_indic_script(english_text) is None


@pytest.mark.parametrize(
    ("text", "expected_language"),
    [
        ("નમસ્તે દુનિયા", "Gujarati"),
        ("ਸਤ ਸ੍ਰੀ ਅਕਾਲ", "Punjabi"),
        ("ନମସ୍କାର ବିଶ୍ୱ", "Odia"),
        ("నమస్కారం ప్రపంచం", "Telugu"),
        ("ನಮಸ್ಕಾರ ಜಗತ್ತು", "Kannada"),
        ("নমস্কাৰ পৃথিৱী", "Assamese"),
        ("নমস্কার বিশ্ব", "Bengali"),
        ("السلام علیکم", "Urdu"),
        ("ꯃꯤꯇꯩ ꯂꯥꯢꯄꯥ", "Manipuri"),
        ("ᱥᱟᱱᱛᱟᱲᱤ", "Santali"),
    ],
)
def test_detect_indic_script_supports_additional_scripts(text, expected_language):
    assert detect_indic_script(text) == expected_language


def test_scheduled_indic_language_options_cover_all_22_languages():
    from src.preprocessing import SCHEDULED_INDIC_LANGUAGES

    assert len(SCHEDULED_INDIC_LANGUAGES) == 22
    assert len(set(SCHEDULED_INDIC_LANGUAGES)) == 22


def test_text_statistics():
    texts = [
        "இந்த படம் அருமை",
        "यह फिल्म अच्छी है और सुंदर है",
        "വളരെ നല്ല സിനിമ",
    ]
    stats = get_text_statistics(texts)
    assert stats["total_texts"] == 3
    assert stats["char_len_min"] > 0
    assert stats["word_len_mean"] > 0
