"""
Dataset preparation script for IndicBERT Multilingual Sentiment Classifier.
Downloads/compiles genuine AI4Bharat IndicSentiment dataset for Tamil, Hindi, Malayalam,
validates schema, cleans text, and performs stratified splitting into train/val/test splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.request

import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data_loader import (
    clean_and_validate_dataset,
    load_dataset_csv,
    save_processed_splits,
    split_dataset,
)
from src.utils import ensure_dir, setup_logger

logger = setup_logger("PrepareDataset")

# Fallback offline seed examples for pipeline demonstration when internet is disconnected
DEMO_OFFLINE_EXAMPLES = [
    # Tamil
    {"text": "இந்த படம் மிகவும் அற்புதமாக உள்ளது, அனைவரும் பார்க்க வேண்டும்.", "language": "Tamil", "label": "positive"},
    {"text": "மிகவும் அருமையான நடிப்பு மற்றும் சிறந்த இயக்கம்.", "language": "Tamil", "label": "positive"},
    {"text": "திரைக்கதை மிகவும் விறுவிறுப்பாகவும் சுவாரஸ்யமாகவும் அமைந்தது.", "language": "Tamil", "label": "positive"},
    {"text": "இந்த உணவகம் மிகச் சிறந்த உணவு மற்றும் சிறந்த சேவையை வழங்குகிறது.", "language": "Tamil", "label": "positive"},
    {"text": "தரமான தயாரிப்பு, மிகவும் பயனுள்ளது, முழு திருப்தி அளிக்கிறது.", "language": "Tamil", "label": "positive"},
    {"text": "மிகவும் மோசமான அனுபவம், முற்றிலும் நேர விரயம்.", "language": "Tamil", "label": "negative"},
    {"text": "படம் மிகவும் சலிப்பை ஏற்படுத்துகிறது, கதை இல்லை.", "language": "Tamil", "label": "negative"},
    {"text": "பொருளின் தரம் மிகக் கேவலமாக உள்ளது, வாங்க வேண்டாம்.", "language": "Tamil", "label": "negative"},
    {"text": "பயங்கரமான சேவை, ஊழியர்களின் நடத்தை மிகவும் மோசமாக இருந்தது.", "language": "Tamil", "label": "negative"},
    {"text": "பண விரயம், ஏமாற்றமே மிஞ்சியது, மிகவும் பரிதாபகரமானது.", "language": "Tamil", "label": "negative"},
    # Hindi
    {"text": "यह फिल्म बहुत ही शानदार और प्रेरणादायक है, जरूर देखें।", "language": "Hindi", "label": "positive"},
    {"text": "कलाकारों का अभिनय बहुत प्रभावशाली और उम्दा था।", "language": "Hindi", "label": "positive"},
    {"text": "कहानी और संगीत दोनों दिल को छू लेने वाले हैं।", "language": "Hindi", "label": "positive"},
    {"text": "यह उत्पाद बहुत उपयोगी और उच्च गुणवत्ता वाला है।", "language": "Hindi", "label": "positive"},
    {"text": "ग्राहक सेवा बहुत सहायक और विनम्र रही, पूरा संतोष मिला।", "language": "Hindi", "label": "positive"},
    {"text": "यह फिल्म बहुत ही खराब और निराशाजनक थी, समय की बर्बादी।", "language": "Hindi", "label": "negative"},
    {"text": "कथानक बिल्कुल बेकार और उबाऊ था, कोई नयापन नहीं।", "language": "Hindi", "label": "negative"},
    {"text": "सामान की गुणवत्ता बहुत घटिया है, बिल्कुल मत खरीदिए।", "language": "Hindi", "label": "negative"},
    {"text": "बहुत बुरा अनुभव रहा, डिलीवरी में बहुत देरी हुई।", "language": "Hindi", "label": "negative"},
    {"text": "पैसे की बर्बादी, उम्मीदों पर पानी फिर गया।", "language": "Hindi", "label": "negative"},
    # Malayalam
    {"text": "ഈ സിനിമ വളരെ മനോഹരവും ഹൃദയസ്പർശിയുമാണ്, തീർച്ചയായും കാണണം.", "language": "Malayalam", "label": "positive"},
    {"text": "അഭിനേതാക്കളുടെ പ്രകടനം മികച്ചുനിന്നു, മികച്ച സംവിധാനം.", "language": "Malayalam", "label": "positive"},
    {"text": "നല്ല കഥയും മനോഹരമായ ഗാനങ്ങളും ഉള്ള ഒരു നല്ല സിനിമ.", "language": "Malayalam", "label": "positive"},
    {"text": "ഈ ഉൽപ്പന്നം വളരെ മികച്ച ഗുണനിലവാരമുള്ളതാണ്, ഉപകാരപ്രദമാണ്.", "language": "Malayalam", "label": "positive"},
    {"text": "വളരെ നല്ല സർവീസ്, ഭക്ഷണം വളരെ രുചികരമായിരുന്നു.", "language": "Malayalam", "label": "positive"},
    {"text": "വളരെ മോശം അനുഭവം, വെറുതെ സമയം കളഞ്ഞു.", "language": "Malayalam", "label": "negative"},
    {"text": "സിനിമ വളരെ ബോറടിപ്പിച്ചു, കഥയിൽ ഒരു പുതുമയും ഇല്ല.", "language": "Malayalam", "label": "negative"},
    {"text": "സാധനത്തിന്റെ ക്വാളിറ്റി വളരെ മോശമാണ്, വാങ്ങരുത്.", "language": "Malayalam", "label": "negative"},
    {"text": "വളരെ മോശം കസ്റ്റമർ സർവീസ്, ജീവനക്കാരുടെ പെരുമാറ്റം ശരിയല്ല.", "language": "Malayalam", "label": "negative"},
    {"text": "പൈസ വേസ്റ്റ് ആയിപ്പോയി, വല്ലാത്ത നിരാശ തോന്നി.", "language": "Malayalam", "label": "negative"},
]


def download_ai4bharat_sentiment_data() -> pd.DataFrame:
    """
    Fetch genuine AI4Bharat IndicSentiment dataset for Tamil, Hindi, and Malayalam.
    Original Benchmark: IndicSentiment (AI4Bharat / ACL 2023 / CC-BY-4.0).
    """
    logger.info("Attempting to download genuine AI4Bharat IndicSentiment benchmark data...")
    records = []
    lang_mapping = {"ta": "Tamil", "hi": "Hindi", "ml": "Malayalam"}

    for code, lang_name in lang_mapping.items():
        for split in ["test", "validation"]:
            url = f"https://huggingface.co/datasets/ai4bharat/IndicSentiment/raw/main/data/{split}/{code}.json"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    for line in resp:
                        line_str = line.decode("utf-8").strip()
                        if not line_str:
                            continue
                        obj = json.loads(line_str)
                        text = obj.get("INDIC REVIEW", "")
                        label = obj.get("LABEL", "")
                        if text and label:
                            records.append({
                                "text": text,
                                "language": lang_name,
                                "label": label,
                            })
                logger.info(f"Successfully loaded {split}/{code}.json from Hugging Face.")
            except Exception as e:
                logger.warning(f"Could not download {url}: {e}")

    if len(records) > 100:
        logger.info(f"Downloaded {len(records)} authentic IndicSentiment records.")
        return pd.DataFrame(records)

    logger.warning("Download failed or incomplete. Falling back to structured demonstration dataset.")
    return pd.DataFrame(DEMO_OFFLINE_EXAMPLES)


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for IndicBERT sentiment classifier.")
    parser.add_argument("--raw-file", type=str, default=None, help="Path to raw CSV file.")
    parser.add_argument("--force-download", action="store_true", help="Force download from AI4Bharat.")
    args = parser.parse_args()

    config = load_config()
    raw_csv_path = Path(args.raw_file) if args.raw_file else config.raw_file
    ensure_dir(config.raw_dir)
    ensure_dir(config.processed_dir)

    # Also save a tiny demo CSV for test verification
    demo_path = config.raw_dir / "sample_demo_indic.csv"
    pd.DataFrame(DEMO_OFFLINE_EXAMPLES).to_csv(demo_path, index=False, encoding="utf-8")
    logger.info(f"Created reference demonstration CSV at: {demo_path}")

    # Check if raw dataset exists or download
    if raw_csv_path.exists() and not args.force_download:
        logger.info(f"Loading existing raw dataset from: {raw_csv_path}")
        raw_df = load_dataset_csv(raw_csv_path)
    else:
        logger.info("Raw dataset not found or download requested. Downloading AI4Bharat benchmark...")
        raw_df = download_ai4bharat_sentiment_data()
        raw_df.to_csv(raw_csv_path, index=False, encoding="utf-8")
        logger.info(f"Saved compiled raw dataset to: {raw_csv_path} ({len(raw_df)} rows)")

    # Clean and validate
    cleaned_df, audit = clean_and_validate_dataset(
        raw_df,
        supported_languages=config.supported_languages,
        supported_labels=config.to_dict().get("model", {}).get("label2id", {}).keys(),
    )
    logger.info(f"Cleaned dataset: {len(cleaned_df)} samples remaining.")

    # Stratified split
    train_df, val_df, test_df = split_dataset(
        cleaned_df,
        train_ratio=config.to_dict()["data"]["split_ratios"]["train"],
        val_ratio=config.to_dict()["data"]["split_ratios"]["val"],
        test_ratio=config.to_dict()["data"]["split_ratios"]["test"],
        random_seed=config.seed,
    )

    # Save splits and summary
    saved_paths = save_processed_splits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        output_dir=config.processed_dir,
        audit_summary=audit,
    )

    logger.info("Dataset preparation successfully completed!")
    logger.info(f"  Train set:      {saved_paths['train']} ({len(train_df)} rows)")
    logger.info(f"  Validation set: {saved_paths['val']} ({len(val_df)} rows)")
    logger.info(f"  Test set:       {saved_paths['test']} ({len(test_df)} rows)")
    logger.info(f"  Summary JSON:   {saved_paths['summary']}")


if __name__ == "__main__":
    main()
