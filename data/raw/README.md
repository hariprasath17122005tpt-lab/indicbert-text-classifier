# Raw Data Directory

This directory stores the unpartitioned, compiled dataset for multilingual Indic sentiment analysis.

- **Primary Raw File**: `indic_sentiment_raw.csv` (3,462 rows)
- **Reference Demonstration File**: `sample_demo_indic.csv` (30 rows, structured offline backup)
- **Columns**: `text, language, label`
- **Languages**: Tamil (`Tamil`), Hindi (`Hindi`), Malayalam (`Malayalam`)
- **Labels**: `Positive`, `Negative`
- **Benchmark Source**: AI4Bharat IndicSentiment (Creative Commons Attribution 4.0 International)
- **Generation Script**: `python scripts/prepare_dataset.py`
