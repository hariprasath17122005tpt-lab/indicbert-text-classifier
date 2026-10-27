# Processed Data Directory

This directory stores the cleaned, normalized, and stratified dataset partitions.

- **`train.csv`**: Training set (70% = 2,418 rows) used for IndicBERT sequence classification fine-tuning.
- **`val.csv`**: Validation set (15% = 519 rows) used for epoch-level evaluation, learning rate scheduling, and early stopping.
- **`test.csv`**: Held-out test set (15% = 519 rows) strictly reserved for final unbiased performance evaluation.
- **`dataset_summary.json`**: Machine-readable metadata audit containing sample counts, class distributions, and text length statistics.

### Stratification:
Splits are stratified across a composite `language_label` stratum, guaranteeing identical representation of Tamil, Hindi, and Malayalam positive/negative reviews across train, val, and test partitions without data leakage.
