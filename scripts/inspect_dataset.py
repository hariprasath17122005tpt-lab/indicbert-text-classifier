"""
Exploratory Data Analysis (EDA) and dataset inspection script.
Analyzes dataset statistics, class balance, language coverage, and generates
publication-quality EDA plots into results/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.utils import ensure_dir, setup_logger

logger = setup_logger("InspectDataset")


def generate_eda_visualizations(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Generate EDA charts and save to results directory.
    """
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Language Distribution Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    lang_counts = df["language"].value_counts().reset_index()
    lang_counts.columns = ["language", "count"]
    sns.barplot(data=lang_counts, x="language", y="count", hue="language", ax=ax, palette="Blues_r", legend=False)
    ax.set_title("Sample Distribution by Indic Language", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Language", fontsize=11, labelpad=8)
    ax.set_ylabel("Number of Reviews", fontsize=11, labelpad=8)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center", va="center", xytext=(0, 6), textcoords="offset points",
                fontweight="bold"
            )
    plt.tight_layout()
    lang_plot_path = output_dir / "eda_language_dist.png"
    fig.savefig(lang_plot_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved language distribution plot: {lang_plot_path}")

    # 2. Sentiment Distribution by Language Plot
    # Normalize label representation for uniform plotting
    df_plot = df.copy()
    df_plot["label_norm"] = df_plot["label"].astype(str).str.lower()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.countplot(
        data=df_plot,
        x="language",
        hue="label_norm",
        ax=ax,
        palette={"positive": "#22C55E", "negative": "#EF4444"},
    )
    ax.set_title("Sentiment Balance per Language", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Language", fontsize=11, labelpad=8)
    ax.set_ylabel("Count", fontsize=11, labelpad=8)
    ax.legend(title="Sentiment", loc="upper right")
    plt.tight_layout()
    sent_plot_path = output_dir / "eda_label_dist.png"
    fig.savefig(sent_plot_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved sentiment distribution plot: {sent_plot_path}")

    # 3. Word Length Distribution Plot
    df["word_count"] = df["text"].astype(str).apply(lambda t: len(t.split()))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(data=df, x="word_count", hue="language", element="step", common_norm=False, kde=True, ax=ax)
    ax.set_title("Review Length Distribution (Word Count) by Language", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Word Count per Review", fontsize=11, labelpad=8)
    ax.set_ylabel("Density / Count", fontsize=11, labelpad=8)
    plt.tight_layout()
    len_plot_path = output_dir / "eda_text_length.png"
    fig.savefig(len_plot_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved text length distribution plot: {len_plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Inspect dataset and generate EDA statistics.")
    parser.add_argument("--file", type=str, default=None, help="CSV file to inspect.")
    args = parser.parse_args()

    config = load_config()
    target_csv = Path(args.file) if args.file else config.raw_file

    if not target_csv.exists():
        logger.error(f"Target dataset file not found: {target_csv}. Please run `python scripts/prepare_dataset.py` first.")
        sys.exit(1)

    df = pd.read_csv(target_csv, encoding="utf-8")
    results_dir = ensure_dir(config.results_dir)

    print("\n" + "=" * 60)
    print("INDIC MULTILINGUAL DATASET AUDIT REPORT")
    print("=" * 60)
    print(f"Dataset path:            {target_csv.resolve()}")
    print(f"Total rows:              {len(df)}")
    print(f"Columns present:         {list(df.columns)}")
    print("\n--- Language Distribution ---")
    print(df["language"].value_counts().to_string())
    print("\n--- Sentiment Label Distribution ---")
    print(df["label"].value_counts().to_string())
    print("\n--- Cross-Tabulation (Language x Sentiment) ---")
    print(pd.crosstab(df["language"], df["label"]))

    df["char_count"] = df["text"].astype(str).apply(len)
    df["word_count"] = df["text"].astype(str).apply(lambda t: len(t.split()))

    print("\n--- Length Statistics ---")
    print(f"Average Character Length: {df['char_count'].mean():.2f} (Min: {df['char_count'].min()}, Max: {df['char_count'].max()})")
    print(f"Average Word Count:       {df['word_count'].mean():.2f} (Min: {df['word_count'].min()}, Max: {df['word_count'].max()})")

    # Generate EDA plots
    generate_eda_visualizations(df, results_dir)
    print("\n" + "=" * 60)
    print(f"Inspection complete. Visualizations saved in: {results_dir.resolve()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
