"""
IndicBERT: Multilingual Indic Text Classifier
Low-Resource Indian-Language Sentiment Analysis Using Transfer Learning
Interactive Streamlit Application for Demonstration and Viva Evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import seaborn as sns
import streamlit as st

# Configure Streamlit page layout and branding
st.set_page_config(
    page_title="IndicBERT Classifier | Multilingual Sentiment",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Project path resolution
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.history import (
    clear_prediction_history,
    export_history_csv,
    filter_history,
    load_prediction_history,
)
from src.model import check_fine_tuned_model_exists
from src.predict import (
    ModelNotAvailableError,
    get_cached_model,
    predict_batch,
    predict_sentiment,
)
from src.preprocessing import (
    SCHEDULED_INDIC_LANGUAGES,
    clean_indic_text,
    detect_indic_script,
)

# Load project configuration
try:
    CONFIG = load_config()
except Exception as e:
    st.error(f"Failed to load project configuration: {e}")
    st.stop()

# Custom CSS for polished typography and badges
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .badge-blue { background-color: #DBEAFE; color: #1E40AF; }
    .badge-green { background-color: #DCFCE7; color: #166534; }
    .badge-purple { background-color: #F3E8FF; color: #6B21A8; }
    .badge-orange { background-color: #FFEDD5; color: #9A3412; }
    .sentiment-positive {
        color: #15803D;
        background-color: #DCFCE7;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .sentiment-negative {
        color: #B91C1C;
        background-color: #FEE2E2;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
st.sidebar.image(
    "https://raw.githubusercontent.com/AI4Bharat/IndicBERT/master/docs/ai4bharat_logo.png",
    use_container_width=True,
    caption="IndicBERT Transfer Learning",
)
st.sidebar.title("Navigation")
menu_choice = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Home",
        "🔍 Single Text Prediction",
        "📁 Batch Prediction (CSV)",
        "📜 Prediction History",
        "📊 Model Evaluation Dashboard",
        "🗃️ Dataset Explorer",
        "📖 About & Academic References",
    ],
)

# Sidebar Status Indicator
st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
model_exists = check_fine_tuned_model_exists(CONFIG.local_model_dir)
if model_exists:
    st.sidebar.success("✅ Fine-Tuned Model: Ready")
else:
    st.sidebar.warning("⚠️ Fine-Tuned Model: Not Found")
    st.sidebar.caption("Run `python scripts/train_model.py` to train.")


# ==============================================================================
# SECTION A: HOME
# ==============================================================================
if menu_choice == "🏠 Home":
    st.markdown("<div class='main-header'>IndicBERT: Multilingual Indic Text Classifier</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Low-Resource Indian-Language Sentiment Analysis Using Transfer Learning</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <span class='badge badge-blue'>Model: IndicBERT v2 (BERT-based)</span>
        <span class='badge badge-green'>Dataset: AI4Bharat IndicSentiment (CC-BY 4.0)</span>
        <span class='badge badge-purple'>Languages: Tamil, Hindi, Malayalam</span>
        <span class='badge badge-orange'>Storage: Local CSV Persistence</span>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 🎯 Problem Statement & Motivation")
        st.write(
            """
            India is home to over 1.4 billion people speaking hundreds of languages, yet most modern NLP tools 
            are overwhelmingly focused on English. Native Indian languages—including Dravidian (Tamil, Malayalam) 
            and Indo-Aryan (Hindi) families—frequently suffer from a severe scarcity of high-quality labelled datasets 
            for downstream tasks like sentiment classification.
            
            This project investigates **Low-Resource Multilingual Transfer Learning**. By fine-tuning **IndicBERT**, 
            a state-of-the-art multilingual model pretrained on millions of Indian-language tokens, we transfer rich cross-lingual 
            representations to sentiment analysis with high sample efficiency.
            """
        )

        st.markdown("### 🔬 Academic Objectives")
        st.write(
            """
            1. **Cross-Lingual Representation**: Exploit IndicBERT shared multilingual embeddings across distinct scripts.
            2. **Low-Resource Data Efficiency**: Empirically evaluate performance across training fractions (10%, 25%, 50%, 100%).
            3. **Native Script Preservation**: Clean and tokenize Tamil, Devanagari, and Malayalam without ASCII degradation.
            4. **Transparent Local Persistence**: Maintain complete prediction records locally using CSV without external database engines.
            5. **Interactive Viva Demonstration**: Provide an intuitive web application for inference, evaluation, and batch processing.
            """
        )

    with col2:
        st.markdown("### 🌐 Supported Languages & Scripts")
        st.info(
            """
            - **Tamil (தமிழ்)**: Dravidian family, agglutinative grammar.
            - **Hindi (हिन्दी)**: Indo-Aryan family, Devanagari script.
            - **Malayalam (മലയാളം)**: Dravidian family, complex conjuncts.
            
            *Classification Target*: **Positive** vs. **Negative**
            """
        )

        st.markdown("### 💻 Technology Stack")
        st.markdown(
            """
            - **Language Modeling**: PyTorch, Hugging Face Transformers (`ai4bharat/IndicBERTv2-MLM-only`)
            - **Preprocessing & EDA**: Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib
            - **Web Deployment**: Streamlit Dashboard
            - **Storage**: UTF-8 Local CSV files
            """
        )

    st.markdown("---")
    st.markdown("### 🚀 Quick Demonstration Guide for Students & Examiners")
    st.write(
        """
        - Navigate to **🔍 Single Text Prediction** to type or click pre-loaded Tamil, Hindi, or Malayalam sample reviews.
        - Open **📁 Batch Prediction** to upload a test CSV file and classify hundreds of sentences at once.
        - Check **📜 Prediction History** to inspect local CSV persistence, filter by sentiment or language, and download history.
        - Visit **📊 Model Evaluation Dashboard** to inspect real test-set accuracy, confusion matrix, and low-resource data efficiency curves.
        """
    )


# ==============================================================================
# SECTION B: SINGLE TEXT SENTIMENT PREDICTION
# ==============================================================================
elif menu_choice == "🔍 Single Text Prediction":
    st.markdown("## 🔍 Single Text Sentiment Prediction")
    st.markdown("Type a sentence in Tamil, Hindi, or Malayalam to classify its emotional polarity.")
    st.markdown("---")

    if not model_exists:
        st.warning(
            "⚠️ No fine-tuned model checkpoint was found on disk.\n\n"
            "To train the model on your laptop, run this command in your PowerShell terminal:\n\n"
            "```powershell\n"
            "python scripts/train_model.py\n"
            "```\n"
            "Or run a fast smoke test:\n"
            "```powershell\n"
            "python scripts/train_model.py --smoke-test\n"
            "```"
        )

    # Sample sentences for quick viva demonstration
    st.markdown("##### 💡 Viva Demo: Click a pre-set sample to test:")
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    
    preset_text = ""
    if sample_col1.button("🇮🇳 Tamil (+): இந்த படம் மிகவும் அருமை!"):
        st.session_state["input_text_area"] = "இந்த படம் மிகவும் அருமையாக உள்ளது, நடிகர்களின் நடிப்பு பிரமாதம்!"
        st.session_state["selected_lang"] = "Tamil"

    if sample_col1.button("🇮🇳 Tamil (-): மிகவும் மோசமான அனுபவம்"):
        st.session_state["input_text_area"] = "மிகவும் மோசமான படம், திரைக்கதை சலிப்பை தருகிறது, நேர விரயம்."
        st.session_state["selected_lang"] = "Tamil"

    if sample_col2.button("🇮🇳 Hindi (+): यह फिल्म बहुत शानदार है"):
        st.session_state["input_text_area"] = "यह फिल्म बहुत शानदार और मनोरंजक है, संगीत दिल को छू लेने वाला है।"
        st.session_state["selected_lang"] = "Hindi"

    if sample_col2.button("🇮🇳 Hindi (-): बिल्कुल घटिया और निराशाजनक"):
        st.session_state["input_text_area"] = "बिल्कुल घटिया और निराशाजनक, कहानी में कोई दम नहीं था, पैसे की बर्बादी।"
        st.session_state["selected_lang"] = "Hindi"

    if sample_col3.button("🇮🇳 Malayalam (+): വളരെ നല്ല സിനിമ"):
        st.session_state["input_text_area"] = "ഈ സിനിമ വളരെ മനോഹരവും ഹൃദയസ്പർശിയുമാണ്, മികച്ച സംവിധാനം!"
        st.session_state["selected_lang"] = "Malayalam"

    if sample_col3.button("🇮🇳 Malayalam (-): വളരെ മോശം അനുഭവം"):
        st.session_state["input_text_area"] = "വളരെ മോശം സിനിമ, വെറുതെ സമയം കളഞ്ഞു, ഒരു പുതുമയും ഇല്ല."
        st.session_state["selected_lang"] = "Malayalam"

    # Input controls
    default_text = st.session_state.get("input_text_area", "")
    default_lang = st.session_state.get("selected_lang", "Auto-Detect")

    user_text = st.text_area(
        "Enter text in an Indic language:",
        value=default_text,
        height=120,
        placeholder="Type or paste review text here in native Indic script...",
    )

    col_lang, col_btn = st.columns([2, 1])
    with col_lang:
        lang_options = ["Auto-Detect", *SCHEDULED_INDIC_LANGUAGES]
        lang_index = lang_options.index(default_lang) if default_lang in lang_options else 0
        selected_language = st.selectbox("Language Identification / Specification:", lang_options, index=lang_index)
        st.caption(
            "Auto-detect recognizes the writing script. Several Indic languages share scripts "
            "(for example, Hindi, Marathi, and Nepali use Devanagari; Bengali and Assamese share "
            "a script), so specify the language here when it cannot be distinguished from script alone. "
            "Language choices change the language label, not the sentiment model's Tamil, Hindi, "
            "and Malayalam training coverage."
        )

    with col_btn:
        st.write("")
        st.write("")
        predict_button = st.button("🚀 Predict Sentiment", type="primary", use_container_width=True)

    if predict_button:
        cleaned_text = clean_indic_text(user_text)
        if not cleaned_text:
            st.error("❌ Please enter some text before clicking Predict.")
        elif not model_exists:
            st.error("❌ Model not trained yet! Please run `python scripts/train_model.py` first.")
        else:
            with st.spinner("Analyzing text using fine-tuned IndicBERT..."):
                try:
                    result = predict_sentiment(
                        text=cleaned_text,
                        language=selected_language,
                        config=CONFIG,
                        log_to_history=True,
                    )

                    st.markdown("---")
                    st.markdown("### 📊 Prediction Results")

                    res_col1, res_col2 = st.columns([1, 1])

                    with res_col1:
                        sentiment = result["predicted_sentiment"]
                        conf = result["confidence"] * 100
                        lang = result["language"]

                        if sentiment == "positive":
                            st.markdown(
                                f"<div class='sentiment-positive'>😊 Predicted Sentiment: POSITIVE</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"<div class='sentiment-negative'>😞 Predicted Sentiment: NEGATIVE</div>",
                                unsafe_allow_html=True,
                            )

                        st.write("")
                        st.metric("Model Confidence Score", f"{conf:.2f}%")
                        st.progress(conf / 100)

                        st.info(
                            f"**Detected / Specified Language**: {lang}\n\n"
                            f"**Model Identifier**: {result['model_name']}\n\n"
                            f"**Persistence**: Automatically logged to local CSV (`data/prediction_history.csv`)"
                        )

                    with res_col2:
                        st.markdown("#### Probability Distribution")
                        probs = result["probabilities"]
                        prob_df = pd.DataFrame(
                            list(probs.items()),
                            columns=["Sentiment", "Probability"],
                        )
                        fig, ax = plt.subplots(figsize=(6, 3.5))
                        colors = ["#EF4444" if s == "negative" else "#22C55E" for s in prob_df["Sentiment"]]
                        sns.barplot(
                            data=prob_df,
                            x="Probability",
                            y="Sentiment",
                            palette=colors,
                            ax=ax,
                        )
                        ax.set_xlim([0, 1.0])
                        ax.set_xlabel("Probability", fontsize=10)
                        for p in ax.patches:
                            width = p.get_width()
                            ax.annotate(
                                f"{width*100:.1f}%",
                                (width, p.get_y() + p.get_height() / 2.0),
                                xytext=(5, 0),
                                textcoords="offset points",
                                va="center",
                                weight="bold",
                            )
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                except ModelNotAvailableError as e:
                    st.error(f"Model Error: {e}")
                except Exception as e:
                    st.error(f"Inference Exception: {e}")


# ==============================================================================
# SECTION C: BATCH PREDICTION
# ==============================================================================
elif menu_choice == "📁 Batch Prediction (CSV)":
    st.markdown("## 📁 Batch Sentiment Prediction")
    st.markdown("Upload a CSV file containing multiple Indian-language sentences to classify them in bulk.")
    st.markdown("---")

    if not model_exists:
        st.warning("⚠️ No fine-tuned model checkpoint found. Train the model first via `python scripts/train_model.py`.")

    st.markdown("##### 📋 Expected CSV Format")
    st.write("Your CSV must contain at least a `text` column. An optional `language` column is also supported.")

    sample_batch_csv = "text,language\n\"இந்த படம் மிகவும் நன்றாக உள்ளது\",Tamil\n\"यह फिल्म बहुत खराब है\",Hindi\n\"ഈ സിനിമ വളരെ നല്ലതാണ്\",Malayalam\n"
    st.download_button(
        "📥 Download Sample Batch Template (CSV)",
        data=sample_batch_csv.encode("utf-8"),
        file_name="sample_indic_batch_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload CSV file for batch classification:", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            df_upload = pd.read_csv(uploaded_file, encoding="utf-8-sig")

        st.markdown("##### 📄 Uploaded Data Preview")
        st.dataframe(df_upload.head(10), use_container_width=True)

        if "text" not in df_upload.columns:
            st.error("❌ The uploaded CSV must contain a 'text' column. Found: " + str(list(df_upload.columns)))
        elif not model_exists:
            st.error("❌ Model checkpoint missing. Cannot classify batch.")
        else:
            col_b1, col_b2 = st.columns([1, 2])
            with col_b1:
                log_batch_history = st.checkbox("Log batch predictions to local history CSV", value=False)
            with col_b2:
                batch_run_btn = st.button("⚡ Classify Entire File", type="primary", use_container_width=True)

            if batch_run_btn:
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                status_text.text("Processing batch through IndicBERT...")

                start_t = time.time()
                try:
                    result_df = predict_batch(
                        df_upload,
                        text_column="text",
                        language_column="language" if "language" in df_upload.columns else None,
                        batch_size=16,
                        config=CONFIG,
                        log_to_history=log_batch_history,
                    )
                    progress_bar.progress(1.0)
                    elapsed = time.time() - start_t
                    status_text.success(f"✅ Classified {len(result_df)} rows in {elapsed:.2f} seconds!")

                    st.markdown("##### 🎯 Classification Results")
                    st.dataframe(result_df, use_container_width=True)

                    # Summary counts
                    st.markdown("##### 📈 Summary of Predictions")
                    sum_col1, sum_col2, sum_col3 = st.columns(3)
                    pos_count = int((result_df["predicted_sentiment"] == "positive").sum())
                    neg_count = int((result_df["predicted_sentiment"] == "negative").sum())
                    avg_conf = float(result_df["confidence"].mean() * 100)

                    sum_col1.metric("Positive Count", pos_count)
                    sum_col2.metric("Negative Count", neg_count)
                    sum_col3.metric("Average Confidence", f"{avg_conf:.1f}%")

                    # Download button for result CSV
                    csv_export = result_df.to_csv(index=False, encoding="utf-8").encode("utf-8")
                    st.download_button(
                        "📥 Download Completed Predictions (CSV)",
                        data=csv_export,
                        file_name="indicbert_predictions_output.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Error during batch classification: {e}")


# ==============================================================================
# SECTION D: PREDICTION HISTORY
# ==============================================================================
elif menu_choice == "📜 Prediction History":
    st.markdown("## 📜 Prediction History")
    st.markdown(
        "All single and batch predictions are stored locally in **`data/prediction_history.csv`**. "
        "No external database is required."
    )
    st.markdown("---")

    hist_df = load_prediction_history(CONFIG.history_file)

    if hist_df.empty:
        st.info("ℹ️ No prediction history recorded yet. Use the prediction page to classify text!")
    else:
        # Search & Filters
        f_col1, f_col2, f_col3 = st.columns([2, 1, 1])

        with f_col1:
            search_query = st.text_input("🔍 Search Text:", placeholder="Search keywords...")

        with f_col2:
            languages = ["All"] + sorted([str(l) for l in hist_df["language"].dropna().unique() if str(l).strip()])
            selected_filter_lang = st.selectbox("Filter by Language:", languages)

        with f_col3:
            sentiments = ["All"] + sorted([str(s) for s in hist_df["predicted_sentiment"].dropna().unique() if str(s).strip()])
            selected_filter_sent = st.selectbox("Filter by Sentiment:", sentiments)

        filtered_df = filter_history(
            hist_df,
            search_query=search_query,
            language=selected_filter_lang,
            sentiment=selected_filter_sent,
        )

        st.markdown(f"**Showing {len(filtered_df)} of {len(hist_df)} recorded inferences** (Newest first)")
        st.dataframe(filtered_df, use_container_width=True)

        exp_col1, exp_col2 = st.columns([1, 1])

        with exp_col1:
            export_bytes = export_history_csv(filtered_df)
            st.download_button(
                "📥 Export Filtered History (CSV)",
                data=export_bytes,
                file_name="prediction_history_export.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with exp_col2:
            with st.expander("⚠️ Clear History Options"):
                confirm_clear = st.checkbox("I confirm I want to clear all history records")
                if st.button("🗑️ Clear Prediction History", type="secondary"):
                    if confirm_clear:
                        if clear_prediction_history(CONFIG.history_file):
                            st.success("History successfully cleared!")
                            st.rerun()
                        else:
                            st.error("Failed to clear history.")
                    else:
                        st.warning("Please check the confirmation box first.")


# ==============================================================================
# SECTION E: MODEL EVALUATION DASHBOARD
# ==============================================================================
elif menu_choice == "📊 Model Evaluation Dashboard":
    st.markdown("## 📊 Model Evaluation Dashboard")
    st.markdown("Empirical test results and performance metrics computed on the held-out test set.")
    st.markdown("---")

    metrics_file = CONFIG.metrics_file
    report_file = CONFIG.report_file
    cm_plot_file = CONFIG.confusion_matrix_plot
    per_lang_file = CONFIG.per_language_metrics_file
    exp_csv_file = CONFIG.experiment_results_csv
    exp_plot_file = CONFIG.experiment_plot_path
    train_history_file = CONFIG.training_history_file

    if not metrics_file.exists():
        st.warning(
            "⚠️ Evaluation metrics have not been generated yet.\n\n"
            "To evaluate your fine-tuned model and generate reports, run:\n\n"
            "```powershell\n"
            "python scripts/evaluate_model.py\n"
            "```"
        )
    else:
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        # Global Metric Cards
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Test Accuracy", f"{metrics_data.get('accuracy', 0)*100:.2f}%")
        m_col2.metric("Macro F1-Score", f"{metrics_data.get('macro_f1', 0):.4f}")
        m_col3.metric("Macro Precision", f"{metrics_data.get('macro_precision', 0):.4f}")
        m_col4.metric("Macro Recall", f"{metrics_data.get('macro_recall', 0):.4f}")
        m_col5.metric("Test Samples", metrics_data.get("sample_count", 0))

        st.markdown("---")

        row1_col1, row1_col2 = st.columns([1, 1])

        with row1_col1:
            st.markdown("#### 🎯 Confusion Matrix (Held-Out Test Set)")
            if cm_plot_file.exists():
                st.image(str(cm_plot_file), use_container_width=True)
            else:
                st.info("Confusion matrix plot not found.")

        with row1_col2:
            st.markdown("#### 🌐 Per-Language Performance Breakdown")
            if per_lang_file.exists():
                lang_perf_df = pd.read_csv(per_lang_file, encoding="utf-8")
                st.dataframe(lang_perf_df, use_container_width=True)

                fig, ax = plt.subplots(figsize=(6, 3.8))
                sns.barplot(data=lang_perf_df, x="language", y="macro_f1", palette="Blues_r", ax=ax)
                ax.set_ylim([0, 1.0])
                ax.set_ylabel("Macro F1-Score")
                ax.set_title("Macro F1 by Language", fontsize=11, weight="bold")
                for p in ax.patches:
                    ax.annotate(
                        f"{p.get_height():.3f}",
                        (p.get_x() + p.get_width() / 2.0, p.get_height()),
                        ha="center", va="center", xytext=(0, 5), textcoords="offset points",
                        fontweight="bold"
                    )
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Per-language metrics CSV not found.")

        st.markdown("---")

        # Low-Resource Learning Efficiency
        st.markdown("### 📈 Low-Resource Data Efficiency Experiment")
        st.write(
            """
            This key academic experiment measures model performance as the volume of available training 
            data varies (e.g. 10%, 25%, 50%, 100%). It demonstrates how pre-training on multilingual Indic data 
            enables the classifier to generalize effectively even with scarce labelled samples.
            """
        )

        if exp_csv_file.exists():
            exp_df = pd.read_csv(exp_csv_file, encoding="utf-8")
            st.dataframe(exp_df, use_container_width=True)
            if exp_plot_file.exists():
                st.image(str(exp_plot_file), use_container_width=True)
        else:
            st.info(
                "ℹ️ Low-resource experiment comparison not generated yet.\n\n"
                "To execute this research experiment, run:\n\n"
                "```powershell\n"
                "python scripts/run_low_resource_experiment.py --mode quick\n"
                "```"
            )

        st.markdown("---")
        st.markdown("### 📖 Metric Interpretation Guide for Viva")
        with st.expander("Click to view explanations of metrics"):
            st.write(
                """
                - **Accuracy**: Overall fraction of correct predictions across all classes. Can be misleading if classes are imbalanced.
                - **Precision**: Proportion of positive identifications that were actually correct: $TP / (TP + FP)$.
                - **Recall**: Proportion of actual positives identified correctly: $TP / (TP + FN)$.
                - **Macro F1-Score**: The unweighted harmonic mean of precision and recall calculated independently for each class and then averaged. Essential in multilingual low-resource tasks because it treats both minority and majority classes equally.
                """
            )


# ==============================================================================
# SECTION F: DATASET EXPLORER
# ==============================================================================
elif menu_choice == "🗃️ Dataset Explorer":
    st.markdown("## 🗃️ Indic Multilingual Dataset Explorer")
    st.markdown("Inspect the distribution, schema, and statistics of the training and evaluation corpus.")
    st.markdown("---")

    summary_file = CONFIG.dataset_metadata_file

    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Usable Samples", summary_data.get("total_samples", 0))
        c2.metric("Training Samples (70%)", summary_data.get("train_samples", 0))
        c3.metric("Validation Samples (15%)", summary_data.get("val_samples", 0))
        c4.metric("Test Samples (15%)", summary_data.get("test_samples", 0))

        st.markdown("---")

        eda_col1, eda_col2, eda_col3 = st.columns(3)
        eda_lang = CONFIG.results_dir / "eda_language_dist.png"
        eda_label = CONFIG.results_dir / "eda_label_dist.png"
        eda_len = CONFIG.results_dir / "eda_text_length.png"

        with eda_col1:
            if eda_lang.exists():
                st.image(str(eda_lang), caption="Language Distribution", use_container_width=True)
        with eda_col2:
            if eda_label.exists():
                st.image(str(eda_label), caption="Sentiment Balance", use_container_width=True)
        with eda_col3:
            if eda_len.exists():
                st.image(str(eda_len), caption="Word Length Distribution", use_container_width=True)

        st.markdown("---")
        st.markdown("### 📋 Preview Dataset Splits")
        split_tab1, split_tab2, split_tab3 = st.tabs(["Train Set", "Validation Set", "Test Set"])

        with split_tab1:
            if CONFIG.train_file.exists():
                tr_df = pd.read_csv(CONFIG.train_file, encoding="utf-8")
                st.dataframe(tr_df.head(20), use_container_width=True)

        with split_tab2:
            if CONFIG.val_file.exists():
                va_df = pd.read_csv(CONFIG.val_file, encoding="utf-8")
                st.dataframe(va_df.head(20), use_container_width=True)

        with split_tab3:
            if CONFIG.test_file.exists():
                te_df = pd.read_csv(CONFIG.test_file, encoding="utf-8")
                st.dataframe(te_df.head(20), use_container_width=True)

        st.markdown("---")
        st.markdown("### 📜 Dataset Provenance & Attribution")
        st.info(
            """
            - **Benchmark Source**: AI4Bharat IndicSentiment / IndicGLUE Benchmark
            - **Authors**: AI4Bharat Consortium & IIT Madras
            - **License**: Creative Commons Attribution 4.0 International (CC-BY 4.0)
            - **Languages Included**: Tamil (ta), Hindi (hi), Malayalam (ml)
            - **Annotation**: Human-annotated real product and media reviews in native scripts.
            """
        )
    else:
        st.warning(
            "⚠️ Processed dataset not found.\n\n"
            "Run `python scripts/prepare_dataset.py` to download and compile the dataset."
        )


# ==============================================================================
# SECTION G: ABOUT & ACADEMIC REFERENCES
# ==============================================================================
elif menu_choice == "📖 About & Academic References":
    st.markdown("## 📖 About Project & Academic References")
    st.markdown("Documentation, methodology, model architecture, and citations for viva defense.")
    st.markdown("---")

    st.markdown("### 🏛️ IndicBERT Architecture Overview")
    st.write(
        """
        **IndicBERT** is a state-of-the-art multilingual model trained specifically on 12–23 major Indian languages. 
        Developed by AI4Bharat and researchers from IIT Madras, it utilizes a shared vocabulary designed to handle 
        the phonetic and morphological characteristics of Indic scripts.
        
        In transfer learning, the pretrained multilingual representations are frozen or adapted, and a task-specific 
        sequence classification head (linear projection with softmax) is trained on labeled sentiment data.
        """
    )

    st.markdown("### 📚 Key Academic Citations")
    st.markdown(
        """
        1. **IndicBERT & IndicGLUE**:
           - Kakwani et al. (2020). *IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages*. Findings of EMNLP 2020.
           - URL: https://aclanthology.org/2020.findings-emnlp.445/
        2. **IndicBERT v2**:
           - Doddapaneni et al. (2023). *Towards Leaving No Indic Language Behind: Building Monolingual and Multilingual Language Models for 23 Indian Languages*. ACL 2023.
        3. **BERT & Sequence Classification**:
           - Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT 2019.
        """
    )

    st.markdown("### 🛠️ Step-by-Step PowerShell Commands for Demonstration")
    st.code(
        """
# 1. Activate Virtual Environment
.\\.venv\\Scripts\\Activate.ps1

# 2. Prepare & Stratify Dataset
python scripts/prepare_dataset.py

# 3. Inspect Dataset & Generate EDA Charts
python scripts/inspect_dataset.py

# 4. Fine-Tune IndicBERT Model
python scripts/train_model.py

# 5. Evaluate on Held-Out Test Set
python scripts/evaluate_model.py

# 6. Run Low-Resource Data Efficiency Experiment
python scripts/run_low_resource_experiment.py --mode quick

# 7. Launch Streamlit Application
streamlit run app.py
        """,
        language="powershell",
    )
