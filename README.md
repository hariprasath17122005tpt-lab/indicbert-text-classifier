# Low-Resource Indic Text Classifier

Indic-language sentiment classification with transfer learning from
[`ai4bharat/IndicBERTv2-MLM-only`](https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only).
The project includes a Streamlit dashboard, dataset preparation and training
scripts, test/evaluation code, and CSV-based prediction history.

> **Project status:** The source code, configuration, datasets, documentation,
> and a local trained checkpoint are present in the working folder. The local
> checkpoint is intentionally excluded from Git because it is about 1.06 GB.
> The dashboard and tests have been verified locally (see
> [Verification status](#verification-status)). No GitHub or Hugging Face
> deployment has been made by this preparation.

## Problem statement

Many Indic languages have fewer labeled examples available for downstream NLP
tasks than high-resource languages. This project explores how a multilingual
pretrained model can be adapted to classify positive and negative sentiment in
Tamil, Hindi, and Malayalam reviews.

## Objectives

- Reuse IndicBERT's pretrained multilingual representations for sentiment
  classification.
- Prepare and stratify a multilingual dataset into train, validation, and test
  splits.
- Compare training with different fractions of the available training data.
- Provide single-text and CSV batch prediction through a Streamlit dashboard.
- Preserve prediction history in a local UTF-8 CSV file without a database.

## Technologies

Python, PyTorch, Hugging Face Transformers, IndicBERT, Pandas, NumPy,
scikit-learn, Streamlit, Matplotlib, Seaborn, PyYAML, Pillow, safetensors,
SentencePiece, and pytest.

## Project structure

```text
app.py                         Streamlit application entry point
config.yaml                    Model, data, training, and output settings
requirements.txt               Python dependencies
src/                           Data, model, training, evaluation, prediction,
                               preprocessing, and history modules
scripts/
  prepare_dataset.py           Prepare and split the dataset
  inspect_dataset.py           Print dataset statistics and generate EDA plots
  train_model.py               Train the classification model
  evaluate_model.py            Evaluate a saved model on the test split
  run_low_resource_experiment.py
                               Compare configured training-data fractions
data/
  raw/                         Source data and small demonstration sample
  processed/                   Train, validation, and test CSV files
  prediction_history.csv       Local prediction history (ignored by Git)
models/                        Local checkpoints (weights ignored by Git)
results/                       Small charts and regeneratable result outputs
tests/                         pytest suite
docs/                          Methodology, architecture, report, and guides
notebooks/                     Exploratory data analysis notebook
.streamlit/config.toml         Streamlit theme and server settings
```

## IndicBERT and transfer learning

The configured pretrained checkpoint is
`ai4bharat/IndicBERTv2-MLM-only`. The code loads it with Hugging Face
Transformers and adds a sequence-classification head. The model implementation
mean-pools the active token representations and passes the result through a
small neural classifier. Training can freeze the transformer backbone for
lower-resource runs or fine-tune the full model with `--full-finetune`.

The selected checkpoint is unchanged. The pretrained base checkpoint may need
to be downloaded from Hugging Face the first time training is run. Training is
not started automatically by the application.

## Dataset preparation

The project uses the AI4Bharat IndicSentiment benchmark for Tamil, Hindi, and
Malayalam. The source CSV is attributed to the
[AI4Bharat IndicSentiment dataset](https://huggingface.co/datasets/ai4bharat/IndicSentiment)
and its stated CC BY 4.0 license. Review the source dataset's attribution and
license terms before redistributing it.

The expected columns are `text`, `language`, and `label`. The configuration
specifies stratified 70% train, 15% validation, and 15% test splits. The
currently present data files were checked locally: 3,462 raw rows and splits
of 2,418 train, 519 validation, and 519 test rows.

From the project root, prepare data with:

```powershell
python scripts/prepare_dataset.py
```

If the configured raw CSV already exists, the script uses it instead of
downloading data. `--force-download` requests a fresh download and overwrites
that configured raw file:

```powershell
python scripts/prepare_dataset.py --force-download
```

To inspect the dataset and generate the EDA charts:

```powershell
python scripts/inspect_dataset.py
```

## Windows installation

Open the project folder in VS Code, then open **Terminal → New Terminal**.
These commands use PowerShell and must be run from the project root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use the virtual-environment interpreter
directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In VS Code, select the interpreter at `.venv\Scripts\python.exe` using
**Python: Select Interpreter**. A CPU-only PyTorch installation is sufficient
for inference and frozen-backbone training; installing a CUDA-enabled build
requires choosing a PyTorch build compatible with the machine's CUDA setup.

## Training

Training can require a substantial model download, storage, and memory. It is
not required to start the dashboard, and this README task did not start
training. The default configuration freezes the base model. To run training
explicitly:

```powershell
python scripts/train_model.py --freeze-base
```

For full fine-tuning on suitable GPU hardware:

```powershell
python scripts/train_model.py --full-finetune
```

For a short smoke test:

```powershell
python scripts/train_model.py --smoke-test
```

These commands save or update model artifacts under the configured local model
directory. Review available RAM, disk space, and the model license before
training.

## Evaluation

After a fine-tuned model is available locally or through the configured Hugging
Face repository, run:

```powershell
python scripts/evaluate_model.py
```

The script evaluates the held-out test CSV and writes metrics, a classification
report, a confusion-matrix chart, and per-language metrics under `results/`.
The CSV, JSON, and text output files are ignored by Git because they are
regeneratable; small charts are retained.

## Low-resource experiment

The quick mode compares the configured 25% and 100% training fractions, using
one epoch per fraction by default:

```powershell
python scripts/run_low_resource_experiment.py --mode quick
```

The full mode uses the configured 10%, 25%, 50%, and 100% fractions and takes
longer:

```powershell
python scripts/run_low_resource_experiment.py --mode full
```

These commands train models and may download the pretrained base model. Run
them only when the required time, memory, and disk space are available.

## Run the Streamlit application

From the project root:

```powershell
python -m streamlit run app.py
```

Streamlit prints a local URL in the terminal. Open that URL in a browser. The
application provides single-text prediction, batch CSV prediction, history,
evaluation visualizations, and dataset exploration.

Predictions require a fine-tuned model. This working folder contains one, but
the model weights and tokenizer JSON are excluded from Git. A fresh GitHub
clone therefore needs either a locally trained/downloaded checkpoint or a
configured Hugging Face Hub model repository. Without either, prediction
reports that a model is unavailable; the app does not invent a prediction.

## Supported languages and limitations

- Tamil
- Hindi
- Malayalam

The application detects Indic scripts and is designed for sentiment labels
`positive` and `negative`. Script detection is not the same as reliable
language identification; mixed-script, transliterated, very short, or
out-of-domain text can be misclassified. Confidence scores are model outputs,
not calibrated probabilities or guarantees. The project is not a general
translation or language-detection service.

## Prediction history

The application appends predictions to `data/prediction_history.csv` in UTF-8.
History can be searched, filtered, and exported from the dashboard. This is a
local file, not a shared database. It may contain text entered by users and is
excluded from Git to avoid publishing private inputs.

## Hugging Face model hosting and deployment

Do not commit the local model weights to GitHub. The current
`models/indicbert_sentiment/model.safetensors` is approximately 1.06 GB; the
tokenizer JSON is approximately 7.39 MB. Host the fine-tuned model and its
tokenizer/configuration files in a Hugging Face model repository instead.

After creating and verifying a model repository, set its actual repository ID
in `config.yaml`:

```yaml
model:
  hf_repo_id: "YOUR_ACCOUNT/YOUR_MODEL_REPOSITORY"
```

The application checks the local model directory first and then uses
`hf_repo_id` when a local fine-tuned checkpoint is not present. Use a public
repository for access without authentication. For a private model repository,
configure `HF_TOKEN` as a secret in the deployment platform; never put the
token in `config.yaml`, `.env` committed to Git, or source code.

For a future Hugging Face Space, use a Docker-based Streamlit setup unless the
Space platform offers a supported Streamlit runtime for your chosen
configuration. The Space needs `app.py`, `src/`, `config.yaml`,
`requirements.txt`, and any small data/config files used by dashboard pages.
For a Docker SDK Space, add and test a Dockerfile that runs
`python -m streamlit run app.py --server.port 7860 --server.address 0.0.0.0`.
Configure `model.hf_repo_id` to point to the model repository and set
`HF_TOKEN` in the Space's secret settings only if the repository is private.
Verify the deployed logs, app startup, and a real prediction before describing
it as deployed. No Space is currently configured or deployed by this project
preparation.

## Tests and verification status

Verified in the current local environment:

- Python 3.13.5 is installed in the existing `.venv`.
- `python -m compileall -q app.py src scripts tests` completed successfully.
- `python -m pytest -q` completed with **39 passed**.
- The YAML configuration loaded, and the raw/train/validation/test CSV files
  were readable with the row counts listed above.
- The local fine-tuned model checkpoint was detected. The prediction tests
  also cover the missing-model error path using a mocked unavailable model.
- The application entry point is `app.py`. The dashboard returned HTTP 200
  at `http://localhost:8501` when started for this verification.
- Evaluation and training were not run as part of this preparation; no new
  accuracy or model-quality result is claimed here.

Re-run the tests from the project root:

```powershell
python -m pytest -q
```

## GitHub preparation

The project folder is not yet a Git repository. `.gitignore` excludes virtual
environments, caches, secrets, local prediction history, model weights, large
tokenizer artifacts, and regeneratable result tables. It keeps source code,
documentation, configuration, the small public dataset files, small model
metadata, and result charts.

Review the candidate files and Git status before staging or committing. Do not
stage, commit, or push until you have reviewed the list. GitHub upload steps
are intentionally left for the project owner; no remote URL or GitHub account
is assumed.

Suggested repository name: `indicbert-text-classifier`

Suggested description: `Low-resource Indic text classification using
IndicBERT transfer learning`

If this local README is present when creating the GitHub repository, create an
**empty** GitHub repository (do not initialize it with another README,
license, or `.gitignore`) to avoid unrelated initial history.

## Troubleshooting

- **`ModuleNotFoundError`**: Confirm VS Code uses `.venv\Scripts\python.exe`
  and install `requirements.txt` in that environment.
- **PowerShell activation is blocked**: Run the `.venv\Scripts\python.exe`
  commands above without activating the environment.
- **No fine-tuned model found**: Train locally or configure a valid Hugging
  Face model repository ID in `config.yaml`.
- **Hugging Face download/authentication error**: Check internet access and
  the repository ID. For a private model, use a platform secret named
  `HF_TOKEN`; do not paste the token into chat or commit it.
- **Out-of-memory during training**: Use `--freeze-base`, reduce the batch
  size in `config.yaml`, or use supported GPU hardware. Full fine-tuning is
  more resource intensive.
- **Dataset download fails**: Use the bundled raw CSV when present, or check
  network access and the dataset's current availability before using
  `--force-download`.
- **History contains sensitive input**: Keep `data/prediction_history.csv`
  local; it is ignored by Git. If a secret was ever committed or pushed,
  removing the file is not enough—revoke/rotate the credential immediately.

## References

- AI4Bharat, [IndicSentiment dataset](https://huggingface.co/datasets/ai4bharat/IndicSentiment).
- AI4Bharat, [IndicBERTv2-MLM-only model](https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only).
- Hugging Face, [Transformers documentation](https://huggingface.co/docs/transformers).
- Streamlit, [documentation](https://docs.streamlit.io/).
- See [`docs/architecture.md`](docs/architecture.md),
  [`docs/methodology.md`](docs/methodology.md), and
  [`docs/project_report.md`](docs/project_report.md) for the project's existing
  technical documentation.
