# Troubleshooting Guide: IndicBERT Classifier

This guide addresses common setup, runtime, and environmental issues encountered when running the project on Windows.

---

## 1. Windows PowerShell Virtual Environment Activation Error

### Problem:
When running `.\.venv\Scripts\Activate.ps1`, PowerShell displays an error:
```
File C:\...\Activate.ps1 cannot be loaded because running scripts is disabled on this system.
```

### Solution:
Windows PowerShell blocks script execution by default. Run the following command in PowerShell:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then retry activating the virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```
Alternatively, you can activate the virtual environment using Command Prompt (cmd.exe):
```cmd
.venv\Scripts\activate.bat
```

---

## 2. Hugging Face 401 Client Error / Gated Repository

### Problem:
Loading `ai4bharat/indic-bert` fails with:
```
huggingface_hub.errors.GatedRepoError: 401 Client Error. Cannot access gated repo... Access to model is restricted.
```

### Cause:
The original ALBERT-based checkpoint `ai4bharat/indic-bert` was made a gated repository requiring user agreement and Hugging Face authentication.

### Solution:
The project is configured by default to use `ai4bharat/IndicBERTv2-MLM-only`, which is open, ungated, and does not require authentication.
If you specifically wish to use a gated checkpoint:
1. Visit https://huggingface.co/ai4bharat/indic-bert and click "Agree and access repository".
2. In your terminal, run:
   ```powershell
   huggingface-cli login
   ```
3. Paste your Hugging Face User Access Token (read permissions).

---

## 3. CPU Out-Of-Memory (OOM) Error During Training

### Problem:
```
RuntimeError: [enforce fail at alloc_cpu.cpp] data. DefaultCPUAllocator: not enough memory
```

### Cause:
IndicBERT v2 contains 278,042,882 parameters. Full backpropagation with AdamW optimizer states requires ~4.5 GB of contiguous RAM, which can exceed free memory on 8 GB or 16 GB consumer laptops.

### Solution:
Use the transfer learning feature freezing option in `config.yaml`:
```yaml
training:
  freeze_base_model: true
  trainable_layers: "head"
```
Or run the training script with:
```powershell
python scripts/train_model.py --freeze-base
```
This freezes the base transformer encoder and trains only the classification head (~1,538 parameters), reducing memory consumption by over 75% and accelerating training without OOM crashes.

---

## 4. Insufficient Disk Space (OS Error 112)

### Problem:
```
safetensors._safetensors_rust.SafetensorError: Error while serializing: There is not enough space on the disk. (os error 112)
```

### Cause:
Hugging Face caches model files in `~/.cache/huggingface/hub`. A single checkpoint plus temporary shards requires ~2 GB of free space during saving.

### Solution:
1. Clean incomplete download blobs from Hugging Face cache:
   ```powershell
   python -c "import shutil, os; p = os.path.expanduser('~/.cache/huggingface/hub'); [shutil.rmtree(os.path.join(p, d)) for d in os.listdir(p) if 'incomplete' in d]"
   ```
2. Save using direct binary serialization (`safe_serialization=False`) as configured in `src/model.py`.
3. If primary drive C: is full, move the Hugging Face cache to another drive (e.g. D:) by setting the environment variable:
   ```powershell
   $env:HF_HOME = "D:\huggingface_cache"
   ```

---

## 5. Unicode / Character Display Issues on Windows Console

### Problem:
Tamil, Hindi, or Malayalam characters print as question marks (`????`) or garbled symbols in PowerShell or CMD.

### Cause:
Default Windows console code page is often CP1252 or CP437, which lacks glyphs for Indic writing systems.

### Solution:
Set the console encoding to UTF-8 before running Python scripts:
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001
```
Note: In the Streamlit web dashboard (`streamlit run app.py`), all Indic scripts render in native Unicode without console font limitations.

---

## 6. Streamlit Web App Fails to Start

### Problem:
```
streamlit : The term 'streamlit' is not recognized...
```

### Solution:
Ensure you are using the virtual environment where dependencies were installed:
```powershell
python -m streamlit run app.py
```
If port 8501 is already in use by another application:
```powershell
python -m streamlit run app.py --server.port 8502
```
