# Odia ASR — Fine-tuning IndicConformer

Fine-tune [ai4bharat/indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) on Odia speech recognition using the OpenSLR-103 (MUCS 2021) dataset, with evaluation support for IWSLT 2026.

---

## Project Structure

```
odia_asr/
├── download_data.py        # Download & extract OpenSLR-103 dataset
├── prepare_dataset.py      # Build NeMo-style JSON manifests
├── train.py                # Fine-tune IndicConformer (CTC head)
├── inference.py            # Transcribe audio with base or fine-tuned model
├── evaluate_iwslt2026.py   # Evaluate on IWSLT 2026 Odia data (WER/CER/BLEU)
├── requirements.txt        # Python dependencies
├── IndicConformer_Odia_Finetune.ipynb  # Interactive notebook walkthrough
└── data/
    ├── raw/                # Raw downloaded audio + transcriptions
    └── manifests/          # Generated JSON manifests (created by prepare_dataset.py)
```

---

## Requirements

- Python 3.9+
- macOS / Linux (GPU recommended for training)

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Dependencies:**

| Package | Version |
|---|---|
| torch | ≥ 2.1.0 |
| torchaudio | ≥ 2.1.0 |
| transformers | ≥ 4.49.0 |
| huggingface_hub | ≥ 0.21.0 |
| onnxruntime | ≥ 1.19.0 |
| onnx | ≥ 1.16.0 |
| numpy | latest |
| gdown | ≥ 5.1.0 (for Google Drive downloads) |

---

## HuggingFace Token

If the base model is gated, set your HuggingFace token as an environment variable:

```bash
export HF_TOKEN=your_token_here
```

> **Never paste tokens in chat, code, or commit them to version control.**  
> Manage tokens at: https://huggingface.co/settings/tokens

---

## Step-by-Step Usage

### 1. Download the Dataset

Three options are supported:

#### Option A — OpenSLR (default)
Downloads directly from openslr.org with EU mirror fallback:

```bash
python download_data.py
```

#### Option B — Google Drive (download via file ID)

1. Upload `Odia_train.tar.gz` and `Odia_test.tar.gz` to your [Google Drive](https://drive.google.com/drive/my-drive).
2. For each file: right-click → **Share** → **Copy link**. The file ID is the long string between `/d/` and `/view` in the URL:
   ```
   https://drive.google.com/file/d/<FILE_ID>/view
   ```
3. Make sure sharing is set to **"Anyone with the link"**.
4. Run:
   ```bash
   python download_data.py --gdrive \
       --train-id <TRAIN_FILE_ID> \
       --test-id  <TEST_FILE_ID>
   ```

#### Option C — Mounted Google Drive (Google Drive for Desktop / Colab)

If Google Drive for Desktop is installed on macOS, your drive is already mounted. Point `--gdrive-mount` to the folder containing `Odia_train/` and `Odia_test/`:

```bash
# macOS — Google Drive for Desktop
python download_data.py \
    --gdrive-mount "/Volumes/GoogleDrive/My Drive/odia_asr_data"

# Google Colab
python download_data.py \
    --gdrive-mount "/content/drive/MyDrive/odia_asr_data"
```

This creates symlinks in `data/raw/` — no copying or re-downloading needed.

---

All options populate:
```
data/raw/Odia_train/
    audio/            # .wav files @ 8 kHz
    transcription.txt
data/raw/Odia_test/
    audio/
    transcription.txt
```

### 2. Prepare Manifests

Builds NeMo-style JSONL manifests with audio path, duration, and transcript:

```bash
python prepare_dataset.py
```

Outputs:
- `data/manifests/odia_train_manifest.json`
- `data/manifests/odia_test_manifest.json`

### 3. Train

Fine-tunes the CTC head on IndicConformer. The encoder is frozen for the first 5 epochs, then unfrozen for joint training:

```bash
python train.py
```

Best checkpoint (by WER on test set) is saved to `outputs/odia_finetuned/best_model.pt`.

**Key training hyperparameters:**

| Parameter | Default |
|---|---|
| Base model | `ai4bharat/indic-conformer-600m-multilingual` |
| Language | Odia (`or`) |
| Batch size | 8 (effective 32 with grad accumulation × 4) |
| Learning rate | 1e-4 |
| Warmup steps | 500 |
| Max epochs | 30 |
| Encoder frozen for first | 5 epochs |
| Audio range | 0.5 – 30.0 seconds |
| Input sample rate | 8 kHz → resampled to 16 kHz |
| Output dir | `outputs/odia_finetuned/` |

### 4. Inference

Transcribe a single audio file using the base or fine-tuned model:

```bash
# Base model (no fine-tuning):
python inference.py --audio path/to/audio.wav

# Fine-tuned model:
python inference.py --audio path/to/audio.wav --checkpoint outputs/odia_finetuned/best_model.pt
```

### 5. Evaluate on IWSLT 2026

Evaluate WER, CER, and BLEU on the IWSLT 2026 Odia dataset (`shashwatup9k/iwslt2026_or-hi-eng`):

```bash
# Fine-tuned model on dev split:
python evaluate_iwslt2026.py \
    --data_root /path/to/iwslt2026_or-hi-eng \
    --split dev \
    --checkpoint outputs/odia_finetuned/best_model.pt

# Baseline (no fine-tuning):
python evaluate_iwslt2026.py \
    --data_root /path/to/iwslt2026_or-hi-eng \
    --split dev \
    --baseline \
    --output results.json
```

**Metrics computed:**
- **WER** — Word Error Rate (ASR transcription quality)
- **CER** — Character Error Rate (more stable for Odia script)
- **BLEU** — Translation quality (when reference translations are available)

---

## Notebook

An interactive walkthrough is available in [IndicConformer_Odia_Finetune.ipynb](IndicConformer_Odia_Finetune.ipynb).

---

## Model Details

- **Base model:** [ai4bharat/indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
- **Architecture:** Conformer encoder + CTC decoder
- **Vocabulary:** Odia Unicode character set (vowels, consonants, dependent vowel signs, digits, punctuation)
- **Training objective:** CTC loss with greedy decoding

---

## License

Please refer to the [OpenSLR-103 dataset license](https://www.openslr.org/103/) and the [AI4Bharat model license](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) before use.
