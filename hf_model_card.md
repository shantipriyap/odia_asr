---
language:
- or
license: cc-by-4.0
tags:
- automatic-speech-recognition
- odia
- conformer
- ctc
- ai4bharat
datasets:
- openslr
- MUCS2021
metrics:
- wer
base_model: ai4bharat/indic-conformer-600m-multilingual
---

# Odia ASR — Fine-tuned IndicConformer 600M

Fine-tuned version of [ai4bharat/indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) on Odia (ଓଡ଼ିଆ) speech recognition.

## Model Details

| Property | Value |
|---|---|
| Base Model | ai4bharat/indic-conformer-600m-multilingual |
| Language | Odia (`or`) |
| Task | Automatic Speech Recognition (ASR) |
| Architecture | Conformer + CTC head |
| Encoder Dim | 1024 |
| Vocabulary | 91 Odia characters |
| Sample Rate | 16 kHz |

## Training Data

- **Train**: [OpenSLR-103](https://www.openslr.org/103/) — MUCS 2021 Odia (~94.5h, 59,782 utterances)
- **Test (OpenSLR)**: OpenSLR-103 test set (~5.5h, 3,471 utterances)
- **Test (IWSLT 2026)**: [OdiaGenAI/iwslt-odia-speech](https://github.com/OdiaGenAI/iwslt-odia-speech) test set (~2.7h, 1,600 utterances)

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Batch size | 8 (effective 32 with grad_accum=4) |
| Learning rate | 1e-4 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Scheduler | Cosine with 500 warmup steps |
| Max epochs | 30 |
| Max audio length | 10s |
| GPU | NVIDIA A100 80GB |

## Architecture Notes

The base IndicConformer-600M model uses:
- **Preprocessor**: NeMo TorchScript mel-spectrogram extractor (80-dim, runs on CUDA)
- **Encoder**: ONNX InferenceSession (1024-dim output, CUDAExecutionProvider)
- **CTC Head**: `nn.Linear(1024, 91)` — the only fine-tuned parameters

The encoder is frozen (ONNX); only the Odia CTC projection head is trained.

## Usage

```python
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained(
    "shantipriya/odia-asr",
    trust_remote_code=True,
)
```

## Citation

If you use this model, please cite:

```bibtex
@misc{odia-asr-2026,
  title  = {Fine-tuned IndicConformer for Odia ASR},
  author = {Shantipriya Parida},
  year   = {2026},
  url    = {https://huggingface.co/shantipriya/odia-asr},
}
```
