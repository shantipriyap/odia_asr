#!/usr/bin/env python3
"""
Inference with fine-tuned IndicConformer on Odia audio.

Usage:
  # Use the original model (no fine-tuning):
  python inference.py --audio path/to/audio.wav

  # Use fine-tuned checkpoint:
  python inference.py --audio path/to/audio.wav --checkpoint outputs/odia_finetuned/best_model.pt
"""

import os
import argparse
import torch
import torchaudio
from transformers import AutoModel

from train import OdiaTokenizer, IndicConformerCTCFinetune

HF_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
LANG_ID     = "or"
TARGET_SR   = 16000


def load_audio(path: str, target_sr: int = TARGET_SR) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
    return wav.squeeze(0)  # (T,)


def transcribe_baseline(audio_path: str, hf_token: str = "") -> str:
    """Use the original model's built-in CTC decoding (no fine-tuning)."""
    print("Loading base model ...")
    kwargs = {"trust_remote_code": True}
    if hf_token:
        kwargs["token"] = hf_token
    model = AutoModel.from_pretrained(HF_MODEL_ID, **kwargs)
    model.eval()

    wav, sr = torchaudio.load(audio_path)
    wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)

    with torch.no_grad():
        result = model(wav, LANG_ID, "ctc")
    return result


def transcribe_finetuned(audio_path: str, checkpoint: str, hf_token: str = "") -> str:
    """Use the fine-tuned CTC head."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading base model + checkpoint: {checkpoint} ...")

    kwargs = {"trust_remote_code": True}
    if hf_token:
        kwargs["token"] = hf_token
    base = AutoModel.from_pretrained(HF_MODEL_ID, **kwargs)

    tokenizer = OdiaTokenizer()
    model = IndicConformerCTCFinetune(base, len(tokenizer))
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device).eval()

    wav = load_audio(audio_path).to(device)
    wav_len = torch.tensor([wav.shape[0]], dtype=torch.long).to(device)

    with torch.no_grad():
        log_probs, enc_lens = model(wav.unsqueeze(0), wav_len)
        best = log_probs.argmax(-1).squeeze(1)  # (T',)
        ids = best[: enc_lens[0]].tolist()

    return tokenizer.decode(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",      required=True, help="Path to .wav file")
    parser.add_argument("--checkpoint", default="",    help="Path to fine-tuned .pt checkpoint")
    parser.add_argument("--hf_token",   default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()

    if args.checkpoint and os.path.exists(args.checkpoint):
        text = transcribe_finetuned(args.audio, args.checkpoint, args.hf_token)
        print(f"\n[Fine-tuned] Transcription:\n{text}")
    else:
        text = transcribe_baseline(args.audio, args.hf_token)
        print(f"\n[Baseline] Transcription:\n{text}")


if __name__ == "__main__":
    main()
