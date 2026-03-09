#!/usr/bin/env python3
"""
Fine-tune ai4bharat/indic-conformer-600m-multilingual on Odia ASR data.

Dataset:  OpenSLR-103 (MUCS 2021 Sub-task 1)
Language: Odia (or)

Pipeline:
  1. Load manifests created by prepare_dataset.py
  2. Load IndicConformer + Odia character tokenizer
  3. Fine-tune CTC head with frozen / partially frozen encoder
  4. Save best checkpoint by WER on test set

Usage:
  python train.py [--config config.yaml]  (or use defaults below)
"""

import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from typing import List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    model_id        = "ai4bharat/indic-conformer-600m-multilingual",
    lang_id         = "or",                    # Odia ISO code
    train_manifest  = "data/manifests/odia_train_manifest.json",
    test_manifest   = "data/manifests/odia_test_manifest.json",
    output_dir      = "outputs/odia_finetuned",
    target_sr       = 16000,   # model expects 16 kHz
    source_sr       = 8000,    # OpenSLR-103 Odia is 8 kHz
    batch_size      = 8,
    grad_accum      = 4,       # effective batch = 32
    learning_rate   = 1e-4,
    warmup_steps    = 500,
    max_epochs      = 30,
    max_audio_secs  = 30.0,    # skip clips longer than this
    min_audio_secs  = 0.5,
    freeze_encoder_epochs = 5, # freeze encoder for first N epochs, then unfreeze
    seed            = 42,
    num_workers     = 4,
    log_interval    = 50,
    eval_every_n_epochs = 1,
    hf_token        = os.environ.get("HF_TOKEN", ""),
)


# ──────────────────────────────────────────────────────────────────────────────
# ODIA CHARACTER VOCABULARY
# Based on standard Odia Unicode block + common punctuation
# ──────────────────────────────────────────────────────────────────────────────
ODIA_VOCAB = (
    ["<blank>", "<unk>", " "] +
    # Odia vowels
    list("ଅଆଇଈଉଊଋଌଏଐଓଔ") +
    # Odia consonants
    list("କଖଗଘଙଚଛଜଝଞଟଠଡଢଣତଥଦଧନପଫବଭମଯରଲଳୱଶଷସହ") +
    # Dependent vowel signs & special marks
    list("ାିୀୁୂୃୄେୈୋୌ‌‍୍ଂଃଁ") +
    # Digits
    list("୦୧୨୩୪୫୬୭୮୯") +
    # ASCII digits fallback
    list("0123456789") +
    # Common punctuation
    list(",.?!।")
)

# Remove duplicates while preserving order
seen = set()
ODIA_VOCAB_DEDUP = []
for c in ODIA_VOCAB:
    if c not in seen:
        ODIA_VOCAB_DEDUP.append(c)
        seen.add(c)

BLANK_IDX = 0


class OdiaTokenizer:
    def __init__(self):
        self.vocab = ODIA_VOCAB_DEDUP
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}
        self.blank_id = BLANK_IDX
        self.unk_id = 1

    def encode(self, text: str) -> List[int]:
        return [self.char2idx.get(c, self.unk_id) for c in text.lower()]

    def decode(self, ids: List[int]) -> str:
        chars = []
        prev = None
        for idx in ids:
            if idx == self.blank_id:
                prev = None
                continue
            if idx != prev:
                chars.append(self.idx2char.get(idx, ""))
            prev = idx
        return "".join(chars)

    def __len__(self):
        return len(self.vocab)


# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────
class OdiaASRDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        tokenizer: OdiaTokenizer,
        target_sr: int = 16000,
        source_sr: int = 8000,
        max_secs: float = 30.0,
        min_secs: float = 0.5,
    ):
        self.tokenizer = tokenizer
        self.target_sr = target_sr
        self.source_sr = source_sr
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=source_sr, new_freq=target_sr
        )

        raw = []
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line.strip())
                dur = rec.get("duration", 999)
                if min_secs <= dur <= max_secs:
                    raw.append(rec)

        self.data = raw
        print(f"  Loaded {len(self.data)} samples from {manifest_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]
        wav, sr = torchaudio.load(rec["audio_filepath"])
        wav = torch.mean(wav, dim=0, keepdim=True)  # mono

        if sr != self.target_sr:
            if sr == self.source_sr:
                wav = self.resampler(wav)
            else:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=self.target_sr
                )
                wav = resampler(wav)

        wav = wav.squeeze(0)  # (T,)
        label = self.tokenizer.encode(rec["text"])
        return wav, torch.tensor(label, dtype=torch.long)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    wavs, labels = zip(*batch)

    wav_lengths  = torch.tensor([w.shape[0] for w in wavs], dtype=torch.long)
    label_lengths = torch.tensor([l.shape[0] for l in labels], dtype=torch.long)

    padded_wavs  = nn.utils.rnn.pad_sequence(wavs, batch_first=True)   # (B, T)
    padded_labels = nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=BLANK_IDX
    )  # (B, L)

    return padded_wavs, wav_lengths, padded_labels, label_lengths


# ──────────────────────────────────────────────────────────────────────────────
# CTC WRAPPER
# Wraps IndicConformer encoder + linear CTC projection
# ──────────────────────────────────────────────────────────────────────────────
class IndicConformerCTCFinetune(nn.Module):
    def __init__(self, base_model, vocab_size: int, encoder_dim: int = 512):
        super().__init__()
        self.base = base_model
        # Replace / add Odia-specific CTC head
        self.ctc_head = nn.Linear(encoder_dim, vocab_size)
        self.ctc_loss = nn.CTCLoss(
            blank=BLANK_IDX, reduction="mean", zero_infinity=True
        )

    def forward(self, wavs: torch.Tensor, wav_lengths: torch.Tensor):
        """
        wavs:        (B, T) float32
        wav_lengths: (B,)  int

        Returns:
          log_probs:    (T', B, V)   for CTCLoss
          input_lengths: (B,)
        """
        # IndicConformer exposes internal encoder via base.encoder
        # We bypass the decoding step and grab encoder outputs directly
        device = wavs.device
        wavs_3d = wavs.unsqueeze(1)  # (B, 1, T) — expected by many conformers

        with torch.set_grad_enabled(self.training):
            # Try NeMo-style encoder call
            try:
                enc_out, enc_len = self.base.encoder(
                    audio_signal=wavs_3d,
                    length=wav_lengths,
                )
            except TypeError:
                # Some wrappers take (wav, length) positionally
                enc_out, enc_len = self.base.encoder(wavs_3d, wav_lengths)

        # enc_out: (B, D, T') or (B, T', D)
        if enc_out.dim() == 3 and enc_out.shape[1] == enc_out.shape[-1]:
            # ambiguous — assume (B, T', D) if D == vocab head in
            pass
        if enc_out.shape[1] != enc_len.max():
            # shape is (B, D, T') — transpose to (B, T', D)
            enc_out = enc_out.transpose(1, 2)

        logits = self.ctc_head(enc_out)          # (B, T', V)
        log_probs = logits.log_softmax(-1)       # (B, T', V)
        log_probs = log_probs.transpose(0, 1)    # (T', B, V)  for CTCLoss

        return log_probs, enc_len

    def compute_loss(self, log_probs, input_lengths, targets, target_lengths):
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)


# ──────────────────────────────────────────────────────────────────────────────
# WER METRIC
# ──────────────────────────────────────────────────────────────────────────────
def word_error_rate(preds: List[str], refs: List[str]) -> float:
    total_words = total_errors = 0
    for pred, ref in zip(preds, refs):
        p_words = pred.split()
        r_words = ref.split()
        total_words += len(r_words)

        # Dynamic programming edit distance
        d = [[0] * (len(p_words) + 1) for _ in range(len(r_words) + 1)]
        for i in range(len(r_words) + 1):
            d[i][0] = i
        for j in range(len(p_words) + 1):
            d[0][j] = j
        for i in range(1, len(r_words) + 1):
            for j in range(1, len(p_words) + 1):
                if r_words[i - 1] == p_words[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
        total_errors += d[len(r_words)][len(p_words)]

    return total_errors / max(total_words, 1)


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────────────
def get_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, loader, tokenizer, device):
    model.eval()
    preds, refs = [], []
    with torch.no_grad():
        for wavs, wav_lens, labels, label_lens in loader:
            wavs = wavs.to(device)
            wav_lens = wav_lens.to(device)
            log_probs, enc_lens = model(wavs, wav_lens)
            # Greedy decode
            best = log_probs.argmax(-1).transpose(0, 1)  # (B, T')
            for i in range(best.shape[0]):
                pred_ids = best[i, : enc_lens[i]].tolist()
                pred_text = tokenizer.decode(pred_ids)
                preds.append(pred_text)
            # References
            for i in range(labels.shape[0]):
                ref_ids = labels[i, : label_lens[i]].tolist()
                ref_text = tokenizer.decode(ref_ids)
                refs.append(ref_text)
    wer = word_error_rate(preds, refs)
    return wer


def train(cfg: dict):
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = OdiaTokenizer()
    print(f"Vocabulary size: {len(tokenizer)}")

    # ── Datasets ───────────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_ds = OdiaASRDataset(
        cfg["train_manifest"], tokenizer,
        target_sr=cfg["target_sr"], source_sr=cfg["source_sr"],
        max_secs=cfg["max_audio_secs"], min_secs=cfg["min_audio_secs"],
    )
    test_ds = OdiaASRDataset(
        cfg["test_manifest"], tokenizer,
        target_sr=cfg["target_sr"], source_sr=cfg["source_sr"],
        max_secs=cfg["max_audio_secs"], min_secs=cfg["min_audio_secs"],
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
        collate_fn=collate_fn, num_workers=cfg["num_workers"],
    )

    # ── Base model ─────────────────────────────────────────────────────────────
    print(f"\nLoading {cfg['model_id']} ...")
    hf_kwargs = {"trust_remote_code": True}
    if cfg["hf_token"]:
        hf_kwargs["token"] = cfg["hf_token"]

    base = AutoModel.from_pretrained(cfg["model_id"], **hf_kwargs)

    # Detect encoder output dimension
    encoder_dim = 512  # default for 600M conformer
    if hasattr(base, "encoder"):
        for name, param in base.encoder.named_parameters():
            if "weight" in name and param.dim() == 2:
                encoder_dim = param.shape[0]
                break

    model = IndicConformerCTCFinetune(base, len(tokenizer), encoder_dim)
    model = model.to(device)

    # Freeze encoder for first N epochs
    def set_encoder_grad(requires_grad: bool):
        for p in model.base.encoder.parameters():
            p.requires_grad = requires_grad

    set_encoder_grad(False)
    print(f"Encoder frozen for first {cfg['freeze_encoder_epochs']} epochs")

    # ── Optimizer ──────────────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg["learning_rate"], weight_decay=1e-4)

    total_steps = (len(train_loader) // cfg["grad_accum"]) * cfg["max_epochs"]
    scheduler = get_scheduler(optimizer, cfg["warmup_steps"], total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Training loop ──────────────────────────────────────────────────────────
    best_wer = float("inf")
    global_step = 0

    for epoch in range(1, cfg["max_epochs"] + 1):

        # Unfreeze encoder after freeze period
        if epoch == cfg["freeze_encoder_epochs"] + 1:
            set_encoder_grad(True)
            # Re-init optimizer with all params
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg["learning_rate"] / 10,  # lower LR for encoder
                weight_decay=1e-4,
            )
            scheduler = get_scheduler(optimizer, cfg["warmup_steps"], total_steps)
            if scaler.is_enabled():
                scaler = torch.cuda.amp.GradScaler()
            print(f"\nEpoch {epoch}: Encoder unfrozen (LR = {cfg['learning_rate']/10:.2e})")

        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, (wavs, wav_lens, labels, label_lens) in enumerate(train_loader, 1):
            wavs = wavs.to(device)
            wav_lens = wav_lens.to(device)
            labels = labels.to(device)
            label_lens = label_lens.to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                log_probs, enc_lens = model(wavs, wav_lens)
                loss = model.compute_loss(log_probs, enc_lens, labels, label_lens)
                loss = loss / cfg["grad_accum"]

            scaler.scale(loss).backward()

            if step % cfg["grad_accum"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * cfg["grad_accum"]

            if step % cfg["log_interval"] == 0:
                avg = epoch_loss / step
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch} [{step}/{len(train_loader)}] "
                    f"loss={avg:.4f}  lr={lr:.2e}"
                )

        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch} complete — avg loss: {avg_loss:.4f}")

        if epoch % cfg["eval_every_n_epochs"] == 0:
            wer = evaluate(model, test_loader, tokenizer, device)
            print(f"  Test WER: {wer * 100:.2f}%")

            ckpt_path = os.path.join(cfg["output_dir"], f"epoch{epoch}_wer{wer*100:.1f}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "wer": wer,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"  Saved checkpoint: {ckpt_path}")

            if wer < best_wer:
                best_wer = wer
                best_path = os.path.join(cfg["output_dir"], "best_model.pt")
                torch.save(model.state_dict(), best_path)
                print(f"  *** New best WER: {best_wer * 100:.2f}% → {best_path}")

    print(f"\nTraining complete. Best WER: {best_wer * 100:.2f}%")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    for key, val in DEFAULTS.items():
        p.add_argument(f"--{key}", type=type(val) if val is not None else str,
                       default=val)
    return vars(p.parse_args())


if __name__ == "__main__":
    cfg = parse_args()
    print("Config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print()
    train(cfg)
