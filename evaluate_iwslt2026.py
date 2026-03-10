#!/usr/bin/env python3
"""
Evaluate fine-tuned IndicConformer on IWSLT 2026 Odia data.

Expected dataset layout (follows shashwatup9k/iwslt2026_or-hi-eng):

  <data_root>/
  ├── train/
  │   ├── stamped.tsv
  │   ├── wav/
  │   └── txt/
  │       ├── train.or        (Odia source transcriptions)
  │       ├── train.hi        (Hindi translations)
  │       └── train.en        (English translations, if present)
  ├── dev/
  │   ├── stamped.tsv
  │   ├── wav/
  │   └── txt/
  │       ├── dev.or
  │       └── dev.hi
  └── test-*/
      ├── stamped.tsv
      └── wav/

stamped.tsv columns (tab-separated, IWSLT standard):
  talk_id  wav_filename  offset  duration  src_text  [tgt_text]
  -or-
  id  wav_filename  src_text  [tgt_text]
  (the parser handles both)

Metrics:
  - WER  (ASR: Odia transcription quality)
  - CER  (character-level, more stable for Odia)
  - BLEU (ST: if a translation reference is available)

Usage:
  python evaluate_iwslt2026.py \
      --data_root /path/to/iwslt2026_or-hi-eng \
      --split dev \
      --checkpoint outputs/best_model.pt \
      [--baseline]          # run base IndicConformer without fine-tuning
      [--output results.json]
"""

import os, json, glob, wave, argparse, re, math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torchaudio


# ──────────────────────────────────────────────────────────
# Odia tokenizer (same as training)
# ──────────────────────────────────────────────────────────
ODIA_VOCAB_RAW = (
    ["<blank>", "<unk>", " "]
    + list("ଅଆଇଈଉଊଋଌଏଐଓଔ")
    + list("କଖଗଘଙଚଛଜଝଞଟଠଡଢଣତଥଦଧନପଫବଭମଯରଲଳୱଶଷସହ")
    + list("ାିୀୁୂୃୄେୈୋୌ\u200c\u200d୍ଂଃଁ")
    + list("୦୧୨୩୪୫୬୭୮୯")
    + list("0123456789")
    + list(",.?!।")
)
seen = set()
ODIA_VOCAB: List[str] = []
for _c in ODIA_VOCAB_RAW:
    if _c not in seen:
        ODIA_VOCAB.append(_c)
        seen.add(_c)


class OdiaTokenizer:
    def __init__(self):
        self.vocab = ODIA_VOCAB
        self.c2i   = {c: i for i, c in enumerate(self.vocab)}
        self.i2c   = {i: c for i, c in enumerate(self.vocab)}
        self.blank_id = 0
        self.unk_id   = 1

    def encode(self, text: str) -> List[int]:
        return [self.c2i.get(c, self.unk_id) for c in text]

    def decode(self, ids: List[int]) -> str:
        chars, prev = [], None
        for idx in ids:
            if idx == self.blank_id:
                prev = None
                continue
            if idx != prev:
                chars.append(self.i2c.get(idx, ""))
            prev = idx
        return "".join(chars)

    def __len__(self):
        return len(self.vocab)


# ──────────────────────────────────────────────────────────
# Model wrapper (same as training)
# ──────────────────────────────────────────────────────────
class CTCModel(nn.Module):
    def __init__(self, base, vocab_size: int, enc_dim: int = 512):
        super().__init__()
        self.base     = base
        self.ctc_head = nn.Linear(enc_dim, vocab_size)

    def encode(self, wavs: torch.Tensor, wav_lens: torch.Tensor):
        # Step 1: Run preprocessor (TorchScript) to get mel features
        preprocessor = self.base.models["preprocessor"]
        encoder_sess = self.base.models["encoder"]
        mel, mel_len = preprocessor(wavs.cuda(), wav_lens.cuda())
        # Step 2: Run ONNX encoder on mel features
        mel_np = mel.cpu().numpy().astype("float32")
        mel_len_np = mel_len.cpu().numpy().astype("int64")
        enc_out = encoder_sess.run(None, {"audio_signal": mel_np, "length": mel_len_np})
        enc = torch.tensor(enc_out[0])
        enc_len = torch.tensor(enc_out[1])
        if enc.shape[1] != enc_len.max():
            enc = enc.transpose(1, 2)
        return enc, enc_len

    def forward(self, wavs: torch.Tensor, wav_lens: torch.Tensor):
        enc, enc_len = self.encode(wavs, wav_lens)
        # Move enc to same device as ctc_head weights
        logits = self.ctc_head(enc.to(self.ctc_head.weight.device))
        log_probs = logits.log_softmax(-1).transpose(0, 1)
        return log_probs, enc_len


# ──────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────
def _edit_distance(a: List[str], b: List[str]) -> int:
    """Standard edit distance between two token lists."""
    la, lb = len(a), len(b)
    d = list(range(lb + 1))
    for i, ca in enumerate(a):
        nd = [i + 1] + [0] * lb
        for j, cb in enumerate(b):
            nd[j + 1] = d[j] if ca == cb else 1 + min(d[j], d[j + 1], nd[j])
        d = nd
    return d[-1]


def wer(preds: List[str], refs: List[str]) -> float:
    total_words, total_err = 0, 0
    for p, r in zip(preds, refs):
        rw = r.split()
        total_words += len(rw)
        total_err   += _edit_distance(p.split(), rw)
    return total_err / max(total_words, 1)


def cer(preds: List[str], refs: List[str]) -> float:
    total_chars, total_err = 0, 0
    for p, r in zip(preds, refs):
        total_chars += len(r)
        total_err   += _edit_distance(list(p), list(r))
    return total_err / max(total_chars, 1)


def bleu(preds: List[str], refs: List[str], max_n: int = 4) -> float:
    """Corpus-level BLEU (simplified, no smoothing)."""
    clip_counts = [0] * max_n
    total_counts = [0] * max_n
    pred_len, ref_len = 0, 0

    for hyp, ref in zip(preds, refs):
        hyp_tok = hyp.split()
        ref_tok = ref.split()
        pred_len += len(hyp_tok)
        ref_len  += len(ref_tok)
        for n in range(1, max_n + 1):
            from collections import Counter
            hyp_ngrams = Counter(
                tuple(hyp_tok[i : i + n]) for i in range(len(hyp_tok) - n + 1)
            )
            ref_ngrams = Counter(
                tuple(ref_tok[i : i + n]) for i in range(len(ref_tok) - n + 1)
            )
            clipped = sum(
                min(c, ref_ngrams[ng]) for ng, c in hyp_ngrams.items()
            )
            clip_counts[n - 1]  += clipped
            total_counts[n - 1] += max(len(hyp_tok) - n + 1, 0)

    bp = min(1.0, math.exp(1 - ref_len / max(pred_len, 1)))
    log_avg = 0.0
    valid = 0
    for n in range(max_n):
        if total_counts[n] > 0 and clip_counts[n] > 0:
            log_avg += math.log(clip_counts[n] / total_counts[n])
            valid   += 1
    if valid == 0:
        return 0.0
    return bp * math.exp(log_avg / valid) * 100


# ──────────────────────────────────────────────────────────
# IWSLT 2026 data loader
# ──────────────────────────────────────────────────────────
def parse_stamped_tsv(tsv_path: str) -> List[Dict]:
    """
    Parse IWSLT stamped.tsv.  Handles three layouts:

    Layout A (standard IWSLT / shashwatup9k):
      talk_id  wav_filename  offset  duration  src_text  [tgt_text]
      (5+ cols; cols[2] and cols[3] are numeric; col[4] = Odia text)

    Layout B (minimal with id):
      id  wav_filename  src_text  [tgt_text]
      (col[2] is NOT numeric; col[2] = Odia text)

    Layout C (OdiaGenAI/iwslt-odia-speech):
      wav_filename  offset  duration
      (3 cols only; col[0] ends in .wav; text loaded separately from or/txt.or)

    Returns list of dicts with keys:
      id, wav_filename, offset, duration, src_text, tgt_text (may be "")
    """
    records = []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 3:
                continue

            # Layout C: exactly 3 cols AND col[0] looks like a wav filename
            if len(cols) == 3 and (cols[0].strip().endswith(".wav") or
                                    cols[0].strip().endswith(".flac")):
                try:
                    records.append({
                        "id":           Path(cols[0].strip()).stem,
                        "wav_filename": cols[0].strip(),
                        "offset":       float(cols[1]),
                        "duration":     float(cols[2]),
                        "src_text":     "",   # filled later from or/txt.or
                        "tgt_text":     "",
                    })
                    continue
                except ValueError:
                    pass  # fall through to other layouts

            # Detect Layout A vs B by whether col[2] looks numeric
            try:
                float(cols[2])
                is_layout_a = True
            except ValueError:
                is_layout_a = False

            if is_layout_a and len(cols) >= 5:
                # Layout A: id, wav, offset, duration, src [, tgt ...]
                records.append({
                    "id":           cols[0].strip(),
                    "wav_filename": cols[1].strip(),
                    "offset":       float(cols[2]),
                    "duration":     float(cols[3]),
                    "src_text":     cols[4].strip(),
                    "tgt_text":     cols[5].strip() if len(cols) > 5 else "",
                })
            else:
                # Layout B: id, wav, src [, tgt ...]
                records.append({
                    "id":           cols[0].strip(),
                    "wav_filename": cols[1].strip(),
                    "offset":       0.0,
                    "duration":     -1.0,
                    "src_text":     cols[2].strip(),
                    "tgt_text":     cols[3].strip() if len(cols) > 3 else "",
                })
    return records


def load_txt_refs(txt_dir: str, split: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Load source (Odia) and target (Hindi/English) references from a directory.
    Handles two naming conventions:
      - Standard IWSLT:  {split}.or / {split}.hi / {split}.en  (shashwatup9k)
      - OdiaGenAI:       txt.or / dev.hi / dev.en               (bare filenames)
    Returns (src_lines, hi_lines, en_lines) — any may be empty if file absent.
    """
    def read_lines(path):
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]

    src = (
        read_lines(os.path.join(txt_dir, f"{split}.or"))
        or read_lines(os.path.join(txt_dir, f"{split}.oria"))
        or read_lines(os.path.join(txt_dir, f"{split}.odi"))
        or read_lines(os.path.join(txt_dir, "txt.or"))   # OdiaGenAI: or/txt.or
        or read_lines(os.path.join(txt_dir, "src"))
    )
    hi = (
        read_lines(os.path.join(txt_dir, f"{split}.hi"))
        or read_lines(os.path.join(txt_dir, "dev.hi"))   # OdiaGenAI: translations/dev.hi
    )
    en = (
        read_lines(os.path.join(txt_dir, f"{split}.en"))
        or read_lines(os.path.join(txt_dir, "dev.en"))   # OdiaGenAI: translations/dev.en
    )
    return src, hi, en


def load_split(data_root: str, split: str) -> List[Dict]:
    """
    Load all samples for a given split.  Supports two repo layouts:

    OdiaGenAI (OdiaGenAI/iwslt-odia-speech):
      test_set/
        audio/               ← wav files
        stamped.tsv          ← 3-col: wav_filename, offset, duration (no text)
        or/txt.or            ← Odia transcriptions (positionally aligned with stamped.tsv)
        translations/dev.hi  ← Hindi translations
        translations/dev.en  ← English translations
      train/
        audio/               ← wav files
        txt.or               ← Odia transcriptions (positionally aligned with sorted audio/)

    Standard IWSLT (shashwatup9k-style):
      {split}/
        wav/                 ← wav files
        stamped.tsv          ← 5-col: id, wav, offset, duration, src_text [, tgt]
        txt/{split}.or/.hi/.en
    """
    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    # Detect wav directory: prefer audio/ (OdiaGenAI) over wav/ (standard IWSLT)
    wav_dir = os.path.join(split_dir, "audio")
    if not os.path.isdir(wav_dir):
        wav_dir = os.path.join(split_dir, "wav")

    tsv_path      = os.path.join(split_dir, "stamped.tsv")
    or_txt_path   = os.path.join(split_dir, "or", "txt.or")   # OdiaGenAI test_set
    root_txt_path = os.path.join(split_dir, "txt.or")          # OdiaGenAI train
    trans_dir     = os.path.join(split_dir, "translations")
    txt_dir       = os.path.join(split_dir, "txt")             # standard IWSLT

    is_odiagen = (
        os.path.isfile(or_txt_path)
        or os.path.isfile(root_txt_path)
        or os.path.isdir(trans_dir)
    )

    records: List[Dict] = []

    if os.path.exists(tsv_path):
        records = parse_stamped_tsv(tsv_path)
    else:
        # No TSV (e.g. OdiaGenAI train split): enumerate sorted wav files
        wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
        records = [{"id": Path(w).stem, "wav_filename": Path(w).name,
                    "offset": 0.0, "duration": -1.0,
                    "src_text": "", "tgt_text": ""} for w in wavs]

    # ── Load text references ──────────────────────────────────────────────
    if is_odiagen:
        def _read(path):
            if not os.path.exists(path):
                return []
            with open(path, encoding="utf-8") as f:
                return [l.strip() for l in f if l.strip()]

        src_lines = _read(or_txt_path) or _read(root_txt_path)
        hi_lines  = _read(os.path.join(trans_dir, "dev.hi"))
        en_lines  = _read(os.path.join(trans_dir, "dev.en"))

        for i, r in enumerate(records):
            if i < len(src_lines):
                r["src_text"] = src_lines[i]
            if i < len(hi_lines):
                r["tgt_hi"] = hi_lines[i]
            if i < len(en_lines):
                r["tgt_en"] = en_lines[i]

    elif os.path.isdir(txt_dir):
        # Standard IWSLT layout: txt/{split}.or, txt/{split}.hi, txt/{split}.en
        split_key = split.replace("test-", "test").replace("test_", "test")
        for key in [split_key, split.split("-")[0], split]:
            src_lines, hi_lines, en_lines = load_txt_refs(txt_dir, key)
            if src_lines:
                for i, r in enumerate(records):
                    if i < len(src_lines):
                        r["src_text"] = src_lines[i]
                    if i < len(hi_lines):
                        r["tgt_hi"] = hi_lines[i]
                    if i < len(en_lines):
                        r["tgt_en"] = en_lines[i]
                break

    # ── Resolve absolute wav paths ────────────────────────────────────────
    for r in records:
        if os.path.isdir(wav_dir):
            candidate = os.path.join(wav_dir, r["wav_filename"])
            if not os.path.exists(candidate) and not r["wav_filename"].endswith(".wav"):
                candidate += ".wav"
            r["wav_path"] = candidate
        else:
            r["wav_path"] = r["wav_filename"]  # may be absolute already

    return records


# ──────────────────────────────────────────────────────────
# Audio loading
# ──────────────────────────────────────────────────────────
def load_audio(
    wav_path: str,
    offset: float = 0.0,
    duration: float = -1.0,
    target_sr: int = 16000,
) -> Optional[torch.Tensor]:
    if not os.path.exists(wav_path):
        return None
    import soundfile as sf
    try:
        wav, sr = sf.read(wav_path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)  # → mono

        # Trim to segment if offset/duration are specified
        if offset > 0 or duration > 0:
            start_frame = int(offset * sr)
            if duration > 0:
                end_frame = start_frame + int(duration * sr)
                wav = wav[start_frame:end_frame]
            else:
                wav = wav[start_frame:]

        if sr != target_sr:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return torch.tensor(wav, dtype=torch.float32)
    except Exception as e:
        print(f"  [WARN] Could not load {wav_path}: {e}")
        return None


# ──────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────
def transcribe_batch(
    model: CTCModel,
    tokenizer: OdiaTokenizer,
    wavs: List[torch.Tensor],
    device: torch.device,
    batch_size: int = 8,
) -> List[str]:
    results = []
    for i in range(0, len(wavs), batch_size):
        batch = wavs[i : i + batch_size]
        wav_lens = torch.tensor([w.shape[0] for w in batch], dtype=torch.long)
        padded   = nn.utils.rnn.pad_sequence(batch, batch_first=True)
        padded, wav_lens = padded.to(device), wav_lens.to(device)

        with torch.no_grad():
            lp, enc_len = model(padded, wav_lens)
        best = lp.argmax(-1).transpose(0, 1)
        for j in range(best.shape[0]):
            ids = best[j, : enc_len[j]].tolist()
            results.append(tokenizer.decode(ids))
    return results


# ──────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print(f"\nLoading split '{args.split}' from: {args.data_root}")
    records = load_split(args.data_root, args.split)
    print(f"  {len(records)} records found")

    # Load model
    tokenizer = OdiaTokenizer()
    print(f"Vocab size: {len(tokenizer)}")

    from transformers import AutoModel
    print(f"Loading base model: {args.model_id}")
    base = AutoModel.from_pretrained(args.model_id, trust_remote_code=True)

    model = CTCModel(base, len(tokenizer), enc_dim=args.enc_dim).to(device)

    if not args.baseline:
        print(f"Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        # Robust patch: handle ctc_head as tensor, nn.Linear, dict, or raw dict
        if isinstance(state, dict):
            if "ctc_head" in state:
                ctc = state["ctc_head"]
                state_dict = {}
                # Handle nn.Linear
                if hasattr(ctc, "weight") and hasattr(ctc, "bias"):
                    state_dict["ctc_head.weight"] = ctc.weight
                    state_dict["ctc_head.bias"] = ctc.bias
                # Handle dict
                elif isinstance(ctc, dict):
                    state_dict["ctc_head.weight"] = ctc.get("weight")
                    state_dict["ctc_head.bias"] = ctc.get("bias")
                # Handle tuple/list
                elif isinstance(ctc, (tuple, list)) and len(ctc) == 2:
                    state_dict["ctc_head.weight"] = ctc[0]
                    state_dict["ctc_head.bias"] = ctc[1]
                # Handle tensor (weight only)
                elif hasattr(ctc, "shape"):
                    state_dict["ctc_head.weight"] = ctc
                model.load_state_dict(state_dict, strict=False)
                print("  Checkpoint loaded (ctc_head weights, robust patch).")
            elif "model_state" in state:
                model.load_state_dict(state["model_state"])
                print("  Checkpoint loaded (model_state).")
            else:
                model.load_state_dict(state, strict=False)
                print("  Checkpoint loaded (raw dict).")
        else:
            model.load_state_dict(state, strict=False)
            print("  Checkpoint loaded (raw object).")
    else:
        print("  Running baseline (no fine-tuning weights loaded).")

    model.eval()

    # Run inference
    print("\nRunning inference...")
    preds: List[str] = []
    refs_asr: List[str]  = []
    refs_hi: List[str]   = []
    refs_en: List[str]   = []
    skipped = 0

    wavs_buf: List[torch.Tensor] = []
    rec_buf: List[Dict] = []

    def flush():
        nonlocal preds
        preds.extend(transcribe_batch(model, tokenizer, wavs_buf, device, args.batch_size))
        wavs_buf.clear()
        rec_buf.clear()

    for idx, rec in enumerate(records):
        wav = load_audio(rec["wav_path"], rec.get("offset", 0.0),
                         rec.get("duration", -1.0), args.target_sr)
        if wav is None:
            skipped += 1
            refs_asr.append(rec.get("src_text", ""))
            refs_hi.append(rec.get("tgt_hi", rec.get("tgt_text", "")))
            refs_en.append(rec.get("tgt_en", ""))
            preds.append("")
            continue

        wavs_buf.append(wav)
        rec_buf.append(rec)
        refs_asr.append(rec.get("src_text", ""))
        refs_hi.append(rec.get("tgt_hi", rec.get("tgt_text", "")))
        refs_en.append(rec.get("tgt_en", ""))

        if len(wavs_buf) >= args.batch_size:
            flush()

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(records)}] buffered={len(wavs_buf)}")

    if wavs_buf:
        flush()

    print(f"  Done. Skipped {skipped} files.")

    # Compute metrics
    print("\n" + "=" * 60)
    print(f"  IWSLT 2026 Odia Evaluation — split: {args.split}")
    print("=" * 60)

    has_src  = any(r for r in refs_asr)
    has_hi   = any(r for r in refs_hi)
    has_en   = any(r for r in refs_en)

    results: Dict = {"split": args.split, "n_samples": len(records),
                     "skipped": skipped, "baseline": args.baseline}

    if has_src:
        wer_val = wer(preds, refs_asr) * 100
        cer_val = cer(preds, refs_asr) * 100
        print(f"  ASR WER : {wer_val:.2f}%")
        print(f"  ASR CER : {cer_val:.2f}%")
        results["wer"] = round(wer_val, 4)
        results["cer"] = round(cer_val, 4)
    else:
        print("  No Odia reference transcriptions found → skipping ASR metrics.")

    if has_hi:
        bleu_hi = bleu(preds, refs_hi)  # hypotheses are Odia, refs are Hindi → only for ST systems
        print(f"  [INFO] Hindi refs present ({sum(1 for r in refs_hi if r)} sentences)")
        results["refs_hi_count"] = sum(1 for r in refs_hi if r)

    if has_en:
        print(f"  [INFO] English refs present ({sum(1 for r in refs_en if r)} sentences)")
        results["refs_en_count"] = sum(1 for r in refs_en if r)

    print("=" * 60)

    # Sample predictions
    print("\nSample predictions (first 5):")
    for i in range(min(5, len(preds))):
        ref_str = refs_asr[i] if refs_asr else ""
        print(f"  [{i}] REF: {ref_str[:80]}")
        print(f"       HYP: {preds[i][:80]}")

    # Save results
    if args.output:
        out_data = {
            "metrics": results,
            "samples": [
                {
                    "id":  records[i].get("id", str(i)),
                    "ref": refs_asr[i] if i < len(refs_asr) else "",
                    "hyp": preds[i],
                    "ref_hi": refs_hi[i] if i < len(refs_hi) else "",
                }
                for i in range(len(preds))
            ],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {args.output}")

    return results


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate IndicConformer on IWSLT 2026 Odia (OdiaGenAI/iwslt-odia-speech)"
    )
    parser.add_argument(
        "--data_root", required=True,
        help="Root directory of the iwslt-odia-speech repo checkout",
    )
    parser.add_argument(
        "--split", default="test_set",
        help="Split to evaluate: train | test_set  (OdiaGenAI repo layout)",
    )
    parser.add_argument(
        "--checkpoint", default="outputs/best_model.pt",
        help="Path to fine-tuned model checkpoint (.pt)",
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Evaluate the base model without loading any checkpoint",
    )
    parser.add_argument(
        "--model_id", default="ai4bharat/indic-conformer-600m-multilingual",
        help="HuggingFace model ID for the base model",
    )
    parser.add_argument(
        "--enc_dim", type=int, default=512,
        help="Encoder output dimension (512 for IndicConformer-600M)",
    )
    parser.add_argument(
        "--target_sr", type=int, default=16000,
        help="Target sample rate for audio resampling",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Inference batch size",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save JSON results (optional)",
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
