#!/usr/bin/env python3
"""
Prepare NeMo-style JSON manifests from the OpenSLR-103 Odia dataset.

Expected raw structure after extraction:
  data/raw/Odia_train/
    audio/          (*.wav files @ 8kHz)
    transcription.txt   (one line per utt: "<file_id> <transcript>")
  data/raw/Odia_test/
    audio/
    transcription.txt

Produces:
  data/manifests/odia_train_manifest.json
  data/manifests/odia_test_manifest.json
"""

import os
import json
import glob
import wave
import argparse

BASE_DIR    = os.path.dirname(__file__)
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
MANIFEST_DIR = os.path.join(BASE_DIR, "data", "manifests")


def get_wav_duration(wav_path: str) -> float:
    with wave.open(wav_path, "r") as wf:
        return wf.getnframes() / wf.getframerate()


def load_transcriptions(trans_file: str) -> dict:
    """
    Parse transcription.txt. Supports formats:
      - "<file_id> <transcript>"   (space-separated, first token = id)
      - "<file_id>\t<transcript>"  (tab-separated)
    """
    trans = {}
    with open(trans_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                parts = line.split("\t", 1)
            else:
                parts = line.split(" ", 1)
            if len(parts) == 2:
                file_id, text = parts
                trans[file_id.strip()] = text.strip()
    return trans


def find_transcription_file(split_dir: str) -> str:
    """Try common names for the transcription file."""
    candidates = [
        "transcription.txt",
        "transcriptions.txt",
        "transcript.txt",
        "text",
        "transcription.tsv",
    ]
    for name in candidates:
        path = os.path.join(split_dir, name)
        if os.path.exists(path):
            return path
    # Fallback: any .txt file at root of split dir
    txts = glob.glob(os.path.join(split_dir, "*.txt"))
    if txts:
        return txts[0]
    raise FileNotFoundError(f"No transcription file found in {split_dir}")


def build_manifest(split_name: str, split_dir: str, out_path: str):
    audio_dir = os.path.join(split_dir, "audio")
    if not os.path.isdir(audio_dir):
        # Some releases put wavs directly in the split dir
        audio_dir = split_dir

    trans_file = find_transcription_file(split_dir)
    print(f"  Using transcriptions: {trans_file}")
    transcriptions = load_transcriptions(trans_file)

    wav_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    print(f"  Found {len(wav_files)} WAV files")

    records = []
    missing_trans = 0
    for wav_path in wav_files:
        file_id = os.path.splitext(os.path.basename(wav_path))[0]
        text = transcriptions.get(file_id)
        if text is None:
            missing_trans += 1
            continue
        try:
            duration = get_wav_duration(wav_path)
        except Exception:
            duration = 0.0
        records.append({
            "audio_filepath": os.path.abspath(wav_path),
            "text": text,
            "duration": round(duration, 3),
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  Written {len(records)} entries to {out_path}")
    if missing_trans:
        print(f"  WARNING: {missing_trans} WAV files had no transcript and were skipped")

    total_hrs = sum(r["duration"] for r in records) / 3600
    print(f"  Total audio: {total_hrs:.2f} hours")


def build_iwslt_manifest(iwslt_split_dir: str, out_path: str):
    """
    Build a manifest from an IWSLT 2026 Odia speech split.

    Expected layout:
      <iwslt_split_dir>/
        stamped.tsv          (wav_filename \\t start \\t end  — one row per utt)
        audio/               (wav files)
        or/txt.or            (Odia transcripts, one per line, same order as TSV)
    """
    tsv_path   = os.path.join(iwslt_split_dir, "stamped.tsv")
    trans_path = os.path.join(iwslt_split_dir, "or", "txt.or")
    audio_dir  = os.path.join(iwslt_split_dir, "audio")

    if not os.path.isfile(tsv_path):
        raise FileNotFoundError(f"stamped.tsv not found in {iwslt_split_dir}")
    if not os.path.isfile(trans_path):
        raise FileNotFoundError(f"or/txt.or not found in {iwslt_split_dir}")

    with open(tsv_path, encoding="utf-8") as f:
        wav_names = [line.split("\t")[0].strip() for line in f if line.strip()]

    with open(trans_path, encoding="utf-8") as f:
        transcripts = [line.strip() for line in f if line.strip()]

    if len(wav_names) != len(transcripts):
        raise ValueError(
            f"stamped.tsv has {len(wav_names)} rows but or/txt.or has "
            f"{len(transcripts)} lines — lengths must match"
        )

    records = []
    missing = 0
    for wav_name, text in zip(wav_names, transcripts):
        wav_path = os.path.join(audio_dir, wav_name)
        if not os.path.isfile(wav_path):
            missing += 1
            continue
        try:
            duration = get_wav_duration(wav_path)
        except Exception:
            duration = 0.0
        records.append({
            "audio_filepath": os.path.abspath(wav_path),
            "text": text,
            "duration": round(duration, 3),
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  Written {len(records)} entries to {out_path}")
    if missing:
        print(f"  WARNING: {missing} WAV files not found and were skipped")
    total_hrs = sum(r["duration"] for r in records) / 3600
    print(f"  Total audio: {total_hrs:.2f} hours")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default=RAW_DIR)
    parser.add_argument("--manifest_dir", default=MANIFEST_DIR)
    parser.add_argument(
        "--iwslt_dir",
        default=None,
        help="Path to cloned iwslt-odia-speech repo (e.g. data/raw/iwslt-odia-speech). "
             "If provided, builds manifests for its test_set/ and train/ splits.",
    )
    args = parser.parse_args()

    # ── OpenSLR-103 splits ────────────────────────────────────────────────────
    splits = {
        "train": os.path.join(args.raw_dir, "Odia_train"),
        "test":  os.path.join(args.raw_dir, "Odia_test"),
    }

    for split, split_dir in splits.items():
        if not os.path.isdir(split_dir):
            print(f"[SKIP] {split_dir} not found — run download_data.py first")
            continue
        out_path = os.path.join(args.manifest_dir, f"odia_{split}_manifest.json")
        print(f"\n[{split.upper()}] {split_dir}")
        build_manifest(split, split_dir, out_path)

    # ── IWSLT 2026 Odia splits (optional) ────────────────────────────────────
    iwslt_dir = args.iwslt_dir
    if iwslt_dir is None:
        # Auto-detect if cloned inside raw_dir
        default_iwslt = os.path.join(args.raw_dir, "iwslt-odia-speech")
        if os.path.isdir(default_iwslt):
            iwslt_dir = default_iwslt

    if iwslt_dir and os.path.isdir(iwslt_dir):
        iwslt_splits = {
            "iwslt_test":  os.path.join(iwslt_dir, "test_set"),
            "iwslt_train": os.path.join(iwslt_dir, "train"),
        }
        for split_name, split_dir in iwslt_splits.items():
            if not os.path.isdir(split_dir):
                print(f"[SKIP] {split_dir} not found")
                continue
            out_path = os.path.join(args.manifest_dir, f"odia_{split_name}_manifest.json")
            print(f"\n[{split_name.upper()}] {split_dir}")
            try:
                build_iwslt_manifest(split_dir, out_path)
            except Exception as e:
                print(f"  ERROR: {e}")
    else:
        print("\n[IWSLT] Skipped — clone with:")
        print("  cd data/raw && git clone https://github.com/OdiaGenAI/iwslt-odia-speech.git")

    print("\nDone. Manifests in:", args.manifest_dir)
    print("Next step: python train.py")


if __name__ == "__main__":
    main()
