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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default=RAW_DIR)
    parser.add_argument("--manifest_dir", default=MANIFEST_DIR)
    args = parser.parse_args()

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

    print("\nDone. Manifests in:", args.manifest_dir)
    print("Next step: python train.py")


if __name__ == "__main__":
    main()
