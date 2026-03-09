#!/usr/bin/env python3
"""
Download and extract the Odia ASR dataset.

Sources (choose one via flags):
  1. OpenSLR-103 (default) — MUCS 2021 Challenge Sub-task 1
     https://www.openslr.org/103/

  2. Google Drive  --gdrive
     Upload Odia_train.tar.gz and Odia_test.tar.gz to your Google Drive,
     then share each file (anyone with the link) and pass the file IDs:
       python download_data.py --gdrive \
           --train-id <GDRIVE_FILE_ID> \
           --test-id  <GDRIVE_FILE_ID>

     To find a file ID: open the file in Google Drive → Share → Copy link.
     The ID is the long string between /d/ and /view in the URL:
       https://drive.google.com/file/d/<FILE_ID>/view

  3. Mounted Google Drive  --gdrive-mount <path>
     If you have Google Drive for Desktop (macOS) or Colab, point directly
     to the folder that already contains the extracted data:
       python download_data.py \
           --gdrive-mount "/Volumes/GoogleDrive/My Drive/odia_asr_data"
     This symlinks (or copies) the folder into data/raw/ without downloading.
"""

import os
import tarfile
import urllib.request
import argparse
import shutil

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

URLS = {
    "Odia_train.tar.gz": "https://openslr.trmal.net/resources/103/Odia_train.tar.gz",
    "Odia_test.tar.gz":  "https://openslr.trmal.net/resources/103/Odia_test.tar.gz",
}

MIRRORS = {
    "Odia_train.tar.gz": "https://openslr.elda.org/resources/103/Odia_train.tar.gz",
    "Odia_test.tar.gz":  "https://openslr.elda.org/resources/103/Odia_test.tar.gz",
}


# ── OpenSLR download ──────────────────────────────────────────────────────────

def progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = int(pct / 2)
        print(f"\r  [{'#' * bar}{'.' * (50 - bar)}] {pct:.1f}%", end="", flush=True)


def download_file(url: str, dest: str, mirror: str = None):
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading {url} ...")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
        print()
    except Exception as e:
        print(f"\n  Primary failed ({e}), trying mirror ...")
        if mirror:
            urllib.request.urlretrieve(mirror, dest, reporthook=progress_hook)
            print()
        else:
            raise


def extract(tar_path: str, extract_to: str):
    print(f"  Extracting {tar_path} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_to)
    print(f"  Extracted to {extract_to}")


# ── Google Drive download (gdown) ─────────────────────────────────────────────

def download_from_gdrive(file_id: str, dest: str):
    """Download a single file from Google Drive using its file ID."""
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return
    try:
        import gdown
    except ImportError:
        raise SystemExit("gdown is required: pip install gdown>=5.1.0")

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  Downloading from Google Drive (id={file_id}) ...")
    gdown.download(url, dest, quiet=False)


# ── Google Drive mount ────────────────────────────────────────────────────────

def use_gdrive_mount(mount_path: str, raw_dir: str):
    """
    Use data already present in a mounted Google Drive folder.
    Expects mount_path to contain Odia_train/ and Odia_test/ subdirectories.
    Creates symlinks in data/raw/ pointing to the Drive folders.
    """
    mount_path = os.path.expanduser(mount_path)
    if not os.path.isdir(mount_path):
        raise SystemExit(f"Google Drive mount path not found: {mount_path}")

    for folder in ("Odia_train", "Odia_test"):
        src = os.path.join(mount_path, folder)
        dst = os.path.join(raw_dir, folder)
        if not os.path.exists(src):
            print(f"  WARNING: {src} not found in mount path — skipping.")
            continue
        if os.path.exists(dst):
            print(f"  Already linked/present: {dst}")
            continue
        os.symlink(src, dst)
        print(f"  Linked: {dst} -> {src}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download Odia ASR dataset")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--gdrive", action="store_true",
        help="Download .tar.gz files from Google Drive using --train-id / --test-id"
    )
    source.add_argument(
        "--gdrive-mount", metavar="PATH",
        help="Path to a mounted Google Drive folder that already contains "
             "Odia_train/ and Odia_test/ subdirectories"
    )
    parser.add_argument("--train-id", metavar="FILE_ID",
                        help="Google Drive file ID for Odia_train.tar.gz")
    parser.add_argument("--test-id",  metavar="FILE_ID",
                        help="Google Drive file ID for Odia_test.tar.gz")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    raw_dir = os.path.join(DATA_DIR, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # ── Option 3: mounted Drive ───────────────────────────────────────────────
    if args.gdrive_mount:
        print(f"\nUsing mounted Google Drive: {args.gdrive_mount}")
        use_gdrive_mount(args.gdrive_mount, raw_dir)
        print("\nDone. Data in:", raw_dir)
        print("Next step: python prepare_dataset.py")
        return

    # ── Option 2: download from Drive ─────────────────────────────────────────
    if args.gdrive:
        if not args.train_id or not args.test_id:
            parser.error("--gdrive requires both --train-id and --test-id")
        gdrive_files = {
            "Odia_train.tar.gz": args.train_id,
            "Odia_test.tar.gz":  args.test_id,
        }
        for filename, file_id in gdrive_files.items():
            dest = os.path.join(raw_dir, filename)
            print(f"\n[{filename}]")
            download_from_gdrive(file_id, dest)
            extract_dir = os.path.join(raw_dir, filename.replace(".tar.gz", ""))
            if not os.path.exists(extract_dir):
                extract(dest, raw_dir)
            else:
                print(f"  Already extracted: {extract_dir}")
        print("\nDone. Data in:", raw_dir)
        print("Next step: python prepare_dataset.py")
        return

    # ── Option 1: OpenSLR (default) ───────────────────────────────────────────
    for filename, url in URLS.items():
        dest = os.path.join(raw_dir, filename)
        mirror = MIRRORS.get(filename)
        print(f"\n[{filename}]")
        download_file(url, dest, mirror)
        extract_dir = os.path.join(raw_dir, filename.replace(".tar.gz", ""))
        if not os.path.exists(extract_dir):
            extract(dest, raw_dir)
        else:
            print(f"  Already extracted: {extract_dir}")

    print("\nDone. Data in:", raw_dir)
    print("Next step: python prepare_dataset.py")


if __name__ == "__main__":
    main()
