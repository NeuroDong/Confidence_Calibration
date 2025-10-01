"""
# -*- coding: utf-8 -*-
# @Time    : 2025/10/01 16:26 
# @File    : Logit_Datasets_for_Neural_Networks.py
# Reference github: "https://github.com/markus93/NN_calibration"
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional
import pickle


# Google Drive folder shared by the user
GDRIVE_FOLDER = "https://drive.google.com/drive/folders/1M7TWT1_ahCiSNoH3uPuOP1EJ5chJZsgy?usp=sharing"

# Optional list of GitHub raw base URLs to try as a fallback. Each should be a
# URL prefix that when joined with the filename yields a direct download link
# to the .p file (for example, a GitHub raw URL). Users can override or add
# bases via the environment variable CONF_CAL_GITHUB_RAW (semicolon-separated).
import os

_env_bases = os.environ.get("CONF_CAL_GITHUB_RAW")
if _env_bases:
    GITHUB_RAW_BASES = [b for b in _env_bases.split(";") if b]
else:
    GITHUB_RAW_BASES = [
        # Example placeholder (modify if you host the files on GitHub or other raw hosts)
        "https://raw.githubusercontent.com/username/repo/main/data/Logit_Datasets_for_Neural_Networks",
    ]


def get_cache_dir() -> Path:
    """Return the Path to the cache directory in the user's home folder."""
    home = Path.home()
    cache_dir = home / ".confidence_calibration" / "Logit_Datasets_for_Neural_Networks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _format_bytes(num: int) -> str:
    """Return a human-readable string for a byte count."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PB"


def _urlretrieve_with_progress(url: str, dest: Path) -> None:
    """Download a URL to dest while printing progress to stdout.

    Uses urllib.request.urlretrieve's reporthook to show percent and bytes.
    """
    dest = Path(dest)
    # Ensure parent dir exists
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {url} -> {dest}")

    def _reporthook(block_num, block_size, total_size):
        if total_size <= 0:
            # Unknown total size
            downloaded = block_num * block_size
            print(f"Downloaded {_format_bytes(downloaded)}", end="\r")
            return
        downloaded = min(total_size, block_num * block_size)
        pct = downloaded * 100.0 / total_size
        print(f"Downloaded {pct:5.1f}% ({_format_bytes(downloaded)}/{_format_bytes(total_size)})", end="\r")

    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=_reporthook)
        # make sure to finish the progress line
        print()
    except Exception:
        # On failure, ensure any partial file is removed
        try:
            if dest.exists():
                dest.unlink()
        except Exception:
            pass
        raise


def _download_from_gdrive_folder(dest: Path, folder_url: str) -> None:
    """Download all files from a Google Drive folder into dest.

    Prefers the python `gdown` package (download_folder). If it's not
    available, tries the `gdown` CLI with `--folder`. Raises RuntimeError if
    neither is available or download fails.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    # Try python gdown first
    try:
        import gdown
        # gdown.download_folder accepts either the folder id or the full URL
        print("Using python gdown to download Google Drive folder (shows its own progress)...")
        gdown.download_folder(folder_url, output=str(dest), quiet=False)
        return
    except Exception:
        # Fall back to CLI gdown
        pass

    try:
        subprocess.check_call(["gdown", "--folder", folder_url, "--output", str(dest)])
        return
    except FileNotFoundError:
        raise RuntimeError("gdown is required to download from Google Drive (install python package 'gdown' or CLI)")
    except Exception as e:
        raise RuntimeError("Failed to download from Google Drive using gdown") from e


def get_logit_datasets(local_only: bool = False) -> Path:
    """Ensure the cache directory contains the dataset .p files.

    Behavior now:
      - If the cache directory already contains at least one non-empty `.p` file,
        return the cache directory path.
      - If not present and `local_only` is False, download the entire Google
        Drive folder into the cache directory (using `gdown` if available),
        then return the cache directory path.

    Args:
        local_only: if True, do not attempt network downloads; raise if cache
            is empty.

    Returns:
        Path to the cache directory containing .p files (may be empty if
        local_only and nothing found — in that case an error is raised).

    Raises:
        FileNotFoundError: if no .p files are present locally and `local_only`
            is True or download did not yield any .p files.
        RuntimeError/HTTPError/URLError: when download fails due to network/server issues.
    """

    cache_dir = get_cache_dir()

    # If any .p files exist already, return the cache dir
    existing = list(cache_dir.glob("*.p"))
    if any(p.stat().st_size > 0 for p in existing):
        return cache_dir

    if local_only:
        raise FileNotFoundError(f"No .p files found in cache {cache_dir} and local_only=True")

    # Attempt to download the entire Google Drive folder into cache_dir
    try:
        _download_from_gdrive_folder(cache_dir, GDRIVE_FOLDER)
    except Exception as e:
        # Fall back to raw hosts: try to download all filenames listed under
        # GITHUB_RAW_BASES by trying to fetch a small index or common names is
        # non-trivial; we will attempt to download any of a small known set of
        # filename patterns if you have them hosted. Since we don't have an
        # authoritative list here, re-raise the exception with an explanatory
        # message.
        raise RuntimeError("Failed to download Google Drive folder; ensure 'gdown' is installed or set CONF_CAL_GITHUB_RAW to a valid raw host") from e

    # Re-check for .p files after download
    existing = list(cache_dir.glob("*.p"))
    if any(p.stat().st_size > 0 for p in existing):
        return cache_dir

    raise FileNotFoundError(f"Download completed but no .p files were found in {cache_dir}")

def unpickle_probs(file, verbose = 0):
    with open(file, 'rb') as f:  # Python 3: open(..., 'rb')
        (y_probs_val, y_val), (y_probs_test, y_test) = pickle.load(f)  # unpickle the content
        
    if verbose:    
        print("y_probs_val:", y_probs_val.shape)  # (5000, 10); Validation set probabilities of predictions
        print("y_true_val:", y_val.shape)  # (5000, 1); Validation set true labels
        print("y_probs_test:", y_probs_test.shape)  # (10000, 10); Test set probabilities
        print("y_true_test:", y_test.shape)  # (10000, 1); Test set true labels
        
    return ((y_probs_val, y_val), (y_probs_test, y_test))

if __name__ == "__main__":
    # Demo: ensure cache contains .p files (this may attempt network downloads)
    cache = get_logit_datasets()
    files = list(cache.glob('*.p'))
    print(f"Cache directory: {cache}")
    print(f"Found {len(files)} .p files")
    for f in files:
        print(" -", f.name)

    # unpickle .p file if available
    file_path = os.path.join(cache, f.name)
    (y_probs_val, y_val), (y_probs_test, y_test) = unpickle_probs(f, True)
    print(y_probs_val[:10])
