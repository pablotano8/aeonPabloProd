"""Automatic checkpoint bootstrap for local runs and deployment."""

from __future__ import annotations

import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

from checkpoint_source import CHECKPOINT_ARCHIVE_URL


REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
DOWNLOAD_PATH = REPO_ROOT / ".checkpoint_download.tmp.zip"
LOCK_PATH = REPO_ROOT / ".checkpoint_download.lock"
REQUIRED_FILES = [
    CHECKPOINTS_DIR / "segmentation" / "dataset.json",
    CHECKPOINTS_DIR / "segmentation" / "dataset_fingerprint.json",
    CHECKPOINTS_DIR / "segmentation" / "plans.json",
    CHECKPOINTS_DIR / "segmentation" / "fold_all" / "checkpoint_final.pth",
    CHECKPOINTS_DIR / "survival" / "coxph_model.pkl",
    CHECKPOINTS_DIR / "survival" / "scaler.pkl",
]


def checkpoints_present() -> bool:
    return all(path.exists() for path in REQUIRED_FILES)


def _acquire_lock(timeout_seconds: int = 900) -> int | None:
    start = time.time()
    while True:
        try:
            return os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if checkpoints_present():
                return None
            if time.time() - start > timeout_seconds:
                raise TimeoutError("Timed out waiting for another process to finish downloading checkpoints.")
            time.sleep(1)


def _release_lock(lock_fd: int | None) -> None:
    if lock_fd is None:
        return
    os.close(lock_fd)
    try:
        LOCK_PATH.unlink()
    except FileNotFoundError:
        pass


def _extract_archive(archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.namelist():
            target = REPO_ROOT / member
            if not str(target.resolve()).startswith(str(REPO_ROOT.resolve())):
                raise ValueError(f"Unsafe path in archive: {member}")
        zf.extractall(REPO_ROOT)


def ensure_checkpoints(force: bool = False, verbose: bool = True) -> None:
    if checkpoints_present() and not force:
        if verbose:
            print("All required checkpoint files are already present.", flush=True)
        return

    if not CHECKPOINT_ARCHIVE_URL:
        raise RuntimeError(
            "Checkpoint archive URL is not configured. Upload checkpoints.zip to Google Drive "
            "and set CHECKPOINT_ARCHIVE_URL in src/checkpoint_source.py."
        )

    lock_fd = _acquire_lock()
    if lock_fd is None:
        if verbose:
            print("Checkpoint download was completed by another process.", flush=True)
        return

    try:
        if checkpoints_present() and not force:
            return

        if force and CHECKPOINTS_DIR.exists():
            shutil.rmtree(CHECKPOINTS_DIR)

        if verbose:
            print("Downloading checkpoint archive...", flush=True)
        if DOWNLOAD_PATH.exists():
            DOWNLOAD_PATH.unlink()

        import gdown

        downloaded = gdown.download(
            url=CHECKPOINT_ARCHIVE_URL,
            output=str(DOWNLOAD_PATH),
            quiet=not verbose,
            fuzzy=True,
        )
        if not downloaded or not DOWNLOAD_PATH.exists():
            raise RuntimeError("Checkpoint download failed.")

        if verbose:
            print("Extracting checkpoint archive...", flush=True)
        _extract_archive(DOWNLOAD_PATH)

        missing = [str(path.relative_to(REPO_ROOT)) for path in REQUIRED_FILES if not path.exists()]
        if missing:
            joined = "\n".join(f"  - {path}" for path in missing)
            raise RuntimeError(f"Checkpoint bootstrap finished, but required files are still missing:\n{joined}")

        if verbose:
            print("Checkpoint bootstrap complete.", flush=True)
    finally:
        try:
            if DOWNLOAD_PATH.exists():
                DOWNLOAD_PATH.unlink()
        finally:
            _release_lock(lock_fd)


def main() -> int:
    force = "--force" in sys.argv[1:]
    ensure_checkpoints(force=force, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
