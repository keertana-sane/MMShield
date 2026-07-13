"""
MMShield — Typographic Module Configuration

Centralizes every filesystem path used by the typographic attack /
detection pipeline (dataset location, generated attack images, feature
vectors, OCR/visualization caches, and metadata).

Path resolution contract
-------------------------
By default, the project root is resolved as two directories above this
file (i.e. this file is expected to live at ``<project_root>/<some_dir>/
config.py``). This can be overridden without touching source code by
setting the ``MMSHIELD_ROOT`` environment variable, e.g.:

    export MMSHIELD_ROOT=/data/mmshield

This is useful for reproducibility across machines (different clone
locations, CI runners, shared compute, etc.) without needing to edit or
fork this file.

All output directories are created on import (``mkdir(parents=True,
exist_ok=True)``) so downstream scripts can assume they exist.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

__all__ = [
    "PROJECT_ROOT",
    "DATASET_PATH",
    "TRAIN_IMAGES",
    "TEST_IMAGES",
    "OUTPUT_PATH",
    "OCR_OUTPUT",
    "VISUALIZATION_OUTPUT",
    "FEATURE_OUTPUT",
    "ATTACK_OUTPUT",
    "METADATA_OUTPUT",
]

# ===============================
# Project Paths
# ===============================

_env_root = os.environ.get("MMSHIELD_ROOT")

PROJECT_ROOT: Path = (
    Path(_env_root).expanduser().resolve()
    if _env_root
    else Path(__file__).resolve().parent.parent
)

DATASET_PATH: Path = PROJECT_ROOT / "datasets" / "SROIE" / "SROIE2019"

TRAIN_IMAGES: Path = DATASET_PATH / "train" / "img"

TEST_IMAGES: Path = DATASET_PATH / "test" / "img"

OUTPUT_PATH: Path = PROJECT_ROOT / "outputs"

OCR_OUTPUT: Path = OUTPUT_PATH / "ocr"

VISUALIZATION_OUTPUT: Path = OUTPUT_PATH / "visualization"

FEATURE_OUTPUT: Path = OUTPUT_PATH / "feature_vectors"

ATTACK_OUTPUT: Path = OUTPUT_PATH / "attacked_documents"

METADATA_OUTPUT: Path = OUTPUT_PATH / "metadata"

_OUTPUT_DIRS = (
    OCR_OUTPUT,
    VISUALIZATION_OUTPUT,
    FEATURE_OUTPUT,
    ATTACK_OUTPUT,
    METADATA_OUTPUT,
)


def _ensure_output_dirs() -> None:
    """
    Creates every required output directory, raising a clear,
    actionable error instead of letting a raw OSError/PermissionError
    propagate from an arbitrary downstream import.
    """

    for directory in _OUTPUT_DIRS:

        try:

            directory.mkdir(parents=True, exist_ok=True)

        except OSError as exc:

            raise OSError(
                f"MMShield could not create required output directory "
                f"'{directory}'. Check filesystem permissions, or set the "
                f"MMSHIELD_ROOT environment variable to a writable "
                f"location. Original error: {exc}"
            ) from exc


def _warn_if_dataset_missing() -> None:
    """
    Logs a non-fatal warning if the expected SROIE training images are
    not present. Scripts that don't need the dataset (e.g. predict.py
    on a single external image) should still be able to import this
    module successfully, so this intentionally does not raise.
    """

    if not TRAIN_IMAGES.exists():

        logger.warning(
            "TRAIN_IMAGES path does not exist: %s\n"
            "Dataset-building scripts will silently produce 0 samples "
            "until the SROIE dataset is placed here, or MMSHIELD_ROOT "
            "is set to point at a valid dataset location.",
            TRAIN_IMAGES,
        )


_ensure_output_dirs()
_warn_if_dataset_missing()

logger.info("MMShield project root resolved to: %s", PROJECT_ROOT)