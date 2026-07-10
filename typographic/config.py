from pathlib import Path

# ===============================
# Project Paths
# ===============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "datasets" / "SROIE" / "SROIE2019"

TRAIN_IMAGES = DATASET_PATH / "train" / "img"

TEST_IMAGES = DATASET_PATH / "test" / "img"

OUTPUT_PATH = PROJECT_ROOT / "outputs"

OCR_OUTPUT = OUTPUT_PATH / "ocr"

VISUALIZATION_OUTPUT = OUTPUT_PATH / "visualization"

FEATURE_OUTPUT = OUTPUT_PATH / "feature_vectors"

# Create output folders automatically
OCR_OUTPUT.mkdir(parents=True, exist_ok=True)
VISUALIZATION_OUTPUT.mkdir(parents=True, exist_ok=True)
FEATURE_OUTPUT.mkdir(parents=True, exist_ok=True)

ATTACK_OUTPUT = OUTPUT_PATH / "attacked_documents"

METADATA_OUTPUT = OUTPUT_PATH / "metadata"

ATTACK_OUTPUT.mkdir(parents=True, exist_ok=True)

METADATA_OUTPUT.mkdir(parents=True, exist_ok=True)