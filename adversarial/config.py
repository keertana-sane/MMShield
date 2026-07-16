"""
config.py

Central configuration module for the Patch Integrity Module of MMShield.

This module defines all paths, hyperparameters, thresholds, and settings
used throughout the pipeline: synthetic patch dataset generation, candidate
region proposal, feature extraction, model training, prediction, evaluation,
and visualization.

No other module should hardcode paths, thresholds, or settings. Everything
configurable lives here so that the entire pipeline stays consistent and
reproducible.

Multi-dataset support
----------------------
DATASETS is the single source of truth for every clean-document dataset
used by the patch module. Each entry provides:

    - clean_train : Path to the training-split image directory
    - clean_test  : Path to the test/eval-split image directory
    - image_glob  : Glob pattern used to enumerate images in that dir

PATCH_BENCHMARK_DIR is a separate, external benchmark used for
evaluation ONLY. It must never be used for training and is not part of
DATASETS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List


# ============================================================================
# PROJECT ROOT
# ============================================================================

# Root directory of the patch_module project. All other paths are derived
# relative to this location so the project is portable across machines.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================================
# MULTI-DATASET REGISTRY
# ============================================================================

DATASETS: dict = {
    "sroie": {
        "clean_train": PROJECT_ROOT / "datasets" / "SROIE" / "SROIE2019" / "train" / "img",
        "clean_test": PROJECT_ROOT / "datasets" / "SROIE" / "SROIE2019" / "test" / "img",
        "image_glob": "*.jpg",
    },
    "cord": {
        "clean_train": PROJECT_ROOT / "datasets" / "CORD" / "train" / "image",
        "clean_test": PROJECT_ROOT / "datasets" / "CORD" / "test" / "image",
        "image_glob": "*.png",
    },
    "funsd": {
        "clean_train": PROJECT_ROOT / "datasets" / "FUNSD" / "dataset" / "training_data" / "images",
        "clean_test": PROJECT_ROOT / "datasets" / "FUNSD" / "dataset" / "testing_data" / "images",
        "image_glob": "*.png",
    },
}

# ============================================================
# External Benchmark (Evaluation Only)
# ============================================================
# PatchBenchmark is an external benchmark used ONLY for
# evaluate_external.py. It is deliberately excluded from
# DATASETS so it is never used for training.
PATCH_BENCHMARK_DIR: Path = PROJECT_ROOT / "datasets" / "PatchBenchmark"
PATCH_BENCHMARK_IMAGES_DIR: Path = (
    PATCH_BENCHMARK_DIR / "benchmark_patches"
)
# Optional. If no labels exist, evaluate_external.py automatically
# switches to prediction-only mode.
PATCH_BENCHMARK_LABELS_CSV: Path = (
    PATCH_BENCHMARK_DIR / "labels.csv"
)


def get_dataset_images(dataset: str, split: str = "train") -> Path:
    """
    Returns the clean image directory for the requested dataset/split.

    Args:
        dataset: One of the keys in DATASETS ("sroie", "cord", "funsd").
        split: "train" or "test".

    Returns:
        Path to the requested clean image directory.

    Raises:
        ValueError: If dataset or split is not recognized.
    """
    dataset = dataset.lower()
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: '{dataset}'. Available: {list(DATASETS.keys())}")

    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")

    key = "clean_train" if split == "train" else "clean_test"
    return DATASETS[dataset][key]


@dataclass(frozen=True)
class PathConfig:
    """
    Filesystem layout for the Patch Integrity Module.

    All paths are resolved relative to PROJECT_ROOT. Helper methods provide
    dataset- and split-aware path resolution dynamically with strict parameter validation.
    """

    project_root: Path = PROJECT_ROOT

    # Multi-dataset registry. See module-level DATASETS for the source of truth.
    datasets: dict = field(default_factory=lambda: DATASETS)

    # External evaluation benchmark (never used for training)
    patch_benchmark_dir: Path = PATCH_BENCHMARK_DIR
    patch_benchmark_images_dir: Path = PATCH_BENCHMARK_IMAGES_DIR
    patch_benchmark_labels_csv: Path = PATCH_BENCHMARK_LABELS_CSV

    # Backward-compatible alias (SROIE train case)
    sroie_dir = DATASETS["sroie"]["clean_train"]

    # Library of transparent PNG patches used as visual modifications.
    patch_library_dir = PROJECT_ROOT / "patches"

    # Base directories for artifacts
    generated_attacks_dir: Path = PROJECT_ROOT / "generated_attacks"
    generated_attacks_images_dir: Path = PROJECT_ROOT / "generated_attacks" / "images"
    candidates_dir: Path = PROJECT_ROOT / "candidates"
    candidates_patches_dir: Path = PROJECT_ROOT / "candidates" / "patches"
    features_dir: Path = PROJECT_ROOT / "features"

    # Model artifacts.
    models_dir: Path = PROJECT_ROOT / "models"
    best_model_path: Path = PROJECT_ROOT / "models" / "best_model.pkl"
    model_metadata_json: Path = PROJECT_ROOT / "models" / "model_metadata.json"

    # Prediction / evaluation / visualization outputs.
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    predictions_csv: Path = PROJECT_ROOT / "outputs" / "predictions.csv"
    confusion_matrix_png: Path = PROJECT_ROOT / "outputs" / "confusion_matrix.png"
    roc_curve_png: Path = PROJECT_ROOT / "outputs" / "roc_curve.png"
    pr_curve_png: Path = PROJECT_ROOT / "outputs" / "pr_curve.png"
    classification_report_txt: Path = (
        PROJECT_ROOT / "outputs" / "classification_report.txt"
    )
    heatmaps_dir: Path = PROJECT_ROOT / "outputs" / "heatmaps"
    annotated_dir: Path = PROJECT_ROOT / "outputs" / "annotated"

    # Logging.
    logs_dir: Path = PROJECT_ROOT / "logs"
    log_file: Path = PROJECT_ROOT / "logs" / "patch_module.log"

    # ========================================================================
    # Private Validation Helpers
    # ========================================================================

    def _validate_inputs(self, dataset: str, split: str) -> Tuple[str, str]:
        """Internal utility to ensure path variables belong to runtime limits."""
        ds_lower = dataset.lower()
        sp_lower = split.lower()
        if ds_lower not in DATASETS:
            raise ValueError(f"Unknown dataset '{dataset}'. Registered choices: {list(DATASETS.keys())}")
        if sp_lower not in ("train", "test"):
            raise ValueError(f"Invalid split '{split}'. Expected 'train' or 'test'.")
        return ds_lower, sp_lower

    # ========================================================================
    # Dynamic Split-Aware Dataset Path Helpers
    # ========================================================================

    def get_generated_attacks_dir(self, dataset: str, split: str) -> Path:
        """Returns the specific subfolder where attacked variant images are stored."""
        ds, sp = self._validate_inputs(dataset, split)
        return self.generated_attacks_images_dir / ds / sp

    def get_generated_attacks_metadata_csv(self, dataset: str, split: str) -> Path:
        """Returns the metadata CSV path mapping transformations for a single split."""
        ds, sp = self._validate_inputs(dataset, split)
        return self.generated_attacks_dir / f"{ds}_{sp}_metadata.csv"

    def get_candidates_csv(self, dataset: str, split: str) -> Path:
        """Returns the candidate bounding box proposal mapping CSV."""
        ds, sp = self._validate_inputs(dataset, split)
        return self.candidates_dir / f"candidates_{ds}_{sp}.csv"

    def get_receipt_features_csv(self, dataset: str, split: str) -> Path:
        """Returns the handcrafted + integrated structural feature vector CSV."""
        ds, sp = self._validate_inputs(dataset, split)
        return self.features_dir / f"receipt_features_{ds}_{sp}.csv"

    def get_cnn_embeddings_dir(self, dataset: str, split: str) -> Path:
        """Returns the storage location for precomputed batch .npy deep embeddings."""
        ds, sp = self._validate_inputs(dataset, split)
        return self.features_dir / "cnn_embeddings" / ds / sp

    def get_combined_dataset_csv(self, split: str) -> Path:
        """Returns the multi-dataset combined file path consumed during training/testing."""
        sp_lower = split.lower()
        if sp_lower not in ("train", "test"):
            raise ValueError(f"Invalid split '{split}'. Expected 'train' or 'test'.")
        return self.features_dir / f"patch_dataset_combined_{sp_lower}.csv"

    def all_output_directories(self) -> List[Path]:
        """Return every directory that a pipeline stage may need to create before writing."""
        return [
            self.patch_library_dir,
            self.generated_attacks_dir,
            self.generated_attacks_images_dir,
            self.candidates_dir,
            self.candidates_patches_dir,
            self.features_dir,
            self.models_dir,
            self.outputs_dir,
            self.heatmaps_dir,
            self.annotated_dir,
            self.logs_dir,
            self.patch_benchmark_dir,
            self.patch_benchmark_images_dir,
        ]


@dataclass(frozen=True)
class RandomSeedConfig:
    """Global random seed configuration for full reproducibility."""

    seed: int = 42


@dataclass(frozen=True)
class ImageConfig:
    """Image sizing and I/O settings shared across the pipeline."""

    target_size: Tuple[int, int] = (512, 512)
    cnn_input_size: Tuple[int, int] = (224, 224)
    valid_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    jpeg_quality: int = 95


@dataclass(frozen=True)
class PatchGenerationConfig:
    """Settings controlling the synthetic patch dataset generator."""

    patched_images_per_receipt: int = 5
    patch_scale_range: Tuple[float, float] = (0.08, 0.35)
    rotation_range_degrees: Tuple[float, float] = (-30.0, 30.0)
    opacity_range: Tuple[float, float] = (0.55, 1.0)
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    contrast_range: Tuple[float, float] = (0.7, 1.3)
    perspective_warp_range: Tuple[float, float] = (0.0, 0.15)
    gaussian_blur_kernel_choices: Tuple[int, ...] = (0, 3, 5, 7)
    severity_levels: Tuple[str, ...] = ("low", "medium", "high")
    attack_type_labels: Tuple[str, ...] = (
        "overlay_patch",
        "corner_patch",
        "edge_patch",
        "occlusion_patch",
    )
    patch_file_extension: str = ".png"


@dataclass(frozen=True)
class CandidateGenerationConfig:
    """Settings for candidate_generator.py region mapping."""

    top_k_candidates: int = 3
    min_candidate_area: int = 400
    max_components_considered: int = 50
    entropy_window_size: int = 9
    lbp_num_points: int = 24
    lbp_radius: int = 3
    lbp_method: str = "uniform"
    canny_threshold_low: int = 50
    canny_threshold_high: int = 150
    sobel_kernel_size: int = 3
    score_weights: dict = field(
        default_factory=lambda: {
            "edge_density": 0.20,
            "entropy": 0.20,
            "lbp_texture": 0.20,
            "gradient_magnitude": 0.20,
            "saliency": 0.20,
        }
    )
    threat_score_threshold: float = 0.6
    candidate_padding: int = 8


@dataclass(frozen=True)
class CNNConfig:
    """Settings for feature_extractor.py (CNN embedding extraction)."""

    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    embedding_dim: int = 1280
    batch_size: int = 32
    num_workers: int = 4
    device_preference: Tuple[str, ...] = ("mps", "cuda", "cpu")
    normalization_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalization_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class HandcraftedFeatureConfig:
    """Settings for receipt_features.py (handcrafted feature computation)."""

    edge_density_bins: int = 1
    entropy_window_size: int = 9
    lbp_num_points: int = 24
    lbp_radius: int = 3
    lbp_histogram_bins: int = 26
    gradient_bins: int = 8
    color_histogram_bins: int = 32
    color_spaces: Tuple[str, ...] = ("RGB", "HSV")


@dataclass(frozen=True)
class DatasetSplitConfig:
    """Ground truth assignment parameters for multi-dataset split matching."""

    label_column: str = "label"
    positive_label: int = 1
    negative_label: int = 0
    stratify_column: str = "label"
    
    # Minimum IoU between a candidate region and the ground-truth patch
    # bounding box for that candidate to be assigned the positive label.
    label_iou_threshold: float = 0.3


@dataclass(frozen=True)
class TrainingConfig:
    """Settings for train.py (model training and comparison)."""

    candidate_models: Tuple[str, ...] = ("random_forest", "xgboost", "svm")
    random_forest_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "n_jobs": -1,
        }
    )
    xgboost_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
        }
    )
    svm_params: dict = field(
        default_factory=lambda: {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "probability": True,
        }
    )
    selection_metric: str = "f1"
    cv_folds: int = 5


@dataclass(frozen=True)
class EvaluationConfig:
    """Settings for evaluate.py."""

    decision_threshold: float = 0.5
    figure_dpi: int = 150
    figure_size: Tuple[int, int] = (6, 5)


@dataclass(frozen=True)
class VisualizationConfig:
    """Settings for visualize.py."""

    bounding_box_color: Tuple[int, int, int] = (255, 0, 0)
    bounding_box_thickness: int = 3
    heatmap_colormap: str = "JET"
    heatmap_alpha: float = 0.45
    font_scale: float = 0.6
    text_color: Tuple[int, int, int] = (255, 255, 255)
    text_background_color: Tuple[int, int, int] = (0, 0, 0)


@dataclass(frozen=True)
class FileNamingConfig:
    """Consistent file and column naming conventions used across modules."""

    clean_image_column: str = "clean_image"
    attacked_image_column: str = "attacked_image"
    patch_column: str = "patch"
    x_column: str = "x"
    y_column: str = "y"
    width_column: str = "width"
    height_column: str = "height"
    rotation_column: str = "rotation"
    opacity_column: str = "opacity"
    brightness_column: str = "brightness"
    contrast_column: str = "contrast"
    blur_column: str = "blur"
    severity_column: str = "severity"
    attack_type_column: str = "attack_type"
    image_id_column: str = "image_id"
    label_column: str = "label"
    candidate_rank_column: str = "candidate_rank"
    threat_score_column: str = "threat_score"
    bbox_columns: Tuple[str, ...] = ("bbox_x", "bbox_y", "bbox_width", "bbox_height")
    model_file_prefix: str = "model"
    best_model_filename: str = "best_model.pkl"


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration shared across all modules."""

    log_level: str = "INFO"
    log_format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_to_file: bool = True
    log_to_console: bool = True


# ============================================================================
# SINGLETON CONFIG INSTANCES
# ============================================================================

PATHS = PathConfig()
SEED = RandomSeedConfig()
IMAGE = ImageConfig()
PATCH_GEN = PatchGenerationConfig()
CANDIDATE_GEN = CandidateGenerationConfig()
CNN = CNNConfig()
HANDCRAFTED = HandcraftedFeatureConfig()
SPLIT = DatasetSplitConfig()
TRAINING = TrainingConfig()
EVALUATION = EvaluationConfig()
VISUALIZATION = VisualizationConfig()
NAMING = FileNamingConfig()
LOGGING = LoggingConfig()


def ensure_directories() -> None:
    """
    Create every output directory declared in PATHS if it does not already
    exist. Safe to call multiple times.
    """
    for directory in PATHS.all_output_directories():
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print(f"Project root: {PATHS.project_root}")
    print("All configured split-aware output directories have been verified.")