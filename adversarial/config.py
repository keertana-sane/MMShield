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


@dataclass(frozen=True)
class PathConfig:
    """
    Filesystem layout for the Patch Integrity Module.

    All paths are resolved relative to PROJECT_ROOT. Directories are not
    created here; each module is responsible for creating the directories
    it writes to (typically via `Path.mkdir(parents=True, exist_ok=True)`)
    at the point of first use.
    """

    project_root: Path = PROJECT_ROOT

    # Raw source dataset (SROIE receipts).
    sroie_dir = PROJECT_ROOT / "datasets" / "SROIE" / "SROIE2019" / "train" / "img"

    # Library of transparent PNG patches used as visual modifications.
    patch_library_dir = PROJECT_ROOT / "patches"

    # Synthetic attack generation outputs.
    generated_attacks_dir: Path = PROJECT_ROOT / "generated_attacks"
    generated_attacks_images_dir: Path = PROJECT_ROOT / "generated_attacks" / "images"
    generated_attacks_metadata_csv: Path = (
        PROJECT_ROOT / "generated_attacks" / "metadata.csv"
    )

    # Candidate region proposal outputs.
    candidates_dir: Path = PROJECT_ROOT / "candidates"
    candidates_patches_dir: Path = PROJECT_ROOT / "candidates" / "patches"
    candidates_csv: Path = PROJECT_ROOT / "candidates" / "candidates.csv"

    # Extracted feature outputs.
    features_dir: Path = PROJECT_ROOT / "features"
    train_features_csv: Path = PROJECT_ROOT / "features" / "train_features.csv"
    validation_features_csv: Path = PROJECT_ROOT / "features" / "validation_features.csv"
    test_features_csv: Path = PROJECT_ROOT / "features" / "test_features.csv"

    # Dataset split outputs (features + metadata + labels merged).
    splits_dir: Path = PROJECT_ROOT / "features"
    train_csv: Path = PROJECT_ROOT / "features" / "train.csv"
    validation_csv: Path = PROJECT_ROOT / "features" / "validation.csv"
    test_csv: Path = PROJECT_ROOT / "features" / "test.csv"

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

    def all_output_directories(self) -> List[Path]:
        """
        Return every directory that a pipeline stage may need to create
        before writing files into it.

        Returns:
            List[Path]: Directories required across the full pipeline.
        """
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
        ]


@dataclass(frozen=True)
class RandomSeedConfig:
    """Global random seed configuration for full reproducibility."""

    seed: int = 42


@dataclass(frozen=True)
class ImageConfig:
    """Image sizing and I/O settings shared across the pipeline."""

    # Standard size (width, height) that images are resized to before
    # candidate generation and feature extraction.
    target_size: Tuple[int, int] = (512, 512)

    # Size fed into the CNN feature extractor.
    cnn_input_size: Tuple[int, int] = (224, 224)

    # Valid image file extensions considered when scanning source folders.
    valid_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    # JPEG quality used when re-saving compressed outputs.
    jpeg_quality: int = 95


@dataclass(frozen=True)
class PatchGenerationConfig:
    """
    Settings controlling the synthetic patch dataset generator
    (attack_generator.py).

    These parameters govern the random transformation ranges applied to
    each patch composited onto a clean receipt image. This module performs
    localized visual modification synthesis only; it does not implement
    any adversarial optimization procedure.
    """

    # Number of patched (positive) images generated per clean receipt.
    patched_images_per_receipt: int = 5

    # Relative patch size range, expressed as a fraction of the shorter
    # edge of the base image (min_scale, max_scale).
    patch_scale_range: Tuple[float, float] = (0.08, 0.35)

    # Rotation range in degrees applied to the patch before compositing.
    rotation_range_degrees: Tuple[float, float] = (-30.0, 30.0)

    # Opacity range for alpha blending the patch onto the base image.
    opacity_range: Tuple[float, float] = (0.55, 1.0)

    # Brightness adjustment factor range (1.0 = unchanged).
    brightness_range: Tuple[float, float] = (0.7, 1.3)

    # Contrast adjustment factor range (1.0 = unchanged).
    contrast_range: Tuple[float, float] = (0.7, 1.3)

    # Perspective warp distortion magnitude range, expressed as a fraction
    # of the patch's own width/height used to jitter its corner points.
    perspective_warp_range: Tuple[float, float] = (0.0, 0.15)

    # Gaussian blur kernel size options (must be odd integers); 0 means
    # no blur applied.
    gaussian_blur_kernel_choices: Tuple[int, ...] = (0, 3, 5, 7)

    # Severity levels used to bucket the combined strength of the applied
    # transformations into a categorical label stored in metadata.
    severity_levels: Tuple[str, ...] = ("low", "medium", "high")

    # Attack type labels describing the category of synthetic patch used.
    # These are descriptive labels for the benchmark, not references to
    # any specific adversarial optimization algorithm.
    attack_type_labels: Tuple[str, ...] = (
        "overlay_patch",
        "corner_patch",
        "edge_patch",
        "occlusion_patch",
    )

    # Accepted patch library file extension.
    patch_file_extension: str = ".png"


@dataclass(frozen=True)
class CandidateGenerationConfig:
    """
    Settings for candidate_generator.py, which locates visually suspicious
    regions in a single image without access to a clean reference image.
    """

    # Number of top-ranked candidate regions retained per image.
    top_k_candidates: int = 3

    # Sliding-window / connected-component minimum area (in pixels) for a
    # region to be considered a valid candidate.
    min_candidate_area: int = 400

    # Maximum number of connected components considered before ranking,
    # to bound computation on noisy score maps.
    max_components_considered: int = 50

    # Local entropy filter window size (must be odd).
    entropy_window_size: int = 9

    # LBP (Local Binary Pattern) parameters.
    lbp_num_points: int = 24
    lbp_radius: int = 3
    lbp_method: str = "uniform"

    # Canny edge detection thresholds.
    canny_threshold_low: int = 50
    canny_threshold_high: int = 150

    # Sobel kernel size for gradient magnitude computation.
    sobel_kernel_size: int = 3

    # Weights combining individual score maps into the final threat score
    # map. Must sum to 1.0.
    score_weights: dict = field(
        default_factory=lambda: {
            "edge_density": 0.20,
            "entropy": 0.20,
            "lbp_texture": 0.20,
            "gradient_magnitude": 0.20,
            "saliency": 0.20,
        }
    )

    # Threshold applied to the normalized threat score map (0-1) before
    # connected component extraction.
    threat_score_threshold: float = 0.6

    # Padding (pixels) added around each candidate bounding box when
    # cropping the patch image for downstream feature extraction.
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

    # ImageNet normalization statistics used for CNN preprocessing.
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
    """Settings for dataset_builder.py (train/validation/test split)."""

    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    stratify_column: str = "label"
    label_column: str = "label"
    positive_label: int = 1
    negative_label: int = 0

    # Minimum IoU between a candidate region and the ground-truth patch
    # bounding box (from attack_generator metadata) for that candidate to
    # be assigned the positive label.
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

    # Metric used to select the best model among candidates.
    selection_metric: str = "f1"

    # Number of cross-validation folds used during model comparison.
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
# Other modules import these instances directly, e.g.:
#   from config import PATHS, IMAGE, PATCH_GEN
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
    print("All configured output directories have been created.")
