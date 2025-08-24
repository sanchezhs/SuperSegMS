import json
import hashlib

from enum import Enum, IntEnum
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, model_validator

# (X,Y,Z) = (width, height, depth)
NIFTI_SIZE = (182, 218)
DEF_RESIZE = (320, 320)
ALL_STEPS = ["preprocess", "train", "predict", "evaluate", "visualize"]

# Example default values for GCS bucket and destination path
DEFAULT_BUCKET = "tfm-training-results"
DEFAULT_DEST_PATH = Path("results")

class LinkMode(Enum):
    SYMLINK = "symlink"
    HARDLINK = "hardlink"

class Net(str, Enum):
    """Available neural network architectures."""
    UNET = "unet"
    YOLO = "yolo"


class SuperScale(IntEnum):
    """Super-resolution scaling factors."""
    NONE = 1
    TWO = 2
    THREE = 3
    EIGHT = 8


class ResizeMethod(str, Enum):
    """Methods for resizing images."""
    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"


class Strategy(str, Enum):
    """Strategy for handling slices in the dataset."""
    ALL_SLICES = "all_slices"
    LESION = "lesion_slices"
    LESION_BLOCK = "lesion_block"
    BRAIN = "brain"


class EnvConfig(BaseModel):
    """Environment configuration for the pipeline."""
    PHONE_NUMBER: Optional[str] = Field(
        None, description="WhatsApp phone number for notifications"
    )
    CMB_API_KEY: Optional[str] = Field(
        None, description="CallMeBot API key for WhatsApp notifications"
    )
    CREDENTIALS_PATH: Optional[str] = Field(
        None, description="Path to GCS credentials JSON file"
    )

class KFoldPrep(BaseModel):
    enable: bool = False
    n_splits: int = 5
    seed: int = 42
    mini_val_frac: float = 0.10
    link_mode: Literal["hardlink", "copy"] = "hardlink"

class Config(BaseModel):
    pass

class PreprocessConfig(Config):
    """Configuration for preprocessing data."""
    net: Net
    src_path: Path
    dst_path: Path
    super_scale: SuperScale
    resize: Tuple[int, int] = Field(..., description="Resize dimensions (width,height)")
    strategy: Strategy
    split: float = Field(0.8, ge=0.0, le=1.0)
    seed: int = Field(42, description="Random seed for reproducibility")
    block_size: Optional[int] = Field(5, description="Block size for lesion detection, if applicable")
    threshold: Optional[int] = Field(0, description="Threshold for lesion detection, if applicable")
    resize_method: Optional[ResizeMethod] = None
    kfold: Optional[KFoldPrep] = None

    def write_config(self) -> None:
        """Write the configuration to a JSON file."""
        with open(self.dst_path / "preprocess_params.json", "w") as f:
            f.write(self.model_dump_json(indent=4))

class TrainConfig(Config):
    """Configuration for training a model."""
    net: Net
    dst_path: Path
    src_path: Path
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 1e-3
    use_kfold: bool = False
    kfold_n_splits: int = 5
    kfold_seed: int = 42

    def write_config(self) -> None:
        """Write the training configuration to a JSON file."""
        with open(self.dst_path / "train_params.json", "w") as f:
            f.write(self.model_dump_json(indent=4))


class EvaluateConfig(Config):
    """Configuration for evaluating a trained model."""
    net: Net
    model_path: Path
    src_path: Path
    pred_path: Path
    gt_path: Path

    def write_config(self) -> None:
        """Write the evaluation configuration to a JSON file."""
        with open(self.src_path / "evaluate_params.json", "w") as f:
            f.write(self.model_dump_json(indent=4))

class PredictConfig(Config):
    """Configuration for making predictions."""
    net: Net
    model_path: Path
    src_path: Path
    dst_path: Path

    def write_config(self) -> None:
        """Write the prediction configuration to a JSON file."""
        with open(self.dst_path / "predict_params.json", "w") as f:
            f.write(self.model_dump_json(indent=4))

class VisualizeConfig(Config):
    """
    Config for the visualization step: reads metrics.json and renders plots.
    """
    net: Net
    pred_path: Path
    plots_dir: Optional[Path] = None         # default: pred_path.parent / "plots"
    plots_title: Optional[str] = None        # default: f"{net.name} Evaluation"
    plot_prefix: Optional[str] = None        # default: stem of model_path if provided
    model_path: Optional[Path] = None

    def write_config(self) -> None:
        (self.pred_path / "visualize_params.json").write_text(self.model_dump_json(indent=4))

class PipelineConfig(BaseModel):
    """Configuration for the entire pipeline."""
    experiment_id: str
    step: str
    preprocess_config: Optional[PreprocessConfig] = None
    train_config: Optional[TrainConfig] = None
    evaluate_config: Optional[EvaluateConfig] = None
    predict_config: Optional[PredictConfig] = None
    visualize_config: Optional[VisualizeConfig] = None

    @model_validator(mode="after")
    def validate_single_config(cls, values):
        configs = [
            values.preprocess_config,
            values.train_config,
            values.evaluate_config,
            values.predict_config,
            values.visualize_config,
        ]
        if sum(config is not None for config in configs) != 1:
            raise ValueError("Exactly one configuration option must be provided.")
        return values

    def active_config(
        self,
    ) -> Union[Optional[PreprocessConfig], Optional[TrainConfig], Optional[EvaluateConfig], Optional[PredictConfig], Optional[VisualizeConfig]]:
        return (
            self.preprocess_config
            or self.train_config
            or self.evaluate_config
            or self.predict_config
            or self.visualize_config
        )


class SegmentationMetrics(BaseModel):
    """Metrics for evaluating segmentation performance."""
    iou: float
    dice_score: float
    precision: float
    recall: float
    specificity: float
    inference_time: Optional[float] = None

    def __str__(self):
        return (
            f"IoU: {self.iou:.4f}, Dice: {self.dice_score:.4f}, "
            f"Precision: {self.precision:.4f}, Recall: {self.recall:.4f}, "
            f"Specificity: {self.specificity if self.specificity else 'N/A'}, "
            f"Inference Time: {self.inference_time if self.inference_time else 'N/A'}s"
        )
    
    def __repr__(self):
        return (
            f"SegmentationMetrics(iou={self.iou:.4f}, "
            f"dice_score={self.dice_score:.4f}, "
            f"precision={self.precision:.4f}, "
            f"recall={self.recall:.4f}, "
            f"specificity={self.specificity if self.specificity else 'N/A'}, "
            f"inference_time={self.inference_time if self.inference_time else 'N/A'})"
        )
    
    def write_to_file(self, path: Path) -> None:
        """Write metrics to a file.
        Args:
            - path: Path to the file where metrics will be saved.
        """
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=4))

class SliceStats(BaseModel):
    patient_id: str
    timepoint: str
    slice_idx: int
    split: str
    strategy: str
    preproc_signature: str
    # lesion geometry
    lesion_area_px: int
    lesion_ratio: float
    num_components: int
    largest_component_area_px: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    bbox_ratio: float
    # intensity
    mean_in: float
    std_in: float
    mean_out: float
    std_out: float
    contrast: float

