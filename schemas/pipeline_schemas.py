import os
import json
from enum import Enum, IntEnum
from typing import Optional, Tuple, Union

from pydantic import BaseModel, Field, model_validator

# (X,Y,Z) = (width, height, depth)
NIFTI_SIZE = (182, 218)
DEF_RESIZE = (320, 320)
ALL_STEPS = ["preprocess", "train", "predict", "evaluate"]

# Example default values for GCS bucket and destination path
DEFAULT_BUCKET = "tfm-training-results"
DEFAULT_DEST_PATH = "results/archive.tar.gz"


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
    TOP_FIVE = "top_five"


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


class PreprocessConfig(BaseModel):
    """Configuration for preprocessing data."""
    net: Net
    src_path: str
    dst_path: str
    super_scale: SuperScale
    resize: Tuple[int, int] = Field(..., description="Resize dimensions (width,height)")
    strategy: Strategy
    split: float = Field(0.8, ge=0.0, le=1.0)
    threshold: Optional[int] = Field(None, description="Threshold for lesion detection, if applicable")
    resize_method: Optional[ResizeMethod] = None

    def write_config(self) -> None:
        """Write the configuration to a JSON file."""
        with open(os.path.join(self.dst_path, "preprocess_params.json"), "w") as f:
            json.dump(self.model_dump(), f, indent=4)

class TrainConfig(BaseModel):
    """Configuration for training a model."""
    net: Net
    dst_path: str
    src_path: str
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 1e-3
    limit_resources: bool = False
    use_kfold: bool = False
    kfold_n_splits: int = 5
    kfold_seed: int = 42

    def write_config(self) -> None:
        """Write the training configuration to a JSON file."""
        with open(os.path.join(self.dst_path, "train_params.json"), "w") as f:
            json.dump(self.model_dump(), f, indent=4)


class EvaluateConfig(BaseModel):
    """Configuration for evaluating a trained model."""
    net: Net
    model_path: str
    src_path: str
    pred_path: str
    gt_path: str

    def write_config(self) -> None:
        """Write the evaluation configuration to a JSON file."""
        with open(os.path.join(self.src_path, "evaluate_params.json"), "w") as f:
            json.dump(self.model_dump(), f, indent=4)


class PredictConfig(BaseModel):
    """Configuration for making predictions."""
    net: Net
    model_path: str
    src_path: str
    dst_path: str

    def write_config(self) -> None:
        """Write the prediction configuration to a JSON file."""
        with open(os.path.join(self.dst_path, "predict_params.json"), "w") as f:
            json.dump(self.model_dump(), f, indent=4)

class PipelineConfig(BaseModel):
    """Configuration for the entire pipeline."""
    step: str
    preprocess_config: Optional[PreprocessConfig] = None
    train_config: Optional[TrainConfig] = None
    evaluate_config: Optional[EvaluateConfig] = None
    predict_config: Optional[PredictConfig] = None

    @model_validator(mode="after")
    def validate_single_config(cls, values):
        configs = [
            values.preprocess_config,
            values.train_config,
            values.evaluate_config,
            values.predict_config,
        ]
        if sum(config is not None for config in configs) != 1:
            raise ValueError("Exactly one configuration option must be provided.")
        return values

    def active_config(
        self,
    ) -> Union[PreprocessConfig, TrainConfig, EvaluateConfig, PredictConfig]:
        return (
            self.preprocess_config
            or self.train_config
            or self.evaluate_config
            or self.predict_config
        )


class SegmentationMetrics(BaseModel):
    """Metrics for evaluating segmentation performance."""
    iou: float
    dice_score: float
    precision: float
    recall: float
    f1_score: float
    specificity: Optional[float] = None
    inference_time: Optional[float] = None

    def __str__(self):
        return (
            f"IoU: {self.iou:.4f}, Dice: {self.dice_score:.4f}, "
            f"Precision: {self.precision:.4f}, Recall: {self.recall:.4f}, F1: {self.f1_score:.4f}, "
            f"Specificity: {self.specificity if self.specificity else 'N/A'}, "
            f"Inference Time: {self.inference_time if self.inference_time else 'N/A'}s"
        )
