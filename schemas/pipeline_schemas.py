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
    UNET = "unet"
    YOLO = "yolo"


class SuperScale(IntEnum):
    NONE = 1
    TWO = 2
    THREE = 3
    EIGHT = 8


class ResizeMethod(str, Enum):
    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"


class Strategy(str, Enum):
    ALL_SLICES = "all_slices"
    LESION = "lesion_slices"
    TOP_FIVE = "top_five"


class EnvConfig(BaseModel):
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
    net: Net
    src_path: str
    dst_path: str
    super_scale: SuperScale
    resize: Tuple[int, int] = Field(..., description="Resize dimensions (width,height)")
    strategy: Strategy
    split: float = Field(0.8, ge=0.0, le=1.0)
    threshold: Optional[int] = Field(None, description="Threshold for lesion detection, if applicable")
    resize_method: Optional[ResizeMethod] = None


class TrainConfig(BaseModel):
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


class EvaluateConfig(BaseModel):
    net: Net
    model_path: str
    src_path: str
    pred_path: str
    gt_path: str


class PredictConfig(BaseModel):
    net: Net
    model_path: str
    src_path: str
    dst_path: str


class PipelineConfig(BaseModel):
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
