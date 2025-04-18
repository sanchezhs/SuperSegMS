from typing import Optional
from dataclasses import dataclass, asdict
from enum import IntEnum, Enum
from loguru import logger

# (X,Y,Z) = (width, height, depth)
NIFTI_SIZE = (182, 218)  # width, height
DEF_RESIZE = (320, 320)  # width, height

class Net(Enum):
    UNET = "unet"
    YOLO = "yolo"

class SuperScale(IntEnum):
    NONE = 1
    TWO = 2
    THREE = 3
    EIGHT = 8

    def __str__(self):
        return str(self.value)

class ResizeMethod(Enum):
    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"


@dataclass
class PreprocessConfig:
    net: Net
    src_path: str
    dst_path: str
    super_scale: SuperScale
    resize: tuple[int, int]
    top_slices: int
    split: float = 0.8
    resize_method: Optional[ResizeMethod] = None

    def as_dict(self) -> dict:
        """Convert the PreprocessConfig to a dictionary."""
        d = asdict(self)
        d["net"] = self.net.value
        return d

@dataclass
class TrainConfig:
    net: Net
    dst_path: str
    src_path: str
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 1e-3
    limit_resources: bool = False

    def as_dict(self) -> dict:
        """Convert the TrainConfig to a dictionary."""
        d = asdict(self)
        d["net"] = self.net.value
        return d

@dataclass
class EvaluateConfig:
    net: Net
    model_path: str
    src_path: str
    pred_path: str
    gt_path: str

    def as_dict(self) -> dict:
        """Convert the EvaluateConfig to a dictionary."""
        d = asdict(self)
        d["net"] = self.net.value
        return d

@dataclass
class PredictConfig:
    net: Net
    model_path: str
    src_path: str
    dst_path: str

    def as_dict(self) -> dict:
        """Convert the PredictConfig to a dictionary."""
        d = asdict(self)
        d["net"] = self.net.value
        return d


@dataclass
class PipelineConfig:
    """Configuration for the pipeline
    This class can only hold one of the following configurations:
    - preprocess_config
    - train_config
    - evaluate_config
    - predict_config
    """
    step: str
    preprocess_config: Optional[PreprocessConfig] = None
    train_config: Optional[TrainConfig] = None
    evaluate_config: Optional[EvaluateConfig] = None
    predict_config: Optional[PredictConfig] = None

    def __post_init__(self) -> None:
        """Ensure exactly one configuration is provided."""
        configs = [
            self.preprocess_config,
            self.train_config,
            self.evaluate_config,
            self.predict_config,
        ]
        if sum(cfg is not None for cfg in configs) != 1:
            raise ValueError("Exactly one configuration option must be provided.")

    @property
    def active_config(
        self,
    ) -> Optional[PreprocessConfig | TrainConfig | EvaluateConfig | PredictConfig]:
        mapping = {
            "preprocess": self.preprocess_config,
            "train": self.train_config,
            "evaluate": self.evaluate_config,
            "predict": self.predict_config,
        }
        return mapping.get(self.step)

    def print_config(self) -> None:
        config = self.active_config
        if config:
            logger.info(f"Active configuration for {self.step} step:")
            for key, value in config.__dict__.items():
                logger.info(f"  {key}: {value}")
        else:
            raise ValueError("Invalid configuration")

    def as_dict(self) -> dict:
        """Convert the Pipeline to a dictionary."""
        return asdict(self)

@dataclass
class SegmentationMetrics:
    iou: float
    dice_score: float
    precision: float
    recall: float
    f1_score: float
    specificity: float = None
    inference_time: float = None

    def __str__(self):
        return (
            f"IoU: {self.iou:.4f}, Dice: {self.dice_score:.4f}, "
            f"Precision: {self.precision:.4f}, Recall: {self.recall:.4f}, F1: {self.f1_score:.4f}, "
            f"Specificity: {self.specificity if self.specificity else 'N/A'}, "
            f"Inference Time: {self.inference_time if self.inference_time else 'N/A'}s"
        )

    def as_dict(self) -> dict:
        """Convert the SegmentationMetrics to a dictionary."""
        return asdict(self)