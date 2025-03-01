from typing import Optional
from dataclasses import dataclass
from enum import Enum

# (X,Y,Z) = (width, height, depth)
NIFTI_SIZE = (182, 218)  # width, height


class Net(Enum):
    UNET = "unet"
    YOLO = "yolo"


class ResizeMethod(Enum):
    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"


@dataclass
class PreprocessConfig:
    net: Net
    dataset_path: str
    processed_dataset_path: str
    split: Optional[float] = 0.8
    resize: Optional[tuple[int, int]] = None
    resize_method: Optional[ResizeMethod] = None
    super_scale: Optional[int] = None


@dataclass
class TrainConfig:
    net: Net
    output_path: str
    dataset_path: str
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 1e-3


@dataclass
class EvaluateConfig:
    model_path: str
    dataset_path: str


@dataclass
class PredictConfig:
    model_path: str
    dataset_path: str
    output_path: str


@dataclass
class PipelineConfig:
    """Configuration for the pipeline
    This class can only hold one of the following configurations:
    - preprocess_config
    - train_config
    ...
    """

    step: str
    preprocess_config: Optional[PreprocessConfig] = None
    train_config: Optional[TrainConfig] = None
    evaluate_config: Optional[EvaluateConfig] = None
    predict_config: Optional[PredictConfig] = None

    def __post_init__(self):
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
    def active_config(self):
        mapping = {
            "preprocess": self.preprocess_config,
            "train": self.train_config,
            "evaluate": self.evaluate_config,
            "predict": self.predict_config,
        }
        return mapping.get(self.step)

    def print_config(self):
        config = self.active_config
        if config:
            print(f"{self.step.capitalize()} config:")
            print(config)
        else:
            raise ValueError("Invalid configuration")
