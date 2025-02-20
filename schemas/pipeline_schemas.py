from typing import Optional
from dataclasses import dataclass
from enum import Enum

class Model(Enum):
    UNET = 'unet'
    YOLO = 'yolo'

@dataclass
class PreprocessConfig:
    model: Model
    dataset_path: str
    processed_dataset_path: str
    split: Optional[float] = 0.8
    resize: Optional[tuple[int, int]] = None

@dataclass
class TrainConfig:
    model: Model
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
    preprocess_config: Optional[PreprocessConfig] = None
    train_config: Optional[TrainConfig] = None
    evaluate_config: Optional[EvaluateConfig] = None
    predict_config: Optional[PredictConfig] = None

    def print_config(self):
        if self.preprocess_config:
            print("Preprocess config:")
            print(self.preprocess_config)
        elif self.train_config:
            print("Train config:")
            print(self.train_config)
        elif self.evaluate_config:
            print("Evaluate config:")
            print(self.evaluate_config)
        elif self.predict_config:
            print("Predict config:")
            print(self.predict_config)
        else:
            raise ValueError("Invalid configuration")