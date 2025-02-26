from schemas.pipeline_schemas import TrainConfig, Model

from net.unet.unet import UNet
from net.yolo.yolo import YOLO


def train(config: TrainConfig) -> None:
    match config.model:
        case Model.UNET:
            UNet(config, mode="train").train()
        case Model.YOLO:
            YOLO(config).train()
        case _:
            raise ValueError(f"Invalid model name {config.model}")