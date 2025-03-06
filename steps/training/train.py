from schemas.pipeline_schemas import TrainConfig, Net

from net.unet.unet import UNet
from net.yolo.yolo import YOLO


def train(config: TrainConfig) -> None:
    match config.net:
        case Net.UNET:
            UNet(config, mode="train").train()
        case Net.YOLO:
            YOLO(config).train()
        case _:
            raise ValueError(f"Invalid Net name {config.net}")