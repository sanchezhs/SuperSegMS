import os
import json
from schemas.pipeline_schemas import TrainConfig, Net

from net.unet.unet import UNet
from net.yolo.yolo import YOLO
from loguru import logger

def write_train_params(config: TrainConfig) -> None:
    """Write training parameters to a file."""
    with open(os.path.join(config.dst_path, "train_params.json"), "w") as f:
        json.dump(
            config.model_dump(),
            f,
            indent=4,
        )
    logger.info(
        f"Training parameters saved to {os.path.join(config.dst_path, 'train_params.json')}"
    )


def train(config: TrainConfig) -> None:
    match config.net:
        case Net.UNET:
            UNet(config, mode="train").train()
        case Net.YOLO:
            YOLO(config).train()
        case _:
            raise ValueError(f"Invalid Net name {config.net}")
    write_train_params(config)
