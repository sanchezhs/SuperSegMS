from net.unet.unet import UNet
from net.yolo.yolo import YOLO
from schemas.pipeline_schemas import Net, TrainConfig


def train(config: TrainConfig) -> None:
    """Train a model based on the provided configuration."""
    match config.net:
        case Net.UNET:
            UNet(config).train()
        case Net.YOLO:
            YOLO(config).train()
        case _:
            raise ValueError(f"Invalid Net name {config.net}")

    config.write_config()