from schemas.pipeline_schemas import TrainConfig, Model

from net.unet.unet import UNet


def train(config: TrainConfig) -> None:
    match config.model:
        case Model.UNET:
            UNet(config, mode="train").train()
        case Model.YOLO:
            print("Training YOLO model")
        case _:
            raise ValueError(f"Invalid model name {config.model}")