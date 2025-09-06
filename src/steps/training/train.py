from net.unet.unet import UNet
from net.yolo.yolo import YOLO
from schemas.pipeline_schemas import Net, TrainConfig


def train(config: TrainConfig) -> None:
    """
    Train a model based on the provided configuration.

    Args:
        config (TrainConfig): The configuration object containing all necessary
            parameters for training, including the network type, training settings,
            and other relevant options.

    Raises:
        ValueError: If the specified network type in the configuration is invalid.

    Notes:
        - Supports training for UNet and YOLO network types.
        - After training, the configuration is saved using the `write_config` method.
    """
    match config.net:
        case Net.UNET:
            UNet(config).train()
        case Net.YOLO:
            YOLO(config).train()
        case _:
            raise ValueError(f"Invalid Net name {config.net}")

    config.write_config()