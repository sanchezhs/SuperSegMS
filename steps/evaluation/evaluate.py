from net.unet.unet import UNet
from net.yolo.yolo import YOLO
from schemas.pipeline_schemas import EvaluateConfig, Net


def evaluate(config: EvaluateConfig) -> None:
    """Evaluate a trained model based on the provided configuration."""
    match config.net:
        case Net.UNET:
            UNet(config).evaluate()
        case Net.YOLO:
            YOLO(config).evaluate()
        case _:
            raise ValueError(f"Invalid net: {config.net}")

    config.write_config()
