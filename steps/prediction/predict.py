from net.unet.unet import UNet
from net.yolo.yolo import YOLO
from schemas.pipeline_schemas import Net, PredictConfig


def predict(config: PredictConfig) -> None:
    """Make predictions using a trained model based on the provided configuration."""
    match config.net:
        case Net.UNET:
            UNet(config, mode="predict").predict()
        case Net.YOLO:
            YOLO(config).predict()
        case _:
            raise ValueError(f"Invalid net: {config.net}")
        
    config.write_config()