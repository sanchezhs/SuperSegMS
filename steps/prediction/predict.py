from schemas.pipeline_schemas import PredictConfig, Net
from net.unet.unet import UNet
from net.yolo.yolo import YOLO

def predict(config: PredictConfig) -> None:
    match config.net:
        case Net.UNET:
            UNet(config, mode="predict").predict()
        case Net.YOLO:
            YOLO(config).predict()
        case _:
            raise ValueError(f"Invalid net: {config.net}")