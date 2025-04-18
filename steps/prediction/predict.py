import os
import json
from schemas.pipeline_schemas import PredictConfig, Net
from net.unet.unet import UNet
from net.yolo.yolo import YOLO
from loguru import logger

def write_predict_params(config: PredictConfig) -> None:
    """Write prediction parameters to a file."""
    with open(os.path.join(config.dst_path, "predict_params.json"), "w") as f:
        json.dump(
            config.as_dict(),
            f,
            indent=4,
        )
    logger.info(
        f"Prediction parameters saved to {os.path.join(config.dst_path, 'predict_params.json')}"
    )


def predict(config: PredictConfig) -> None:
    match config.net:
        case Net.UNET:
            UNet(config, mode="predict").predict()
        case Net.YOLO:
            YOLO(config).predict()
        case _:
            raise ValueError(f"Invalid net: {config.net}")
    write_predict_params(config)
