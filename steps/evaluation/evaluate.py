import os
import json
from net.yolo.yolo import YOLO
from schemas.pipeline_schemas import EvaluateConfig, Net
from net.unet.unet import UNet
from loguru import logger

def write_evaluate_params(config: EvaluateConfig) -> None:
    """Write evaluation parameters to a file."""
    with open(os.path.join(config.pred_path, "evaluate_params.json"), "w") as f:
        json.dump(
            config.as_dict(),
            f,
            indent=4,
        )
    logger.info(
        f"Evaluation parameters saved to {os.path.join(config.pred_path, 'evaluate_params.json')}"
    )


def evaluate(config: EvaluateConfig) -> None:
    match config.net:
        case Net.UNET:
            UNet(config, mode="evaluate").evaluate()
        case Net.YOLO:
            YOLO(config).evaluate()
        case _:
            raise ValueError(f"Invalid net: {config.net}")


    write_evaluate_params(config)
