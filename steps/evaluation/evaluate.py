import os
import json
from schemas.pipeline_schemas import EvaluateConfig, Net
from net.unet.unet import UNet


def write_evaluate_params(config: EvaluateConfig) -> None:
    """Write evaluation parameters to a file."""
    with open(os.path.join(config.dst_path, "evaluate_params.json"), "w") as f:
        json.dump(
            config.as_dict(),
            f,
            indent=4,
        )
    print(
        f"Evaluation parameters saved to {os.path.join(config.dst_path, 'evaluate_params.json')}"
    )


def evaluate(config: EvaluateConfig) -> None:
    match config.net:
        case Net.Unet:
            UNet(config, mode="evaluate").evaluate()
        case Net.YOLO:
            pass
        case _:
            raise ValueError(f"Invalid net: {config.net}")


    write_evaluate_params(config)
