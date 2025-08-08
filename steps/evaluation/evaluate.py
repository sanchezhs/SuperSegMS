from net.unet.unet import UNet
from net.yolo.yolo import YOLO
from schemas.pipeline_schemas import EvaluateConfig, Net
from steps.evaluation.viz.metrics import MetricsViz


def evaluate(config: EvaluateConfig) -> None:
    """Evaluate a trained model based on the provided configuration."""
    match config.net:
        case Net.UNET:
            metrics = UNet(config).evaluate()
        case Net.YOLO:
            metrics = YOLO(config).evaluate()
        case _:
            raise ValueError(f"Invalid net: {config.net}")

    MetricsViz(metrics).save_all_plots(
        base_path=config.pred_path.parent / "plots",
        prefix=config.model_path.stem,
        suffix=config.model_path.stem,
    )

    config.write_config()
