from net.unet.unet import UNet
from net.yolo.yolo import YOLO
from schemas.pipeline_schemas import EvaluateConfig, Net
from steps.visualization.core.plotter import MetricsPlotter


def evaluate(config: EvaluateConfig) -> None:
    """
    Evaluate a trained model based on the provided configuration.

    This function evaluates a trained model using the specified network type 
    and configuration. It supports multiple network architectures and generates 
    evaluation metrics. Additionally, it saves plots of the evaluation metrics 
    and writes the configuration to a file.

    Args:
        config (EvaluateConfig): The configuration object containing all necessary 
            parameters for evaluation, including the network type, model path, 
            prediction path, and other settings.

    Raises:
        ValueError: If the specified network type in the configuration is invalid.

    Supported Networks:
        - UNet
        - YOLO

    Side Effects:
        - Saves evaluation metric plots to the specified directory.
        - Writes the evaluation configuration to a file.
    """
    """Evaluate a trained model based on the provided configuration."""
    match config.net:
        case Net.UNET:
            metrics = UNet(config).evaluate()
        case Net.YOLO:
            metrics = YOLO(config).evaluate()
        case _:
            raise ValueError(f"Invalid net: {config.net}")

    MetricsPlotter(metrics).save_all_plots(
        base_path=config.pred_path.parent / "plots",
        prefix=config.model_path.stem,
        suffix=config.model_path.stem,
    )

    config.write_config()
