import json
from pathlib import Path
from loguru import logger
from schemas.pipeline_schemas import VisualizeConfig, SegmentationMetrics
from schemas.pipeline_schemas import Net
from steps.visualization.core.plotter import MetricsPlotter

def visualize(config: VisualizeConfig) -> None:
    """
    Generate and save visualization plots for segmentation model evaluation metrics.
    Args:
        config (VisualizeConfig): Configuration object containing parameters for visualization.
    Raises:
        ValueError: If the specified network type in the configuration is not supported.
        FileNotFoundError: If the prediction path or required files (e.g., metrics.json) do not exist.
    The function performs the following steps:
        1. Validates the network type and the existence of the prediction path.
        2. Loads evaluation metrics from `metrics.json` and optionally `kfold_summary.json`.
        3. Creates a directory for saving plots if it does not already exist.
        4. Loads per-image metrics and lesion area data if available.
        5. Initializes a `MetricsPlotter` object to generate plots.
        6. Saves all generated plots to the specified directory with a given prefix.
        7. Writes the configuration to a file for reproducibility.
        8. Logs the location of the saved visualizations.
    Note:
        - Supported network types for visualization are UNET and YOLO.
        - Ensure that the `evaluate` step has been run prior to visualization to generate the required metrics files.
    """
    if config.net != Net.UNET and config.net != Net.YOLO:
        raise ValueError(f"Unsupported network type for visualization: {config.net}. Only UNET and YOLO are supported.")

    if not config.pred_path.exists():
        raise FileNotFoundError(f"Prediction path does not exist: {config.pred_path}")

    metrics_path = Path(config.pred_path) / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found at {metrics_path}. Run evaluate first.")

    data = json.loads(metrics_path.read_text())
    metrics = SegmentationMetrics(**data)

    kfold_summary_file = Path(config.pred_path) / "kfold_summary.json"
    kfold_summary = None
    if kfold_summary_file.exists():
        kfold_summary = json.loads(kfold_summary_file.read_text())

    plots_dir = Path(config.plots_dir) if config.plots_dir else (Path(config.pred_path).parent / "plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    per_image = json.loads((Path(config.pred_path) / "per_image_metrics.json").read_text())
    lesion_areas = [d.get("lesion_area_px") for d in per_image] if per_image and "lesion_area_px" in per_image[0] else None

    title = config.plots_title or f"{config.net.name.upper()} Evaluation"
    if config.plot_prefix:
        prefix = config.plot_prefix
    elif config.model_path:
        prefix = Path(config.model_path).stem
    else:
        prefix = "metrics"

    plotter = MetricsPlotter(
        metrics,
        per_image=per_image,
        lesion_areas=lesion_areas,
        kfold_summary=kfold_summary,
        title=title
    )
    
    plotter.save_all_plots(
        base_path=plots_dir,
        prefix=prefix,
        title_prefix=title
    )

    config.write_config()
    logger.info(f"Visualization saved to {plots_dir}")
