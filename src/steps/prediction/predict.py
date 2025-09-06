from net.unet.unet import UNet
from net.yolo.yolo import YOLO
from schemas.pipeline_schemas import Net, PredictConfig


def predict(config: PredictConfig) -> None:
    """
    Make predictions using a trained model based on the provided configuration.
    Args:
        config (PredictConfig): The configuration object containing the parameters 
            for the prediction process, including the model type and other settings.
    Raises:
        ValueError: If the specified model type in the configuration (`config.net`) 
            is not supported.
    Supported Models:
        - UNet: Uses the UNet architecture for predictions.
        - YOLO: Uses the YOLO architecture for predictions.
    Side Effects:
        - Writes the updated configuration to a file or persistent storage 
          using `config.write_config()`.
    """
    """Make predictions using a trained model based on the provided configuration."""
    match config.net:
        case Net.UNET:
            UNet(config).predict()
        case Net.YOLO:
            YOLO(config).predict()
        case _:
            raise ValueError(f"Invalid net: {config.net}")
        
    config.write_config()