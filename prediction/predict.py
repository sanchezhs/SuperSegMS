from schemas.pipeline_schemas import PredictConfig
from net.unet.unet import UNet


def predict(config: PredictConfig) -> None:
    UNet(config, mode="predict").predict()