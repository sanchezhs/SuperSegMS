from schemas.pipeline_schemas import EvaluateConfig
from net.unet.unet import UNet


def evaluate(config: EvaluateConfig) -> None:
    UNet(config, mode="evaluate").evaluate()
