import os
import sys
import json
from loguru import logger
from pydantic import ValidationError
from utils.gcs import upload_file_to_bucket

from schemas.pipeline_schemas import (
    PipelineConfig,
    PreprocessConfig,
    TrainConfig,
    PredictConfig,
    EvaluateConfig,
)

from steps.preprocessing.preprocess import preprocess
from steps.training.train import train
from steps.prediction.predict import predict
from steps.evaluation.evaluate import evaluate

ALL_STEPS = ["preprocess", "train", "predict", "evaluate"]
BUCKET = "tfm-training-results"

def parse_json_experiment(config_path: str, experiment_id: str, step: str) -> PipelineConfig:
    logger.info(f"Parsing JSON config from: `{config_path}` | Experiment ID: `{experiment_id}` | Step: `{step}`")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        data = json.load(f)

    experiments = data.get("experiments", [])
    selected = next((exp for exp in experiments if exp.get("id") == experiment_id), None)
    if not selected:
        raise ValueError(f"No experiment with ID '{experiment_id}' found.")

    step_config = selected.get(step)
    if not step_config:
        raise ValueError(f"Step '{step}' not defined for experiment '{experiment_id}'.")

    try:
        return PipelineConfig(
            step=step,
            preprocess_config=PreprocessConfig(**step_config) if step == "preprocess" else None,
            train_config=TrainConfig(**step_config) if step == "train" else None,
            predict_config=PredictConfig(**step_config) if step == "predict" else None,
            evaluate_config=EvaluateConfig(**step_config) if step == "evaluate" else None,
        )
    except ValidationError as e:
        logger.error(f"Validation error parsing config for step '{step}':\n{e}")
        raise

def run_step(config: PipelineConfig):
    logger.info(f"Running step: {config.step}")
    logger.info(config.active_config().model_dump())

    if config.step == "preprocess":
        preprocess(config.preprocess_config)
    elif config.step == "train":
        train(config.train_config)
    elif config.step == "predict":
        predict(config.predict_config)
    elif config.step == "evaluate":
        evaluate(config.evaluate_config)
    else:
        raise ValueError(f"Unknown step '{config.step}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <experiment_ids> [steps]")
        print("Examples:")
        print("  python main.py A")
        print("  python main.py A,B")
        print("  python main.py A preprocess,train")
        print("  python main.py A,B,C predict")
        print("  python main.py A-C train")
        sys.exit(1)

    experiment_ids = sys.argv[1].split(",")
    if len(experiment_ids) == 1 and "-" in experiment_ids[0]:
        start, end = experiment_ids[0].split("-")
        experiment_ids = [chr(i) for i in range(ord(start), ord(end) + 1)]

    steps = sys.argv[2].split(",") if len(sys.argv) > 2 else ALL_STEPS

    for experiment_id in experiment_ids:
        logger.info(f"Running experiment '{experiment_id}' with steps: {steps}")
        for step in steps:
            config = parse_json_experiment("config.json", experiment_id, step)
             # Skip evaluation for k-fold training
            if step == "evaluate" and config.train_config.use_kfold:
                break
            run_step(config)

    # upload_file_to_bucket(
    #         bucket_name=BUCKET,
    #         local_path="results",
    #         destination_path="resultados.tar.gz",
    #     )

    # upload_file_to_bucket(
    #     bucket_name=BUCKET,
    #     local_path="datasets",
    #     destination_path="datasets.tar.gz",
    # )