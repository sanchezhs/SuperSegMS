import os
import json
import argparse
from loguru import logger
from pydantic import ValidationError
from utils.gcs import upload_file_to_bucket

from schemas.pipeline_schemas import (
    PipelineConfig,
    PreprocessConfig,
    TrainConfig,
    PredictConfig,
    EvaluateConfig,
    # Constants
    ALL_STEPS,
    DEFAULT_BUCKET,
    DEFAULT_DEST_PATH,
)

from steps.preprocessing.preprocess import preprocess
from steps.training.train import train
from steps.prediction.predict import predict
from steps.evaluation.evaluate import evaluate


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

def expand_experiment_range(expr: str) -> list[str]:
    """
    If expr is a single letter range like "A-C", expand it to ["A", "B", "C"].
    Otherwise, split on commas and return the list.
    """
    if "-" in expr and len(expr.split('-')) == 2:
        start, end = expr.split("-")
        if len(start) == 1 and len(end) == 1 and start.isalpha() and end.isalpha():
            return [chr(i) for i in range(ord(start), ord(end) + 1)]
    return expr.split(",")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run one or more pipeline experiments with specified steps."
    )

    parser.add_argument(
        "experiment_ids",
        help=(
            "Comma-separated list of experiment IDs to run. "
            "You can specify a range with a dash, e.g. 'A-C' expands to ['A', 'B', 'C']."
        ),
    )
    parser.add_argument(
        "--steps",
        default=",".join(ALL_STEPS),
        help=(
            "Comma-separated list of steps to run for each experiment. "
            f"Allowed values: {ALL_STEPS}. Defaults to all steps."
        ),
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to the JSON configuration file. Defaults to 'config.json'.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="If set, upload results to a GCS bucket after each experiment run.",
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help=f"Name of the GCS bucket to upload to. Defaults to '{DEFAULT_BUCKET}'.",
    )
    parser.add_argument(
        "--local_path",
        default="results",
        help=(
            "Local path to the directory containing the results to upload. "
            "Defaults to 'results', which is where the pipeline stores results."
        ),
    )
    parser.add_argument(
        "--destination",
        default=DEFAULT_DEST_PATH,
        help=(
            "Destination path (including filename) inside the bucket for the uploaded archive. "
            f"Defaults to '{DEFAULT_DEST_PATH}'."
        ),
    )

    args = parser.parse_args()

    # Expand and validate experiment IDs
    experiment_ids = []
    for expr in args.experiment_ids.split(","):
        expr = expr.strip()
        if "-" in expr:
            experiment_ids.extend(expand_experiment_range(expr))
        else:
            experiment_ids.append(expr)

    # Parse steps and validate against ALL_STEPS
    steps = [s.strip() for s in args.steps.split(",")]
    invalid_steps = [s for s in steps if s not in ALL_STEPS]
    if invalid_steps:
        logger.error(f"Invalid step(s) specified: {invalid_steps}. Allowed steps are: {ALL_STEPS}")
        exit(1)

    # Main loop over experiments
    for experiment_id in experiment_ids:
        logger.info(f"Running experiment '{experiment_id}' with steps: {steps}")
        for step in steps:
            config = parse_json_experiment(args.config, experiment_id, step)
            # Skip evaluation for k-fold training
            if step == "evaluate" and config.train_config and config.train_config.use_kfold:
                logger.info(f"Skipping evaluation for '{experiment_id}' because use_kfold=True.")
                break
            run_step(config)

        # If upload flag is set, upload the results directory to the bucket
        if args.upload:
            upload_file_to_bucket(
                bucket_name=args.bucket,
                local_path=args.local_path,
                destination_path=args.destination,
            )
            logger.info(
                f"Results for experiment '{experiment_id}' uploaded to bucket '{args.bucket}' "
                f"at '{args.destination}'."
            )
        else:
            logger.info(f"Upload skipped for experiment '{experiment_id}'.")
