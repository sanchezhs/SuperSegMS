import json
import os
import sys
from loguru import logger
from pydantic import ValidationError

from schemas.pipeline_schemas import ALL_STEPS, EvaluateConfig, PipelineConfig, PredictConfig, PreprocessConfig, TrainConfig
from steps.evaluation.evaluate import evaluate
from steps.prediction.predict import predict
from steps.preprocessing.preprocess import preprocess
from steps.training.train import train
from utils.gcs import upload_file_to_bucket

def parse_json_experiment(config_path: str, experiment_id: str, step: str) -> PipelineConfig:
    """
    Parse a JSON configuration file and return the configuration for a specific experiment and step.
    Args:
        config_path (str): Path to the JSON configuration file.
        experiment_id (str): ID of the experiment to run.
        step (str): Step to run (e.g., "preprocess", "train", "predict", "evaluate").
    Returns:
        PipelineConfig: The configuration for the specified experiment and step.
    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the experiment ID or step is not found in the config.
        ValidationError: If the configuration data is invalid.
    """
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


def run_step(config: PipelineConfig) -> None:
    """
    Run a specific step of the pipeline based on the provided configuration.
    Args:
        config (PipelineConfig): The configuration for the step to run.
    """
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
    If expr is a single-letter range like "A-C", expand it to ["A", "B", "C"].
    Otherwise, split on commas and return the list.
    Args:
        expr (str): The expression to expand, e.g. "A-C" or "A,B,C".
    Returns:
        list[str]: Expanded list of experiment IDs.
    """
    if "-" in expr and len(expr.split('-')) == 2:
        start, end = expr.split("-")
        if len(start) == 1 and len(end) == 1 and start.isalpha() and end.isalpha():
            return [chr(i) for i in range(ord(start), ord(end) + 1)]
    return expr.split(",")

def run_pipeline(args):
    """
    Main function to run the pipeline based on command line arguments.
    Args:
        args: Parsed command line arguments.
    """
    if not args.experiment_ids:
        logger.error("No experiment IDs provided. Use --generate-config to create a template, or specify experiment IDs to run.")
        sys.exit(1)

    # Expand and validate experiment IDs
    experiment_ids: list[str] = []
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
        sys.exit(1)

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

        # If upload flag is set, upload results directory to the bucket
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
