import argparse
import json
from loguru import logger
from schemas.pipeline_schemas import ALL_STEPS, DEFAULT_BUCKET, DEFAULT_DEST_PATH

def parse_cli_args():
    """
    Parse command-line arguments for the pipeline runner.
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run one or more pipeline experiments with specified steps.\n\n"
            "JSON config format:\n"
            "  The configuration file must be a JSON object with a top-level key \"experiments\",\n"
            "  which is a list of objects. Each experiment object must include:\n"
            "    - \"id\": unique string identifier (e.g. \"A\", \"B\").\n"
            "    - \"preprocess\": { … }   (fields required by schemas.pipeline_schemas.PreprocessConfig)\n"
            "    - \"train\": { … }        (fields required by schemas.pipeline_schemas.TrainConfig)\n"
            "    - \"predict\": { … }      (fields required by schemas.pipeline_schemas.PredictConfig)\n"
            "    - \"evaluate\": { … }     (fields required by schemas.pipeline_schemas.EvaluateConfig)\n"
            "  All four step‐objects are mandatory in each experiment block, though you can leave unused\n"
            "  keys blank or fill with placeholders if you intend to skip steps.\n\n"
            "  Example structure:\n"
            "  {\n"
            "      \"experiments\": [\n"
            "          {\n"
            "              \"id\": \"A\",\n"
            "              \"preprocess\": { /* see PreprocessConfig fields */ },\n"
            "              \"train\": { /* see TrainConfig fields */ },\n"
            "              \"predict\": { /* see PredictConfig fields */ },\n"
            "              \"evaluate\": { /* see EvaluateConfig fields */ }\n"
            "          },\n"
            "          … more experiments …\n"
            "      ]\n"
            "  }\n\n"
            "Use `--generate-config` to emit a default/template JSON you can modify.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "experiment_ids",
        nargs="?",
        help=(
            "Comma-separated list of experiment IDs to run. "
            "You can specify a range with a dash, e.g. 'A-C' expands to ['A', 'B', 'C'].\n"
            "If omitted and --generate-config is provided, no experiments will be run."
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
        help=f"GCS bucket name to upload to. Defaults to '{DEFAULT_BUCKET}'.",
    )
    parser.add_argument(
        "--local_path",
        default="results",
        help=(
            "Local path to the directory containing results to upload. "
            "Defaults to 'results', which is where the pipeline writes outputs."
        ),
    )
    parser.add_argument(
        "--destination",
        default=DEFAULT_DEST_PATH,
        help=(
            "Destination path (including filename) inside the GCS bucket for the uploaded archive. "
            f"Defaults to '{DEFAULT_DEST_PATH}'."
        ),
    )
    parser.add_argument(
        "--generate-config",
        nargs="?",
        const="stdout",
        metavar="[OUTPUT_PATH]",
        help=(
            "Emit a default/template JSON config. If no OUTPUT_PATH is given, prints to stdout. "
            "If OUTPUT_PATH is provided, writes default JSON to that file (overwrites if exists)."
        ),
    )
    return parser.parse_args()


def handle_generate_config(output_path: str | None):
    """
    Generate a default JSON configuration file with a single experiment template.
    If output_path is None or "stdout", print to stdout; otherwise write to the specified file.
    """
    default = {
        "experiments": [
            make_default_experiment_template()
        ]
    }
    content = json.dumps(default, indent=4)
    if output_path in (None, "stdout"):
        print(content)
    else:
        with open(output_path, "w") as f:
            f.write(content)
        logger.info(f"Default config written to: {output_path}")

def make_default_experiment_template() -> dict:
    """
    Return a single-experiment template with placeholders for each step.
    Users can copy/paste, fill in real values, and repeat under "experiments".
    """
    return {
        "id": "<experiment_id>",
        "preprocess": {
            "net": "<e.g. 'unet' or 'yolo'>",
            "src_path": "<path to raw data folder>",
            "dst_path": "<where to write processed data>",
            "resize": [320, 320],
            "split": 0.7,
            "strategy": "<e.g. 'all_slices', 'lesion_slices', 'top_five'>",
            "super_scale": 1
        },
        "train": {
            "net": "<same as preprocess.net>",
            "src_path": "<path to preprocessed data>",
            "dst_path": "<where to write training outputs>",
            "batch_size": 16,
            "use_kfold": False,
            "epochs": 25,
            "learning_rate": 0.001
        },
        "predict": {
            "net": "<same as train.net>",
            "model_path": "<path/to/trained/model/file>",
            "src_path": "<path to data for prediction>",
            "dst_path": "<where to write predicted outputs>"
        },
        "evaluate": {
            "net": "<same as train.net>",
            "model_path": "<path/to/trained/model/file>",
            "src_path": "<path to data for evaluation>",
            "pred_path": "<predictions folder>",
            "gt_path": "<ground-truth labels folder>"
        }
    }


def output_default_config(to_stdout: bool, filename: str | None):
    """
    Build a default JSON structure with one template experiment. 
    If to_stdout is True, print to stdout; otherwise write to 'filename'.
    """
    default = {
        "experiments": [
            make_default_experiment_template()
        ]
    }
    content = json.dumps(default, indent=4)
    if to_stdout:
        print(content)
    else:
        # Write to given filename (overwrites if exists)
        with open(filename, "w") as f:
            f.write(content)
        logger.info(f"Default config written to: {filename}")

