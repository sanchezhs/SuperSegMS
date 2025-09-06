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
    Args:
        output_path (str | None): Path to write the default config. If None or "stdout", print to stdout.
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
    Each field is pre-filled with a hint to guide the user.
    """
    return {
        "id": "<experiment_id: unique identifier, e.g. 'A'>",
        "preprocess": {
            "net": "<network to prepare data for: 'unet' or 'yolo'>",
            "src_path": "<path to raw MRI dataset>",
            "dst_path": "<where processed dataset will be saved>",
            "resize": [320, 320],  # [width, height] target size for slices
            "seed": 42,            # random seed for reproducibility
            "split": 0.7,          # fraction of patients for training (rest goes to test/val)
            "strategy": "<slice selection: 'all_slices', 'lesion_slices', 'lesion_block', 'brain_slices'>",
            "super_scale": 1,      # factor for super-resolution (1 = none, 2 = 2x upscaling, etc.)
            "kfold": {
                "enable": True,      # enable k-fold cross-validation
                "n_splits": 5,       # number of folds
                "seed": 42,          # seed for fold splits
                "mini_val_frac": 0.10, # fraction of training fold to hold out as mini-validation
                "link_mode": "hardlink" # how to reference files: 'copy' or 'hardlink'
            }
        },
        "train": {
            "net": "<must match preprocess.net>",
            "src_path": "<path to preprocessed dataset>",
            "dst_path": "<output directory for training results>",
            "batch_size": 16,         # how many slices per training step
            "use_kfold": True,       # whether to train with k-fold CV
            "epochs": 25,             # number of training epochs
            "learning_rate": 0.001    # initial learning rate
        },
        "predict": {
            "net": "<must match train.net>",
            "model_path": "<path/to/trained_model.pth>",
            "src_path": "<dataset to run inference on>",
            "dst_path": "<where predictions will be written>"
        },
        "evaluate": {
            "net": "<must match train.net>",
            "model_path": "<path/to/trained_model.pth>",
            "src_path": "<dataset to evaluate on>",
            "pred_path": "<predictions folder (from predict step)>",
            "gt_path": "<ground truth labels folder>"
        }
    }


def output_default_config(to_stdout: bool, filename: str | None) -> None:
    """
    Build a default JSON structure with one template experiment. 
    If to_stdout is True, print to stdout; otherwise write to 'filename'.
    Args:
        to_stdout (bool): If True, print the default config to stdout.
        filename (str | None): If provided, write the default config to this file.
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

