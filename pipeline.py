import toml
import os
import argparse

from schemas.pipeline_schemas import (
    PreprocessConfig,
    TrainConfig,
    EvaluateConfig,
    PredictConfig,
    PipelineConfig,
    Net,
    ResizeMethod,
)

from steps.preprocessing.preprocess import preprocess
from steps.training.train import train
from steps.evaluation.evaluate import evaluate
from steps.prediction.predict import predict


def setup_preprocess_parser(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--net",
        type=str,
        required=True,
        choices=[x.value for x in Net],
        help="Net type to preprocess the dataset for. Choices: 'unet' or 'yolo'.",
    )
    subparser.add_argument(
        "--src_path",
        type=str,
        required=True,
        help="Path to the raw dataset that needs preprocessing.",
    )
    subparser.add_argument(
        "--dst_path",
        type=str,
        required=True,
        help="Path to store the preprocessed dataset.",
    )
    subparser.add_argument(
        "--resize",
        type=str,
        default="",
        help="Resize images to a specific size, format: WIDTHxHEIGHT (e.g., 256x256). Leave empty to keep original size.",
    )
    subparser.add_argument(
        "--resize_method",
        type=str,
        choices=[x.value for x in ResizeMethod],
        help="Method to use for resizing. Required if --resize is used.",
    )
    subparser.add_argument(
        "--super_scale",
        type=int,
        choices=[2, 3, 4],
        help="Super resolution scale factor. Required if --resize_method is 'fsrcnn'.",
    )
    subparser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Fraction of the dataset used for training. The rest is used for validation. Default: 0.8 (80% training, 20% validation).",
    )

    def validate_resize_args(args: argparse.Namespace) -> None:
        """
        Validate that the resize-related arguments are provided correctly.
        """
        if args.resize and not args.resize_method:
            subparser.error("--resize requires --resize_method to be specified.")
        if args.resize and args.super_scale:
            subparser.error("--resize and --super_scale cannot be used together.")
        if args.resize_method and not args.resize:
            subparser.error("--resize_method requires --resize to be specified.")

    subparser.set_defaults(func=validate_resize_args)


def setup_train_parser(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--net",
        type=str,
        required=True,
        choices=["unet", "yolo"],
        help="Net to train. Choices: 'unet' or 'yolo'.",
    )
    subparser.add_argument(
        "--dst_path",
        type=str,
        required=True,
        help="Directory to save training results, including model checkpoints and logs.",
    )
    subparser.add_argument(
        "--src_path",
        type=str,
        required=True,
        help="Path to the dataset used for training.",
    )
    subparser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of samples processed per training step. Default: 8.",
    )
    subparser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Total number of training iterations over the dataset. Default: 10.",
    )
    subparser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Step size at each iteration when updating model weights. Default: 0.001.",
    )
    subparser.add_argument(
        "--limit_resources",
        type=bool,
        default=False,
        help="Limit GPU memory usage to 50%.",
    )


def setup_evaluate_parser(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model file to be evaluated.",
    )
    subparser.add_argument(
        "--src_path",
        type=str,
        required=True,
        help="Path to the dataset used for evaluation.",
    )


def setup_predict_parser(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--net",
        type=str,
        required=True,
        choices=[x.value for x in Net],
        help="Net type to make predictions with. Choices: 'unet' or 'yolo'.",
    )
    subparser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model file used for making predictions.",
    )
    subparser.add_argument(
        "--src_path",
        type=str,
        required=True,
        help="Path to the dataset on which predictions will be performed.",
    )
    subparser.add_argument(
        "--dst_path",
        type=str,
        required=True,
        help="Directory to save the generated predictions.",
    )


def parse_toml_config(config: str | None) -> PipelineConfig:
    config_path = config or "./config.toml"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found {config_path}.")

    conf = None

    with open(config_path, "r") as f:
        conf = f.read()

    if not conf:
        raise ValueError(f"Config file {config_path} is empty.")

    parsed_toml = toml.loads(conf)
    step = parsed_toml.get("step", None)

    if not step:
        raise ValueError("Step is required in the config file.")

    match parsed_toml["step"]:
        case "preprocess":
            resize = parsed_toml["preprocess"].get("resize", None)

            if resize:
                parts = resize.split("x")

            if len(parts) == 2:
                resize = (int(parts[0]), int(parts[1]))
            else:
                resize = None

            return PipelineConfig(
                step=step,
                preprocess_config=PreprocessConfig(
                    net=Net(parsed_toml["preprocess"]["net"]),
                    src_path=parsed_toml["preprocess"]["src_path"],
                    dst_path=parsed_toml["preprocess"][
                        "dst_path"
                    ],
                    split=parsed_toml["preprocess"]["split"],
                    resize=resize,
                    super_scale=parsed_toml["preprocess"].get("super_scale", None),
                    resize_method=parsed_toml["preprocess"].get("resize_method", None),
                ),
            )
        case "train":
            return PipelineConfig(
                step=step,
                train_config=TrainConfig(
                    net=Net(parsed_toml["train"]["net"]),
                    dst_path=parsed_toml["train"]["dst_path"],
                    src_path=parsed_toml["train"]["src_path"],
                    batch_size=parsed_toml["train"]["batch_size"],
                    epochs=parsed_toml["train"]["epochs"],
                    learning_rate=parsed_toml["train"]["learning_rate"],
                    limit_resources=parsed_toml["train"].get("limit_resources", False),
                ),
            )
        case "evaluate":
            return PipelineConfig(
                step=step,
                evaluate_config=EvaluateConfig(
                    net=Net(parsed_toml["evaluate"]["net"]),
                    model_path=parsed_toml["evaluate"]["model_path"],
                    src_path=parsed_toml["evaluate"]["src_path"],
                    pred_path=parsed_toml["evaluate"]["pred_path"],
                    gt_path=parsed_toml["evaluate"]["gt_path"],
                ),
            )
        case "predict":
            return PipelineConfig(
                step=step,
                predict_config=PredictConfig(
                    net=Net(parsed_toml["predict"]["net"]),
                    model_path=parsed_toml["predict"]["model_path"],
                    src_path=parsed_toml["predict"]["src_path"],
                    dst_path=parsed_toml["predict"]["dst_path"],
                ),
            )
        case _:
            raise ValueError(f"Invalid step {parsed_toml['step']}")


def parse_cli_args(args: argparse.Namespace) -> PipelineConfig:
    if not args.config and not args.step:
        raise ValueError("Either --config or --step must be provided")

    if args.config and args.step:
        raise ValueError("Cannot provide both --config and --step")

    if args.config:
        return parse_toml_config(args.config)

    if args.step == "preprocess":
        parts = args.resize.split("x")

        if len(parts) == 2:
            args.resize = (int(parts[0]), int(parts[1]))
        else:
            args.resize = None

        return PipelineConfig(
            step=args.step,
            preprocess_config=PreprocessConfig(
                net=Net(args.net),
                src_path=args.src_path,
                dst_path=args.dst_path,
                split=args.split,
                resize=args.resize,
                super_scale=args.super_scale,
                resize_method=args.resize_method or None,
            ),
        )
    elif args.step == "train":
        return PipelineConfig(
            step=args.step,
            train_config=TrainConfig(
                net=Net(args.net),
                dst_path=args.dst_path,
                src_path=args.src_path,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
            ),
        )
    elif args.step == "evaluate":
        return PipelineConfig(
            step=args.step,
            evaluate_config=EvaluateConfig(
                net=Net(args.net),
                model_path=args.model_path,
                src_path=args.src_path,
                pred_path=args.pred_path,
                gt_path=args.gt_path
            ),
        )
    elif args.step == "predict":
        return PipelineConfig(
            step=args.step,
            predict_config=PredictConfig(
                net=Net(args.net),
                model_path=args.model_path,
                src_path=args.src_path,
                dst_path=args.dst_path,
            ),
        )
    else:
        raise ValueError(f"Invalid step {args.step}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""A CLI program for preprocessing, training, evaluating, and predicting with different neural network models.
Each step is a separate module that can be run independently.
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration file in TOML format. If provided, CLI arguments will be ignored.",
        default="config.toml",
        required=False,
    )

    subparsers = parser.add_subparsers(
        dest="step", required=False, description="Choose a pipeline step to execute."
    )

    # Preprocess
    setup_preprocess_parser(
        subparsers.add_parser("preprocess", help="Preprocess the dataset for training.")
    )

    # Train
    setup_train_parser(
        subparsers.add_parser("train", help="Train a neural network model.")
    )

    # Evaluate
    setup_evaluate_parser(
        subparsers.add_parser("evaluate", help="Evaluate a trained model on a dataset.")
    )

    # Predict
    setup_predict_parser(
        subparsers.add_parser(
            "predict", help="Use a trained model to make predictions on a dataset."
        )
    )

    args = parser.parse_args()
    pipeline_config = parse_cli_args(args)
    pipeline_config.print_config()

    if pipeline_config.step == "preprocess" and pipeline_config.preprocess_config:
        preprocess(pipeline_config.preprocess_config)
    elif pipeline_config.step == "train" and pipeline_config.train_config:
        train(pipeline_config.train_config)
    elif pipeline_config.step == "evaluate" and pipeline_config.evaluate_config:
        evaluate(pipeline_config.evaluate_config)
    elif pipeline_config.step == "predict" and pipeline_config.predict_config:
        predict(pipeline_config.predict_config)
    else:
        raise ValueError(f"Invalid step {pipeline_config.step}")
