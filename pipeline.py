import argparse

from schemas.pipeline_schemas import PreprocessConfig, TrainConfig, EvaluateConfig, PredictConfig, PipelineConfig, Model

from preprocessing.preprocess import preprocess
from training.train import train
from evaluation.evaluate import evaluate
from prediction.predict import predict

def setup_preprocess_parser(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--model", type=str, required=True, choices=["unet", "yolo"])
    subparser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset to preprocess")
    subparser.add_argument("--processed_dataset_path", type=str, required=True, help="Path to save the preprocessed dataset")
    subparser.add_argument("--resize", type=str, default="", help="Resize images to this size")
    subparser.add_argument("--split", type=float, default=0.8, help="Train/val split")

def setup_train_parser(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--model", type=str, required=True, choices=["unet", "yolo"])
    subparser.add_argument("--output_path", type=str, required=True, help="Path to save the results")
    subparser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset dataset")
    subparser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    subparser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    subparser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")

def setup_evaluate_parser(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    subparser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset to evaluate")

def setup_predict_parser(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    subparser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset to evaluate")
    subparser.add_argument("--output_path", type=str, required=True, help="Path to save the predictions")

def parse_cli_args(args: argparse.Namespace) -> PipelineConfig:
    if args.step == "preprocess":
        parts = args.resize.split("x")
        
        if len(parts) == 2:
            args.resize = (int(parts[0]), int(parts[1]))
        else:
            args.resize = None

        return PipelineConfig(
            preprocess_config=PreprocessConfig(
                model=Model(args.model),
                dataset_path=args.dataset_path,
                processed_dataset_path=args.processed_dataset_path,
                split=args.split,
                resize=args.resize
            )
        )
    elif args.step == "train":
        return PipelineConfig(
            train_config=TrainConfig(
                model=Model(args.model),
                output_path=args.output_path,
                dataset_path=args.dataset_path,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate
            )
        )
    elif args.step == "evaluate":
        return PipelineConfig(
            evaluate_config=EvaluateConfig(
                model_path=args.model_path,
                dataset_path=args.dataset_path
            )
        )
    elif args.step == "predict":
        return PipelineConfig(
            predict_config=PredictConfig(
                model_path=args.model_path,
                dataset_path=args.dataset_path,
                output_path=args.output_path
            )
        )
    else:
        raise ValueError(f"Invalid step {args.step}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TFM Pipeline")
    subparsers = parser.add_subparsers(dest="step", required=True)
    
    # Preprocess
    setup_preprocess_parser(subparsers.add_parser("preprocess"))

    # Train
    setup_train_parser(subparsers.add_parser("train"))

    # Evaluate
    setup_evaluate_parser(subparsers.add_parser("evaluate"))

    # Predict
    setup_predict_parser(subparsers.add_parser("predict"))

    args = parser.parse_args()
    pipeline_config = parse_cli_args(args)
    pipeline_config.print_config()

    if args.step == "preprocess":
        preprocess(pipeline_config.preprocess_config)
    elif args.step == "train":
        train(pipeline_config.train_config)
    elif args.step == "evaluate":
        evaluate(pipeline_config.evaluate_config)
    elif args.step == "predict":
        predict(pipeline_config.predict_config)
    else:
        raise ValueError(f"Invalid step {args.step}")
