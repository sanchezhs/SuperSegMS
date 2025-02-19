import argparse
import json

from preprocess.preprocess import apply_preprocessing

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TFM Pipeline")
    parser.add_argument("--step", type=str, required=True, choices=["preprocess", "train", "evaluate", "predict"])
    parser.add_argument("--model", type=str, required=True, choices=["unet", "yolo"])
    parser.add_argument("--config", type=str, default="config.json",  help="Path to configuration file")

    args = parser.parse_args()

    config = load_config(args.config)

    parts = config["hyperparameters"]["img_size"].split("x")
    img_size = (int(parts[0]), int(parts[1]))

    if args.step == "preprocess":
        apply_preprocessing(
            args.model,
            img_size,
            float(config["hyperparameters"]["split"]),
            config["dataset"]["input_path"], 
            config["dataset"]["output_path"]
        )
    elif args.step == "train":
        raise NotImplementedError("Train step not implemented")
    elif args.step == "evaluate":
        raise NotImplementedError("Evaluate step not implemented")
    elif args.step == "predict":
        raise NotImplementedError("Predict step not implemented")

