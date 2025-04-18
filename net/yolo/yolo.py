import yaml
import os
import cv2
import numpy as np
import json
import shutil
from ultralytics import YOLO as uYOLO
from schemas.pipeline_schemas import TrainConfig, PredictConfig, EvaluateConfig
from loguru import logger

class YOLO:
    def __init__(self, config: TrainConfig|PredictConfig|EvaluateConfig) -> None:
        self.config = config
        self.src_path = config.src_path

        if not os.path.exists(self.src_path):
            raise FileNotFoundError(f"Source path {self.src_path} does not exist. Did you run the preprocess step?")

        if isinstance(config, TrainConfig):
            self.dst_path = config.dst_path
            self.batch_size = config.batch_size
            self.epochs = config.epochs
            self.yaml_path = os.path.join(self.dst_path, "data.yaml")

            # Default segmentation model for training
            self.model_path = "./net/yolo/models/yolo11m-seg.pt"
        elif isinstance(config, PredictConfig):
            self.dst_path = config.dst_path
            self.model_path = config.model_path
            self.yaml_path = os.path.join(self.dst_path, "data.yaml")
        elif isinstance(config, EvaluateConfig):
            self.pred_path = config.pred_path
            self.model_path = config.model_path
            self.gt_path = config.gt_path
        else:
            raise ValueError("Invalid config type. Allowed types are TrainConfig and PredictConfig.")

        self.create_yaml()

    def train(self) -> None:
        imgsz = self._get_image_size()[0]
        model = uYOLO(self.model_path)
        
        model.train(
            data=self.yaml_path,
            epochs=self.epochs,
            batch=self.batch_size,
            save=True,
            imgsz=imgsz,
            project=self.dst_path,
            device="cuda",
            verbose=True,
        )

    def predict(self) -> None:
        model = uYOLO(self.model_path)

        if os.path.exists(os.path.join(self.dst_path, "predict")):
            logger.info(f"Prediction folder already exists: {os.path.join(self.dst_path, 'predict')}. Removing it.")
            shutil.rmtree(os.path.join(self.dst_path, "predict"))

        # Test
        model.predict(
            source=os.path.join(self.src_path, "images", "test"),
            project=self.dst_path,
            save_txt=True,
            save_conf=True,
            save_crop=False,
            device="cuda",
        )

        # Val
        # model.predict(
        #     source=os.path.join(self.src_path, "images", "val"),
        #     project=self.dst_path,
        #     save_txt=True,
        #     save_conf=True,
        #     save_crop=False,
        #     device="cuda",
        # )

        self.draw_predictions()
        # self.draw_predictions(folder_name="predict_val")

    def evaluate(self) -> None:
        """Evaluate the YOLO predictions using the ground truth masks."""
        logger.info(f"Evaluating YOLO predictions in folder {self.pred_path}")
        folder_name = os.path.basename(self.pred_path)
        if not os.path.exists(self.pred_path):
            raise FileNotFoundError(f"Prediction path {self.pred_path} does not exist. Did you run the predict step?")
        gt_dir = os.path.join(self.gt_path, "labels", folder_name.split("_")[-1])
        pred_dir = os.path.join(self.pred_path, folder_name, "masks")

        image_names = sorted(os.listdir(pred_dir))
        metrics = []

        for name in image_names:
            pred_mask = cv2.imread(os.path.join(pred_dir, name), cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.imread(os.path.join(gt_dir, name), cv2.IMREAD_GRAYSCALE)

            if pred_mask is None or gt_mask is None:
                logger.warning(f"Skipping {name} due to missing prediction or ground truth.")
                continue

            pred_bin = (pred_mask > 127).astype(np.uint8)
            gt_bin = (gt_mask > 127).astype(np.uint8)

            tp = np.logical_and(pred_bin, gt_bin).sum()
            fp = np.logical_and(pred_bin, np.logical_not(gt_bin)).sum()
            fn = np.logical_and(np.logical_not(pred_bin), gt_bin).sum()
            tn = np.logical_and(np.logical_not(pred_bin), np.logical_not(gt_bin)).sum()

            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            metrics.append({
                "image": name,
                "iou": iou,
                "dice": dice,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "specificity": specificity
            })

        avg_metrics = {
            key: np.mean([m[key] for m in metrics])
            for key in ["iou", "dice", "precision", "recall", "f1_score", "specificity"]
        }

        logger.info("Average metrics:")
        for k, v in avg_metrics.items():
            logger.info(f"{k}: {v:.4f}")

        # Save metrics to a JSON file
        metrics_path = os.path.join(self.dst_path, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.success(f"Metrics saved to {metrics_path}")
        logger.success("Evaluation completed.")

    def create_yaml(self) -> None:
        """Creates a YAML file with the dataset configuration for YOLO training."""
        os.makedirs(self.dst_path, exist_ok=True)
        # Yolo ultralytics adds an additional "datasets" folder to the path
        parts = self.src_path.split("/")
        path = "/".join(parts[-1:])
        data_yaml = {
            "path": path,
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": ["lesion"],
            "task": "segment",
        }

        with open(self.yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

    def draw_predictions(self) -> None:
        output_dir = os.path.join(self.dst_path, "masks")
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Creadted output directory for image masks: {output_dir}")
        prediction_dir = os.path.join(self.dst_path, "predict", "labels")

        if not os.path.exists(prediction_dir):
            raise FileNotFoundError(f"Prediction directory {prediction_dir} does not exist. Did you run the predict step?")

        img_size = self._get_image_size()
        image_dir = os.path.join(self.src_path, "images", "test")
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg"))])

        for image_file in image_files:
            base_name = os.path.splitext(image_file)[0]
            prediction_file = f"{base_name}.txt"

            image_path = os.path.join(image_dir, image_file)
            prediction_path = os.path.join(prediction_dir, prediction_file)

            if not os.path.exists(prediction_path):
                logger.info(f"Warning: Prediction not found for {image_file}")
                continue

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Error: Could not load image {image_path}")
                continue

            img = cv2.resize(img, img_size[:2])
            mask = np.zeros_like(img)

            with open(prediction_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                values = list(map(float, line.split()))

                if len(values) > 2:
                    coords = values[1:]

                    if len(coords) % 2 != 0:
                        logger.warning(f"Odd number of coordinates in {line}, discarding last value.")
                        coords = coords[:-1]

                    if len(coords) >= 4:
                        points = np.array(coords).reshape(-1, 2)
                        points[:, 0] *= img_size[1]
                        points[:, 1] *= img_size[0]
                        points = points.astype(np.int32)

                        cv2.polylines(mask, [points], isClosed=True, color=255, thickness=1)
                        cv2.fillPoly(mask, [points], color=255)

            dst_path = os.path.join(output_dir, f"{base_name}.png")
            cv2.imwrite(dst_path, mask)
            logger.info(f"Saved: {dst_path}")
            
        logger.success("Predictions drawn and saved correctly.")

    def _get_image_size(self) -> tuple[int, int]:
        """Get the image size from the data.yaml file."""
        path = self.src_path + "/images/test"
        img = os.listdir(path)
        img = os.path.join(path, img[0])
        img = cv2.imread(img)
        return img.shape  # (height, width, channels)
