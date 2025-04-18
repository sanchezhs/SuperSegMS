from typing import Literal
import yaml
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO as uYOLO

from schemas.pipeline_schemas import TrainConfig, PredictConfig
from loguru import logger

class YOLO:
    def __init__(self, config: TrainConfig|PredictConfig) -> None:
        self.config = config
        self.src_path = config.src_path
        self.dst_path = config.dst_path
        self.yaml_path = os.path.join(self.dst_path, "data.yaml")

        if not os.path.exists(self.src_path):
            raise FileNotFoundError(f"Source path {self.src_path} does not exist. Did you run the preprocess step?")

        if isinstance(config, TrainConfig):
            self.batch_size = config.batch_size
            self.epochs = config.epochs
            # Default segmentation model for training
            self.model_path = "./net/yolo/models/yolo11m-seg.pt"
        elif isinstance(config, PredictConfig):
            self.model_path = config.model_path
        else:
            raise ValueError("Invalid config type. Allowed types are TrainConfig and PredictConfig.")

        self.create_yaml()

    def train(self) -> None:
        imgsz = self._get_image_size()
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
        model.predict(
            source=os.path.join(self.src_path, "images", "val"),
            project=self.dst_path,
            save_txt=True,
            save_conf=True,
            save_crop=False,
            device="cuda",
        )

        self.draw_predictions(folder_name="predict_test")
        self.draw_predictions(folder_name="predict_val")
        # self.visualize_predictions()

    def evaluate(self) -> None:
        pass

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

    def draw_predictions(self, folder_name: str = Literal["predict_test", "predict_val"]) -> None:
        output_dir = os.path.join(self.dst_path, folder_name, "masks")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_dir = os.path.join(self.src_path, "images", folder_name.split("_")[-1])
        prediction_dir = os.path.join(self.dst_path, folder_name, "labels")
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg"))])
        img_size = self._get_image_size()
        img_size = (img_size, img_size)

        for image_file in image_files:
            base_name = os.path.splitext(image_file)[0]
            prediction_file = f"{base_name}.txt"

            image_path = os.path.join(image_dir, image_file)
            prediction_path = os.path.join(prediction_dir, prediction_file)

            if not os.path.exists(prediction_path):
                logger.info(f"Warning: Prediction not found for {image_file}")
                continue

            # Load the image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Error: Could not load image {image_path}")
                continue

            img = cv2.resize(img, img_size)  # Resize if necessary
            mask = np.zeros_like(img)

            # Read the predictions
            with open(prediction_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                values = list(map(float, line.split()))

                # If the line has more than two values, interpret as segmentation coordinates
                if len(values) > 2:
                    class_id = int(values[0])
                    coords = values[1:]

                    if len(coords) % 2 != 0:
                        logger.warning(f"Warning: Odd number of coordinates in {line}, discarding last value.")
                        coords = coords[:-1]

                    if len(coords) >= 4:
                        points = np.array(coords).reshape(-1, 2)
                        points[:, 0] *= img_size[1]
                        points[:, 1] *= img_size[0]
                        points = points.astype(np.int32)

                        # Draw the segmentation on the mask
                        cv2.polylines(mask, [points], isClosed=True, color=255, thickness=1)
                        cv2.fillPoly(mask, [points], color=255)

            # Save the image with the mask
            dst_path = os.path.join(output_dir, f"{base_name}.png")
            cv2.imwrite(dst_path, mask)
            logger.info(f"Saved: {dst_path}")

    def _get_image_size(self) -> int:
        """Get the image size from the data.yaml file."""
        img = self.src_path + "/images/test/P54.png"
        img = cv2.imread(img)
        return img.shape[0]

    # def visualize_predictions(self):
    #     yolo_predictions_path = "yolo_res_single/predictions/predict/labels"
    #     test_images_path = "datasets/yolo_single/images/test"
    #     command = f"yolo predict model={self.model_path} task=segment overlap_mask=True imgsz=256"

    #     if os.path.exists(yolo_predictions_path) and os.path.exists(test_images_path):
    #         test_images = sorted([f for f in os.listdir(test_images_path) if f.endswith((".png", ".jpg"))])

    #         for image in test_images:
    #             image_path = os.path.join(test_images_path, image)
    #             command += f" source={image_path}"
    #             os.system(command) 
    #     else:
    #         print("Predictions or test images directory not found.")
