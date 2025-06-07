import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from loguru import logger
from sklearn.model_selection import GroupKFold
from ultralytics import YOLO as uYOLO

from schemas.pipeline_schemas import EvaluateConfig, Net, PredictConfig, SegmentationMetrics, TrainConfig
from utils.send_msg import send_whatsapp_message


class YOLO:
    """Class to handle YOLO training, prediction, and evaluation.
    This class supports both standard training and k-fold cross-validation.
    It can also predict on new images and evaluate the predictions against ground truth masks.
    Attributes:
        config (TrainConfig | PredictConfig | EvaluateConfig): Configuration object for the step.
        src_path (str): Source path for images and labels.
        dst_path (str): Destination path for training results.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs for training.
        use_kfold (bool): Whether to use k-fold cross-validation.
        kfold_n_splits (int): Number of splits for k-fold cross-validation.
        kfold_seed (int): Random seed for k-fold cross-validation.
        yaml_path (str): Path to the data.yaml file used by YOLO.
        model_path (str): Path to the YOLO model weights.
    """
    def __init__(self, config: TrainConfig | PredictConfig | EvaluateConfig) -> None:
        self.config = config
        self.src_path = config.src_path

        if not self.src_path.exists():
            raise FileNotFoundError(
                f"Source path {self.src_path} does not exist. Did you run the preprocess step?"
            )

        if isinstance(config, TrainConfig):
            self.dst_path = config.dst_path
            self.batch_size = config.batch_size
            self.epochs = config.epochs
            self.use_kfold = getattr(config, "use_kfold", False)
            self.kfold_n_splits = getattr(config, "kfold_n_splits", 5)
            self.kfold_seed = getattr(config, "kfold_seed", 42)
            self.yaml_path = self.dst_path / "data.yaml"
            self.model_path = Path("./net/yolo/models/yolo11m-seg.pt")
            self.dst_path.mkdir(parents=True, exist_ok=True)
            if not self.use_kfold:
                self._create_yaml()

        elif isinstance(config, PredictConfig):
            self.dst_path = config.dst_path
            self.model_path = config.model_path
            self.yaml_path = self.dst_path / "data.yaml"
            self.dst_path.mkdir(parents=True, exist_ok=True)
            self._create_yaml()

        elif isinstance(config, EvaluateConfig):
            self.pred_path = config.pred_path
            self.model_path = config.model_path
            self.gt_path = config.gt_path

        else:
            raise ValueError(
                "Invalid config type. Allowed types are TrainConfig, PredictConfig, or EvaluateConfig."
            )

    def train(self) -> None:
        """Train the YOLO model using the provided configuration.
        If `use_kfold` is set to True, performs k-fold cross-validation.
        If `use_kfold` is False, performs a standard single-run training.
        """
        if getattr(self, "use_kfold", False):
            fold_metrics = []
            image_mask_group_quads = []

            # We use all images and masks from train, val, and test splits
            # for the k-fold cross-validation.
            for split in ["train", "val", "test"]:
                img_dir = self.src_path / "images" / split
                mask_dir = self.src_path / "labels" / split
                for fname in sorted(img_dir.iterdir()):
                    if fname.name.lower().endswith((".png", ".jpg", ".jpeg")):
                        base = fname.stem
                        mask_name = f"{base}.txt"
                        img_path = img_dir / f"{base}.png"
                        txt_mask_path = mask_dir / mask_name
                        img_mask_path = mask_dir / f"{base}.png"
                        if not txt_mask_path.exists():
                            raise FileNotFoundError(f"Missing mask for {img_path}")
                        group_id = base.split('_')[0]  # e.g., "P11"
                        image_mask_group_quads.append((img_path, txt_mask_path, img_mask_path, group_id))

            all_img_paths, all_txt_mask_paths, all_img_mask_paths, group_labels = zip(*image_mask_group_quads)
            gkf = GroupKFold(n_splits=self.kfold_n_splits)

            for fold, (train_idx, val_idx) in enumerate(
                gkf.split(all_img_paths, groups=group_labels)
            ):
                train_patients = set(group_labels[i] for i in train_idx)
                val_patients = set(group_labels[i] for i in val_idx)
                print(f"Fold {fold}:")
                print(f"  Train patients: {len(train_patients)}, Val patients: {len(val_patients)}")
                print(f"  Train images: {len(train_idx)}, Val images: {len(val_idx)}")
                
                fold_dir = self.dst_path / f"fold_{fold}"
                for split in ["train", "val"]:
                    (fold_dir / "images" / split).mkdir(parents=True, exist_ok=True)
                    (fold_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

                for idx in train_idx:
                    shutil.copy2(all_img_paths[idx], fold_dir / "images/train")
                    shutil.copy2(all_txt_mask_paths[idx], fold_dir / "labels/train")
                    shutil.copy2(all_img_mask_paths[idx], fold_dir / "labels/train")
                for idx in val_idx:
                    img_path = all_img_paths[idx]
                    txt_mask_path = all_txt_mask_paths[idx]
                    img_mask_path = all_img_mask_paths[idx]
                    shutil.copy2(img_path, fold_dir / "images" / "val" / img_path.name)
                    shutil.copy2(txt_mask_path, fold_dir / "labels" / "val" / txt_mask_path.name)
                    shutil.copy2(img_mask_path, fold_dir / "labels" / "val" / img_mask_path.name)

                # write fold-specific YAML
                data_yaml = {
                    "path": str(fold_dir),
                    "train": "images/train",
                    "val": "images/val",
                    "nc": 1,
                    "names": ["lesion"],
                    "task": "segment",
                }
                fold_yaml = fold_dir / "data.yaml"
                with open(fold_yaml, "w") as f:
                    yaml.dump(data_yaml, f, default_flow_style=False)
                # train model
                model = uYOLO(self.model_path)
                model.train(
                    data=fold_yaml,
                    epochs=self.epochs,
                    batch=self.batch_size,
                    save=True,
                    imgsz=self._get_image_size(),
                    project=fold_dir,
                    device="cuda",
                    verbose=True,
                    # disable autoaugmentation for unet comparison
                    hsv_h=0.0,
                    hsv_s=0.0,
                    hsv_v=0.0,
                    translate=0.0,
                    scale=0.0,
                    fliplr=0.0,
                    mosaic=0.0,
                    erasing=0.0,
                    auto_augment=None,
                    augment=False,
                )

                # evaluate fold
                eval_cfg = EvaluateConfig(
                    net=Net.YOLO,
                    src_path=fold_dir,
                    pred_path=fold_dir / "predictions",
                    model_path=fold_dir / "train" / "weights" / "best.pt",
                    gt_path=fold_dir / "labels" / "val",
                )
                evaluator = YOLO(eval_cfg)

                # predict on validation set
                evaluator.predict(
                    image_dir=fold_dir / "images" / "val",
                    pred_dir=fold_dir / "predictions",
                )
                
                # evaluate predictions
                evaluator.evaluate()
                # read per-fold metrics
                with open(fold_dir / "predictions" / "metrics.json", "r") as mf:
                    metrics = json.load(mf)

                fold_metrics.append(metrics)

                send_whatsapp_message(
                    f"YOLO Fold {fold+1}/{self.kfold_n_splits} completed. "
                    f"Metrics: {json.dumps(metrics, indent=4)}"
                )

            # aggregate and save summary
            summary = {
                k: {
                    "mean": float(np.mean([m[k] for m in fold_metrics])),
                    "std": float(np.std([m[k] for m in fold_metrics])),
                }
                for k in fold_metrics[0]
            }
            with open(self.dst_path / "cv_summary.json", "w") as sf:
                json.dump(summary, sf, indent=4)
            logger.success(
                f"Cross-validation summary saved to {self.dst_path / 'cv_summary.json'}"
            )

            send_whatsapp_message(
                f"YOLO Cross-validation completed. Summary: {json.dumps(summary, indent=4)}"
            )

            return summary

        else:
            # Standard single-run training
            self._create_yaml()
            model = uYOLO(self.model_path)
            model.train(
                data=self.yaml_path,
                epochs=self.epochs,
                batch=self.batch_size,
                save=True,
                imgsz=self._get_image_size(),
                project=self.dst_path,
                device="cuda",
                verbose=True,
                # disable autoaugmentation for unet comparison
                hsv_h=0.0,
                hsv_s=0.0,
                hsv_v=0.0,
                translate=0.0,
                scale=0.0,
                fliplr=0.0,
                mosaic=0.0,
                erasing=0.0,
                auto_augment=None,
                augment=False,
            )

    def predict(self, image_dir: Optional[Path] = None, pred_dir: Optional[Path] = None):
        """Run inference on images using the trained YOLO model.
        Args:
            image_dir (str, optional): Directory containing images to predict. Defaults to None.
            pred_dir (str, optional): Directory to save predictions. Defaults to None.
        """
        model = uYOLO(self.model_path)

        if image_dir is None:
            image_dir = self.src_path / "images" / "test"

        if pred_dir is None:
            pred_dir = self.dst_path

        if os.path.exists(pred_dir):
            logger.info(f"Prediction folder already exists: {pred_dir}. Removing it.")
            shutil.rmtree(pred_dir)

        pred_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            [
                f
                for f in image_dir.iterdir()
                if f.suffix.lower() in (".png", ".jpg", ".jpeg")
            ]
        )
        inference_times = []

        for img_file in image_files:
            start = time.perf_counter()
            model.predict(
                source=img_file,
                project=pred_dir,
                name=".",  # disables subfolder creation
                save_txt=True,
                save_conf=True,
                save_crop=False,
                device="cuda",
                exist_ok=True,
                conf=0.25,
            )
            elapsed = time.perf_counter() - start
            inference_times.append(elapsed)

        with open(pred_dir / "inference_times.json", "w") as f:
            json.dump(inference_times, f, indent=4)

        logger.success(f"Inference times saved to {pred_dir / 'inference_times.json'}")

        self._draw_predictions(pred_dir=pred_dir, image_dir=image_dir)

    def evaluate(self) -> dict:
        """
        Evaluate the YOLO predictions using the ground truth masks on a per-image basis.
        Computes IoU, Dice score, precision, recall, F1 score, specificity, and inference time for each image,
        then returns the average of each metric across all images. Results are saved in JSON format.
        """
        logger.info(f"Evaluating YOLO predictions in folder {self.pred_path}")

        # Load inference times
        times_path = self.pred_path / "inference_times.json"
        with open(times_path, 'r') as f:
            times = json.load(f)

        mask_dir = self.pred_path / "masks"
        image_names = sorted(mask_dir.iterdir(), key=lambda x: x.name)

        # Lists for per-image metrics
        iou_list = []
        dice_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        specificity_list = []
        time_list = []

        for name in image_names:
            # Load predicted and ground truth masks
            pred_mask = cv2.imread(str(mask_dir / name.name), cv2.IMREAD_GRAYSCALE)
            gt_mask   = cv2.imread(str(self.gt_path / name.name), cv2.IMREAD_GRAYSCALE)
            
            # if pred_mask is None or gt_mask is None:
            #     logger.warning(f"Skipping {name}: missing prediction or ground truth.")
            #     continue

            # Binarize masks
            pred_bin = (pred_mask > 127).astype(np.uint8)
            gt_bin   = (gt_mask   > 127).astype(np.uint8)

            # Compute pixel-level confusion
            tp = np.logical_and(pred_bin, gt_bin).sum()
            fp = np.logical_and(pred_bin, np.logical_not(gt_bin)).sum()
            fn = np.logical_and(np.logical_not(pred_bin), gt_bin).sum()
            tn = np.logical_and(np.logical_not(pred_bin), np.logical_not(gt_bin)).sum()

            # Compute metrics per image
            iou         = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
            dice        = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
            precision   = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall      = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            f1_score    = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0

            # Append metrics
            iou_list.append(iou)
            dice_list.append(dice)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1_score)
            specificity_list.append(specificity)

            # Inference time per image
            idx = image_names.index(name)
            time_list.append(times[idx] if idx < len(times) else 0.0)

        # Compute average metrics
        avg_metrics = {
            "iou": float(np.mean(iou_list)),
            "dice_score": float(np.mean(dice_list)),
            "precision": float(np.mean(precision_list)),
            "recall": float(np.mean(recall_list)),
            "f1_score": float(np.mean(f1_list)),
            "specificity": float(np.mean(specificity_list)),
            "inference_time": float(np.mean(time_list)),
        }

        # Save to JSON
        metrics_path = self.pred_path / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(avg_metrics, f, indent=4)
        logger.success(f"Metrics saved to {metrics_path}")

        logger.info("Average metrics:")
        for k, v in avg_metrics.items():
            logger.info(f"{k}: {v:.4f}")

        return avg_metrics
    # ------------------------- Private Methods -------------------------
    def _create_yaml(self) -> None:
        """Create a data.yaml file for YOLO training or prediction.
        This file contains paths to training and validation images, number of classes,
        class names, and task type.
        """
        data_yaml = {
            "path": str(self.src_path),
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": ["lesion"],
            "task": "segment",
        }
        with open(self.yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

    def _draw_predictions(self, pred_dir: Path, image_dir: Path) -> None:
        """Draw predictions on images and save them as masks.
        Args:
            pred_dir (str): Directory where predictions are stored.
            image_dir (str): Directory containing the images to draw predictions on.
        """
        output_dir = pred_dir / "masks"
        output_dir.mkdir(parents=True, exist_ok=True)

        prediction_dir = pred_dir / "labels"
        if not prediction_dir.exists():
            raise FileNotFoundError(
                f"Prediction directory {prediction_dir} does not exist. Did you run the predict step?"
            )

        image_files = sorted(
            [f for f in image_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
        )

        for idx, image_file in enumerate(image_files):
            base_name = image_file.stem
            prediction_path = prediction_dir / f"{base_name}.txt"
            image_path = image_dir / image_file.name

            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Could not load image {image_path}")
                continue

            img_size = self._get_image_size()
            img = cv2.resize(img, img_size[:2])
            mask = np.zeros_like(img)

            if not os.path.exists(prediction_path):
                logger.warning(f"Prediction file not found for {image_file}. Saving empty mask.")
            else:
                with open(prediction_path, "r") as f:
                    lines = f.readlines()

                if not lines:
                    # If the prediction file is empty, log a warning and save an empty mask
                    # we want to save an empty mask to avoid a bias in evaluation
                    logger.info(f"No predictions for {image_file}. Saving empty mask.")
                else:
                    for line in lines:
                        values = list(map(float, line.split()))
                        if len(values) > 2:
                            coords = values[1:]
                            if len(coords) % 2 != 0:
                                coords = coords[:-1]
                            if len(coords) >= 4:
                                points = np.array(coords).reshape(-1, 2)
                                points[:, 0] *= img_size[1]
                                points[:, 1] *= img_size[0]
                                points = points.astype(np.int32)
                                cv2.fillPoly(mask, [points], color=255)

            cv2.imwrite(os.path.join(output_dir, f"{base_name}.png"), mask)

        logger.success(f"Predictions drawn and saved correctly to {output_dir}")

    def _get_image_size(self) -> tuple[int, int, int]:
        """Get the image size from the data.yaml file.
        Returns:
            tuple: (height, width, channels) of the first image in the training set.
        """
        train_path = self.src_path / "images" / "train"
        image_files = list(train_path.iterdir())
        if not image_files:
            raise FileNotFoundError(f"No images found in {train_path}")
        
        first_image = image_files[0]
        img = cv2.imread(str(first_image))
        if img is None:
            raise ValueError(f"Could not load image {first_image}")
        
        return img.shape  # (height, width, channels)