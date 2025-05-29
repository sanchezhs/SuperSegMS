import json
import os
import shutil
import time

import cv2
import numpy as np
import yaml
from loguru import logger
from sklearn.model_selection import GroupKFold
from typing import Optional
from ultralytics import YOLO as uYOLO

from schemas.pipeline_schemas import EvaluateConfig, PredictConfig, TrainConfig, Net
from utils.send_msg import send_whatsapp_message


class YOLO:
    def __init__(self, config: TrainConfig | PredictConfig | EvaluateConfig) -> None:
        self.config = config
        self.src_path = config.src_path

        if not os.path.exists(self.src_path):
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
            self.yaml_path = os.path.join(self.dst_path, "data.yaml")
            self.model_path = "./net/yolo/models/yolo11m-seg.pt"
            os.makedirs(self.dst_path, exist_ok=True)
            if not self.use_kfold:
                self.create_yaml()

        elif isinstance(config, PredictConfig):
            self.dst_path = config.dst_path
            self.model_path = config.model_path
            self.yaml_path = os.path.join(self.dst_path, "data.yaml")
            os.makedirs(self.dst_path, exist_ok=True)
            self.create_yaml()

        elif isinstance(config, EvaluateConfig):
            self.pred_path = config.pred_path
            self.model_path = config.model_path
            self.gt_path = config.gt_path

        else:
            raise ValueError(
                "Invalid config type. Allowed types are TrainConfig, PredictConfig, or EvaluateConfig."
            )

    def train(self) -> None:
        if getattr(self, "use_kfold", False):
            fold_metrics = []
            image_mask_group_quads = []

            for split in ["train", "val"]:
                img_dir = os.path.join(self.src_path, "images", split)
                mask_dir = os.path.join(self.src_path, "labels", split)
                for fname in sorted(os.listdir(img_dir)):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        base = os.path.splitext(fname)[0]
                        mask_name = f"{base}.txt"  # o .png si estás usando segmentación
                        img_path = os.path.join(img_dir, fname)
                        txt_mask_path = os.path.join(mask_dir, mask_name)
                        img_mask_path = os.path.join(mask_dir, f"{base}.png")
                        if not os.path.exists(txt_mask_path):
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
                
                fold_dir = os.path.join(self.dst_path, f"fold_{fold}")
                for split in ["train", "val"]:
                    os.makedirs(os.path.join(fold_dir, "images", split), exist_ok=True)
                    os.makedirs(os.path.join(fold_dir, "labels", split), exist_ok=True)

                for idx in train_idx:
                    shutil.copy2(all_img_paths[idx], os.path.join(fold_dir, "images/train"))
                    shutil.copy2(all_txt_mask_paths[idx], os.path.join(fold_dir, "labels/train"))
                    shutil.copy2(all_img_mask_paths[idx], os.path.join(fold_dir, "labels/train"))
                for idx in val_idx:
                    img_path = all_img_paths[idx]
                    txt_mask_path = all_txt_mask_paths[idx]
                    img_mask_path = all_img_mask_paths[idx]
                    shutil.copy2(img_path, os.path.join(fold_dir, "images", "val", os.path.basename(img_path)))
                    shutil.copy2(txt_mask_path, os.path.join(fold_dir, "labels", "val", os.path.basename(txt_mask_path)))
                    shutil.copy2(img_mask_path, os.path.join(fold_dir, "labels", "val", os.path.basename(img_mask_path)))

                # write fold-specific YAML
                data_yaml = {
                    "path": fold_dir,
                    "train": "images/train",
                    "val": "images/val",
                    "nc": 1,
                    "names": ["lesion"],
                    "task": "segment",
                }
                fold_yaml = os.path.join(fold_dir, "data.yaml")
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
                )

                # evaluate fold
                eval_cfg = EvaluateConfig(
                    net=Net.YOLO,
                    src_path=fold_dir,
                    pred_path=os.path.join(fold_dir, "predictions"),
                    model_path=os.path.join(fold_dir, "train", "weights", "best.pt"),
                    gt_path=os.path.join(fold_dir, "labels", "val"),
                )
                evaluator = YOLO(eval_cfg)

                # predict on validation set
                evaluator.predict(
                    image_dir=os.path.join(fold_dir, "images", "val"),
                    pred_dir=os.path.join(fold_dir, "predictions"),
                )
                
                # evaluate predictions
                evaluator.evaluate()
                # read per-fold metrics
                with open(os.path.join(fold_dir, "predictions", "metrics.json"), "r") as mf:
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
            with open(os.path.join(self.dst_path, "cv_summary.json"), "w") as sf:
                json.dump(summary, sf, indent=4)
            logger.success(
                f"Cross-validation summary saved to {os.path.join(self.dst_path, 'cv_summary.json')}"
            )

            send_whatsapp_message(
                f"YOLO Cross-validation completed. Summary: {json.dumps(summary, indent=4)}"
            )

            return summary

        else:
            # Standard single-run training
            self.create_yaml()
            model = uYOLO(self.model_path)
            model.train(
                data=self.yaml_path,
                epochs=self.epochs,
                batch=self.batch_size,
                save=True,
                imgsz=None,
                project=self.dst_path,
                device="cuda",
                verbose=True,
            )

    def predict(self, image_dir: Optional[str] = None, pred_dir: Optional[str] = None):
        model = uYOLO(self.model_path)

        if image_dir is None:
            image_dir = os.path.join(self.src_path, "images", "test")

        if pred_dir is None:
            pred_dir = os.path.join(self.dst_path, "predict")

        if os.path.exists(pred_dir):
            logger.info(f"Prediction folder already exists: {pred_dir}. Removing it.")
            shutil.rmtree(pred_dir)

        os.makedirs(pred_dir, exist_ok=True)

        image_files = sorted(
            [
                f
                for f in os.listdir(image_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        inference_times = []

        for img_file in image_files:
            image_path = os.path.join(image_dir, img_file)
            start = time.perf_counter()
            model.predict(
                source=image_path,
                project=pred_dir,
                name="",  # disables subfolder creation
                save_txt=True,
                save_conf=True,
                save_crop=False,
                device="cuda",
                exist_ok=True,
            )
            elapsed = time.perf_counter() - start
            inference_times.append(elapsed)

        with open(os.path.join(f"{pred_dir}/predict", "inference_times.json"), "w") as f:
            json.dump(inference_times, f, indent=4)

        logger.success(f"Inference times saved to {os.path.join(pred_dir, 'inference_times.json')}")

        self.draw_predictions(pred_dir=pred_dir, image_dir=image_dir)

    def evaluate(self) -> None:
        """Evaluate the YOLO predictions using the ground truth masks."""
        logger.info(f"Evaluating YOLO predictions in folder {self.pred_path}")

        times_path = os.path.join(self.pred_path, "predict", "inference_times.json")
        if not os.path.exists(times_path):
            raise FileNotFoundError(
                f"Times path {times_path} does not exist. Did you run the predict step?"
            )

        print(f"Loading inference times from {times_path}")
        if not os.path.exists(self.pred_path):
            raise FileNotFoundError(
                f"Prediction path {self.pred_path} does not exist. Did you run the predict step?"
            )

        mask_dir = os.path.join(self.pred_path, "masks")
        image_names = sorted(os.listdir(mask_dir))
        metrics = []

        for name in image_names:
            pred_mask = cv2.imread(os.path.join(mask_dir, name), cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.imread(os.path.join(self.gt_path, name), cv2.IMREAD_GRAYSCALE)

            if pred_mask is None or gt_mask is None:
                logger.warning(
                    f"Skipping {name} due to missing prediction or ground truth."
                )
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

            metrics.append(
                {
                    "image": name,
                    "iou": iou,
                    "dice_score": dice,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "specificity": specificity,
                }
            )

        avg_metrics = {
            key: np.mean([m[key] for m in metrics])
            for key in [
                "iou",
                "dice_score",
                "precision",
                "recall",
                "f1_score",
                "specificity",
            ]
        }

        # with open(times_path, "r") as f:
        #     inference_times = json.load(f)

        # avg_inference_time = float(np.mean(inference_times))
        # avg_metrics["inference_time"] = avg_inference_time

        logger.info("Average metrics:")
        for k, v in avg_metrics.items():
            logger.info(f"{k}: {v:.4f}")

        # Save metrics to a JSON file
        metrics_path = os.path.join(self.pred_path, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(avg_metrics, f, indent=4)
        logger.success(f"Metrics saved to {metrics_path}")
        logger.success("Evaluation completed.")

        return avg_metrics

    def create_yaml(self) -> None:
        data_yaml = {
            "path": os.path.abspath(self.src_path),
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": ["lesion"],
            "task": "segment",
        }
        with open(self.yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

    def draw_predictions(self, pred_dir: str, image_dir: str) -> None:
        output_dir = os.path.join(pred_dir, "masks")
        os.makedirs(output_dir, exist_ok=True)

        prediction_dir = os.path.join(pred_dir, "predict", "labels")
        if not os.path.exists(prediction_dir):
            raise FileNotFoundError(
                f"Prediction directory {prediction_dir} does not exist. Did you run the predict step?"
            )

        image_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg"))]
        )

        for image_file in image_files:
            base_name = os.path.splitext(image_file)[0]
            prediction_path = os.path.join(prediction_dir, f"{base_name}.txt")
            image_path = os.path.join(image_dir, image_file)

            if not os.path.exists(prediction_path):
                logger.warning(f"Prediction not found for {image_file}")
                continue

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Could not load image {image_path}")
                continue

            img_size = self._get_image_size()
            img = cv2.resize(img, img_size[:2])
            mask = np.zeros_like(img)

            with open(prediction_path, "r") as f:
                lines = f.readlines()

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
            logger.info(f"Saved: {os.path.join(output_dir, f'{base_name}.png')}")

        logger.success("Predictions drawn and saved correctly.")


    # def draw_predictions(self, dst_path: Optional[str]) -> None:
    #     if dst_path:
    #         self.dst_path = dst_path
    #         img_val_dir = "val" # If kfold test is val, otherwise it is test
    #     output_dir = os.path.join(self.dst_path, "masks")
    #     os.makedirs(output_dir, exist_ok=True)

    #     logger.info(f"Creadted output directory for image masks: {output_dir}")
    #     prediction_dir = os.path.join(self.dst_path, "predict", "labels")

    #     if not os.path.exists(prediction_dir):
    #         raise FileNotFoundError(
    #             f"Prediction directory {prediction_dir} does not exist. Did you run the predict step?"
    #         )

    #     img_size = self._get_image_size()
    #     image_dir = os.path.join(self.src_path, "images", img_val_dir)
    #     image_files = sorted(
    #         [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg"))]
    #     )

    #     for image_file in image_files:
    #         base_name = os.path.splitext(image_file)[0]
    #         prediction_file = f"{base_name}.txt"

    #         image_path = os.path.join(image_dir, image_file)
    #         prediction_path = os.path.join(prediction_dir, prediction_file)

    #         if not os.path.exists(prediction_path):
    #             logger.info(f"Warning: Prediction not found for {image_file}")
    #             continue

    #         img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #         if img is None:
    #             logger.warning(f"Error: Could not load image {image_path}")
    #             continue

    #         img = cv2.resize(img, img_size[:2])
    #         mask = np.zeros_like(img)

    #         with open(prediction_path, "r") as f:
    #             lines = f.readlines()

    #         for line in lines:
    #             values = list(map(float, line.split()))

    #             if len(values) > 2:
    #                 coords = values[1:]

    #                 if len(coords) % 2 != 0:
    #                     logger.warning(
    #                         f"Odd number of coordinates in {line}, discarding last value."
    #                     )
    #                     coords = coords[:-1]

    #                 if len(coords) >= 4:
    #                     points = np.array(coords).reshape(-1, 2)
    #                     points[:, 0] *= img_size[1]
    #                     points[:, 1] *= img_size[0]
    #                     points = points.astype(np.int32)

    #                     cv2.polylines(
    #                         mask, [points], isClosed=True, color=255, thickness=1
    #                     )
    #                     cv2.fillPoly(mask, [points], color=255)

    #         dst_masks_path = os.path.join(output_dir, f"{base_name}.png")
    #         cv2.imwrite(dst_masks_path, mask)
    #         logger.info(f"Saved: {dst_masks_path}")

    #     logger.success("Predictions drawn and saved correctly.")

    def _get_image_size(self) -> tuple[int, int]:
        """Get the image size from the data.yaml file."""
        path = self.src_path + "/images/train"
        img = os.listdir(path)
        img = os.path.join(path, img[0])
        img = cv2.imread(img)
        return img.shape  # (height, width, channels)
