import json
import os
import time
from typing import Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.model_selection import GroupKFold
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm

from utils.send_msg import send_whatsapp_message
from schemas.pipeline_schemas import (
    EvaluateConfig,
    PredictConfig,
    SegmentationMetrics,
    TrainConfig,
)


class MRIDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str = None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.check_directories()
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir)) if mask_dir else None

    def check_directories(self):
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory {self.img_dir} does not exist.")
        if self.mask_dir and not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Mask directory {self.mask_dir} does not exist.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.img_dir, self.images[idx])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img = torch.tensor(img).unsqueeze(0)

        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, self.masks[idx])
            mask = (
                cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            )
            mask = torch.tensor(mask).unsqueeze(0)
            return img, mask, self.images[idx]

        return img, self.images[idx]

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        probs = torch.sigmoid(inputs)
        smooth = 1e-5

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2. * intersection + smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + smooth)
        dice_loss = 1 - dice.mean()

        return bce_loss + dice_loss


class UNet:
    def __init__(
        self,
        config: TrainConfig | EvaluateConfig | PredictConfig,
        mode: Literal["train", "evaluate", "predict"],
    ) -> None:
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_name = "resnet34"
        self.config = config

        if mode == "train":
            assert isinstance(config, TrainConfig), "Train mode requires a TrainConfig"

            if config.limit_resources:
                torch.backends.cudnn.benchmark = True
                torch.cuda.set_per_process_memory_fraction(
                    0.5, device=self.device.index
                )
                torch.cuda.empty_cache()

            self.src_path = config.src_path
            self.batch_size = config.batch_size
            self.epochs = config.epochs
            self.learning_rate = config.learning_rate
            self.metrics = None
            self.dst_path = config.dst_path
            self.use_kfold = config.use_kfold
            self.kfold_n_splits = config.kfold_n_splits
            self.kfold_seed = config.kfold_seed
            self.model_name = self.dst_path.split("/")[-1]
            self.model_path = os.path.join(self.dst_path, "models", f"{self.model_name}.pth")

            os.makedirs(self.dst_path, exist_ok=True)
            os.makedirs(os.path.join(self.dst_path, "models"), exist_ok=True)

            self.train_loader = DataLoader(
                MRIDataset(
                    os.path.join(self.src_path, "images", "train"),
                    os.path.join(self.src_path, "labels", "train"),
                ),
                num_workers=4,
            )
            self.val_loader = DataLoader(
                MRIDataset(
                    os.path.join(self.src_path, "images", "val"),
                    os.path.join(self.src_path, "labels", "val"),
                ),
                num_workers=4,
            )

            if not self.use_kfold:
                self.train_loader = DataLoader(
                    MRIDataset(
                        os.path.join(self.src_path, "images", "train"),
                        os.path.join(self.src_path, "labels", "train"),
                    ),
                    batch_size=self.batch_size,
                    num_workers=4,
                )
                self.val_loader = DataLoader(
                    MRIDataset(
                        os.path.join(self.src_path, "images", "val"),
                        os.path.join(self.src_path, "labels", "val"),
                    ),
                    batch_size=self.batch_size,
                    num_workers=4,
                )

            self.model = smp.Unet(
                encoder_name=self.encoder_name, in_channels=1, classes=1, activation=None
            ).to(self.device)

            # self.criterion = nn.BCEWithLogitsLoss()
            self.criterion = BCEDiceLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.1, patience=5
            )

            self.train_losses = []
            self.val_losses = []

        elif mode == "evaluate":
            assert isinstance(config, EvaluateConfig), (
                "Evaluate mode requires an EvaluateConfig"
            )
            self.src_path = config.src_path
            self.model_path = config.model_path
            self.pred_path = config.pred_path
            self.gt_path = config.gt_path

            if not os.path.exists(self.pred_path):
                raise FileNotFoundError(
                    f"Prediction directory {self.pred_path} does not exist. Did you run the prediction step?"
                )

            self.model = smp.Unet(
                encoder_name=self.encoder_name, in_channels=1, classes=1, activation=None
            ).to(self.device)
            self._load_model()

            self.val_loader = DataLoader(
                MRIDataset(
                    os.path.join(self.src_path, "images", "val"),
                    os.path.join(self.src_path, "labels", "val"),
                )
            )

        elif mode == "predict":
            assert isinstance(config, PredictConfig), (
                "Predict mode requires a PredictConfig"
            )
            self.src_path = config.src_path
            self.dst_path = config.dst_path
            self.model_path = config.model_path
            self.model = smp.Unet(
                encoder_name=self.encoder_name, in_channels=1, classes=1, activation=None
            ).to(self.device)
            self._load_model()
            self.test_loader = DataLoader(
                MRIDataset(os.path.join(self.src_path, "images", "test"))
            )

    def train(self) -> None:
        logger.info(f"Training model {self.model_name}")

        if getattr(self, 'use_kfold', False):
            # combine train, val
            splits = ["train", "val"]
            datasets = [
                MRIDataset(
                    os.path.join(self.src_path, "images", s),
                    os.path.join(self.src_path, "labels", s)
                ) for s in splits
            ]
            # build group labels from filenames (assumes IDs before underscore)
            group_labels = []
            for ds in datasets:
                for img_name in ds.images:
                    pid = img_name.split('_')[0]
                    group_labels.append(pid)

            dataset = ConcatDataset(datasets)
            gkf = GroupKFold(n_splits=self.kfold_n_splits)
            all_metrics = []

            for fold, (train_idx, val_idx) in enumerate(gkf.split(
                X=np.arange(len(dataset)),
                groups=group_labels
            )):
                logger.info(f"Starting fold {fold+1}/{self.kfold_n_splits}")
                train_loader = DataLoader(
                    Subset(dataset, train_idx),
                    batch_size=self.batch_size,
                    num_workers=2,
                )
                val_loader = DataLoader(
                    Subset(dataset, val_idx),
                    batch_size=self.batch_size,
                    num_workers=2,
                )
                fold_trainer = UNet.from_kfold(
                    self.config, fold, train_loader, val_loader
                )
                fold_trainer.train()
                metrics = fold_trainer.evaluate()
                all_metrics.append(metrics)

                send_whatsapp_message(
                    f"U-Net Fold {fold+1}/{self.kfold_n_splits} completed. "
                    f"Metrics: {metrics}"
                )

            summary = self.compute_mean_std_metrics(all_metrics)
            self.write_summary(summary)
            send_whatsapp_message(
                f"U-Net Cross-validation completed. Summary: {summary}"
            )
            return summary

        # Standard training
        best_val_loss = float("inf")
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for images, masks, _ in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):  
                images, masks = images.to(self.device), masks.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            val_loss = self.validate()
            self.scheduler.step(val_loss)
            self.train_losses.append(epoch_loss/len(self.train_loader))
            self.val_losses.append(val_loss)
            logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss/len(self.train_loader):.4f}, Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)
                logger.info(f"Saved best model to {self.model_path}")
        self._plot_loss_curve()

    @classmethod
    def from_kfold(cls, config: TrainConfig, fold: int, train_loader, val_loader):
        instance = cls.__new__(cls)
        instance.mode = "train"
        instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        instance.encoder_name = "resnet34"
        instance.config = config
        instance.batch_size = config.batch_size
        instance.epochs = config.epochs
        instance.learning_rate = config.learning_rate
        instance.use_kfold = False  

        base = os.path.basename(config.dst_path.rstrip('/'))
        instance.dst_path = os.path.join(config.dst_path, f"{base}_fold_{fold}")
        instance.model_name = f"{base}_fold_{fold}"
        instance.model_path = os.path.join(instance.dst_path, "models", f"{instance.model_name}.pth")
        os.makedirs(instance.dst_path, exist_ok=True)
        os.makedirs(os.path.join(instance.dst_path, "models"), exist_ok=True)

        instance.train_loader = train_loader
        instance.val_loader = val_loader
        instance.model = smp.Unet(
            encoder_name=instance.encoder_name,
            in_channels=1,
            classes=1,
            activation=None
        ).to(instance.device)
        instance.criterion = BCEDiceLoss()
        instance.optimizer = optim.Adam(instance.model.parameters(), lr=instance.learning_rate)
        instance.scheduler = optim.lr_scheduler.ReduceLROnPlateau(instance.optimizer, mode="min", factor=0.1, patience=5)
        instance.train_losses = []
        instance.val_losses = []
        instance.pred_path = instance.dst_path
        return instance

    def train2(self) -> None:
        logger.info(f"Training model {self.model_name}")

        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0

            for images, masks, _ in tqdm(
                self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}"
            ):
                images, masks = images.to(self.device), masks.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            val_loss = self.validate()
            self.scheduler.step(val_loss)

            avg_train_loss = epoch_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)

            logger.info(
                f"Epoch [{epoch + 1}/{self.epochs}], Loss: {epoch_loss / len(self.train_loader):.4f}, Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)
                logger.info(f"New best model saved in {self.model_path}")
        
        self._plot_loss_curve()


    def validate(self) -> float:
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks, _ in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

    def evaluate(self) -> None:
        logger.info(f"Evaluating model {self.model_path}")

        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        self.model.eval()
        metrics_list = []

        for images, masks, _ in tqdm(self.val_loader, desc="Evaluating"):
            images, masks = images.to(self.device), masks.to(self.device)
            pred_masks, inference_time = self._infer_and_time(images)
            iou, dice, precision, recall, f1_score, specificity = (
                self._compute_all_metrics(pred_masks, masks.cpu().numpy().squeeze())
            )
            metrics_list.append(
                {
                    "iou": iou,
                    "dice_score": dice,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "specificity": specificity,
                    "inference_time": inference_time,
                }
            )

        avg_metrics = {
            key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]
        }

        logger.info(f"Metrics saved in {self.pred_path}/metrics.json")
        self._write_metrics(SegmentationMetrics(**avg_metrics), format="json")
        self.metrics = avg_metrics
        logger.info("Average metrics:")
        logger.info(
            "\n\t- ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
        )

        return avg_metrics

    def predict(self) -> None:
        logger.info(f"Predicting images in {self.src_path} and saving to {self.dst_path}")
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        self.model.eval()

        os.makedirs(os.path.join(self.dst_path), exist_ok=True)

        for i, (image, image_name) in enumerate(tqdm(self.test_loader, desc="Predicting")):
            image = image.to(self.device)
            with torch.no_grad():
                pred_mask = torch.sigmoid(self.model(image)).cpu().numpy().squeeze()
            pred_img = ((pred_mask > 0.5) * 255).astype(np.uint8)
            output_path = os.path.join(self.dst_path, image_name[0])
            cv2.imwrite(output_path, pred_img)

        logger.info(f"Predictions saved in test directory: {self.dst_path}")

    def _load_model(self) -> None:
        """Load pre-trained model"""
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        self.model.eval()

    def _compute_all_metrics(
        self, pred, mask
    ) -> tuple[float, float, float, float, float, float]:
        pred_bin = (pred > 0.5).astype(np.uint8)
        mask_bin = (mask > 0.5).astype(np.uint8)

        tp = np.logical_and(pred_bin, mask_bin).sum()
        fp = np.logical_and(pred_bin, np.logical_not(mask_bin)).sum()
        fn = np.logical_and(np.logical_not(pred_bin), mask_bin).sum()
        tn = np.logical_and(np.logical_not(pred_bin), np.logical_not(mask_bin)).sum()

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return iou, dice, precision, recall, f1_score, specificity

    def _write_metrics(self, metrics: SegmentationMetrics, format: str = "json") -> None:
        if format == "csv":
            with open(os.path.join(self.pred_path, "metrics.csv"), "w") as f:
                f.write("Metric,Value\n")
                for key, value in metrics.items():
                    f.write(f"{key},{value}\n")
        else:
            with open(os.path.join(self.pred_path, "metrics.json"), "w") as f:
                json.dump(metrics.model_dump(), f, indent=4)

        logger.info(f"Metrics saved in {self.pred_path}/metrics.{format}")

    def _plot_loss_curve(self) -> None:
        title = self.model_name if self.mode == "train" else "Loss Curve"
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss Curve - {title}')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.dst_path, f"loss_curve_{title}.png"))
        logger.info(f"Loss curve saved at {self.dst_path}/loss_curve.png")

    def _infer_and_time(self, image: torch.Tensor) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        with torch.no_grad():
            pred = torch.sigmoid(self.model(image)).cpu().numpy().squeeze()
        elapsed = time.perf_counter() - start
        return pred, elapsed

    def compute_mean_std_metrics(self, metrics_list):
        keys = metrics_list[0].keys()
        return {
            key: {
                "mean": float(np.mean([m[key] for m in metrics_list])),
                "std": float(np.std([m[key] for m in metrics_list]))
            } for key in keys
        }

    def write_summary(self, metrics_dict):
        summary_path = os.path.join(self.dst_path, "cv_summary.json")
        with open(summary_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        logger.info(f"Cross-validation summary saved at {summary_path}")
        print("\nCross-Validation Summary:\n")
        for key, val in metrics_dict.items():
            print(f"{key:15s}: {val['mean']:.4f} ± {val['std']:.4f}")
