import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Literal

from schemas.pipeline_schemas import TrainConfig, EvaluateConfig, PredictConfig


class MRIDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str = None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir)) if mask_dir else None

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


class UNet:
    def __init__(
        self,
        config: TrainConfig | EvaluateConfig | PredictConfig,
        mode: Literal["train", "evaluate", "predict"],
    ) -> None:
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            self.dst_path = config.dst_path
            self.model_path = os.path.join(self.dst_path, "models", "unet_model.pth")

            os.makedirs(self.dst_path, exist_ok=True)
            os.makedirs(os.path.join(self.dst_path, "models"), exist_ok=True)

            self.train_loader = DataLoader(
                MRIDataset(
                    os.path.join(self.src_path, "images", "train"),
                    os.path.join(self.src_path, "labels", "train"),
                )
            )
            self.val_loader = DataLoader(
                MRIDataset(
                    os.path.join(self.src_path, "images", "val"),
                    os.path.join(self.src_path, "labels", "val"),
                )
            )
            self.model = smp.Unet(
                encoder_name="resnet34", in_channels=1, classes=1, activation=None
            ).to(self.device)

            self.criterion = nn.BCEWithLogitsLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.1, patience=5, verbose=True
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

            self.model = smp.Unet(
                encoder_name="resnet34", in_channels=1, classes=1, activation=None
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
                encoder_name="resnet34", in_channels=1, classes=1, activation=None
            ).to(self.device)
            self._load_model()
            self.test_loader = DataLoader(
                MRIDataset(os.path.join(self.src_path, "images", "test"))
            )

    def train(self) -> None:
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

            print(
                f"Epoch [{epoch + 1}/{self.epochs}], Loss: {epoch_loss / len(self.train_loader):.4f}, Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)
                print(f"New best model saved in {self.model_path}")
        
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
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        self.model.eval()
        metrics_list = []

        for images, masks in self.val_loader:
            images, masks = images.to(self.device), masks.to(self.device)
            with torch.no_grad():
                pred_masks = torch.sigmoid(self.model(images)).cpu().numpy().squeeze()

            iou, dice, precision, recall, f1_score, specificity = (
                self._compute_all_metrics(pred_masks, masks.cpu().numpy().squeeze())
            )

            metrics_list.append(
                {
                    "IoU": iou,
                    "Dice Score": dice,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1_score,
                    "Specificity": specificity,
                }
            )

        avg_metrics = {
            key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]
        }

        self._write_metrics(avg_metrics, format="csv")

        print(f"Average IoU: {avg_metrics['IoU']:.4f}")
        print(f"Average Dice Score: {avg_metrics['Dice Score']:.4f}")
        print(f"Average Precision: {avg_metrics['Precision']:.4f}")
        print(f"Average Recall: {avg_metrics['Recall']:.4f}")
        print(f"Average F1 Score: {avg_metrics['F1 Score']:.4f}")
        print(f"Average Specificity: {avg_metrics['Specificity']:.4f}")

    def predict(self) -> None:
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        self.model.eval()

        os.makedirs(os.path.join(self.dst_path), exist_ok=True)

        for i, (image, image_name) in enumerate(self.test_loader):
            image = image.to(self.device)
            with torch.no_grad():
                pred_mask = torch.sigmoid(self.model(image)).cpu().numpy().squeeze()
            pred_img = ((pred_mask > 0.5) * 255).astype(np.uint8)
            output_path = os.path.join(self.dst_path, image_name[0])
            cv2.imwrite(output_path, pred_img)

        print("Predictions saved in test directory.")

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

    def _write_metrics(self, metrics: dict, format: str = "csv") -> None:
        if format == "csv":
            with open(os.path.join(self.pred_path, "metrics.csv"), "w") as f:
                f.write("Metric,Value\n")
                for key, value in metrics.items():
                    f.write(f"{key},{value}\n")
        else:
            raise ValueError("Invalid format. Only 'csv' is supported.")

        print(f"Metrics saved in {self.pred_path}/metrics.csv")

    def plot_loss_curve(self) -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curve')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.dst_path, "loss_curve.png"))
        print(f"Loss curve saved at {self.dst_path}/loss_curve.png")
