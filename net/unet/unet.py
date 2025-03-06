import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
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
            return img, mask

        return img


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
                torch.cuda.set_per_process_memory_fraction(0.5, device=self.device.index)
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

        elif mode == "evaluate":
            assert isinstance(config, EvaluateConfig), (
                "Evaluate mode requires an EvaluateConfig"
            )
            self.src_path = config.src_path
            self.model_path = config.model_path

            self.model = smp.Unet(
                encoder_name="resnet34", in_channels=1, classes=1, activation=None
            ).to(self.device)
            self.load_model()

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
            self.load_model()
            self.test_loader = DataLoader(
                MRIDataset(os.path.join(self.src_path, "images", "test"))
            )

    def train(self) -> None:
        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0

            for images, masks in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}"):
                images, masks = images.to(self.device), masks.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            val_loss = self.validate()
            self.scheduler.step(val_loss)

            print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {epoch_loss / len(self.train_loader):.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)
                print(f"Nuevo mejor modelo guardado en {self.model_path}")

    def validate(self) -> float:
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in self.val_loader:
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
        ious, dices = [], []
        all_preds, all_masks = [], []
        for images, masks in self.val_loader:
            images, masks = images.to(self.device), masks.to(self.device)
            with torch.no_grad():
                pred_masks = torch.sigmoid(self.model(images)).cpu().numpy().squeeze()
            iou, dice = self.compute_metrics(pred_masks, masks.cpu().numpy().squeeze())
            ious.append(iou)
            dices.append(dice)
            all_preds.append(pred_masks)
            all_masks.append(masks.cpu().numpy().squeeze())

        print(f"Average IoU: {np.mean(ious):.4f}")
        print(f"Average Dice Score: {np.mean(dices):.4f}")

        # Compute confusion matrix
        all_preds = np.concatenate(all_preds).ravel()
        all_masks = np.concatenate(all_masks).ravel()
        tp = np.sum((all_preds > 0.5) & (all_masks > 0.5))
        tn = np.sum((all_preds <= 0.5) & (all_masks <= 0.5))
        fp = np.sum((all_preds > 0.5) & (all_masks <= 0.5))
        fn = np.sum((all_preds <= 0.5) & (all_masks > 0.5))

        print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    def predict(self) -> None:
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        self.model.eval()
        
        os.makedirs(os.path.join(self.dst_path), exist_ok=True)
        next_patient = 54 # 54 is the first patient in the test set
        for i, image in enumerate(self.test_loader):
            image = image.to(self.device)
            with torch.no_grad():
                pred_mask = torch.sigmoid(self.model(image)).cpu().numpy().squeeze()
            pred_img = ((pred_mask > 0.5) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(self.dst_path, f"P{next_patient}.png"), pred_img)
            next_patient += 1
        print("Predictions saved in test directory.")

    def load_model(self) -> None:
        """Load pre-trained model"""
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        self.model.eval()

    @staticmethod
    def compute_metrics(pred, mask) -> tuple[float, float]:
        pred_bin = (pred > 0.5).astype(np.uint8)
        mask_bin = (mask > 0.5).astype(np.uint8)
        intersection = np.logical_and(pred_bin, mask_bin).sum()
        union = np.logical_or(pred_bin, mask_bin).sum()
        iou = intersection / union if union > 0 else 0
        dice = (
            (2.0 * intersection) / (pred_bin.sum() + mask_bin.sum())
            if (pred_bin.sum() + mask_bin.sum()) > 0
            else 0
        )
        return iou, dice
