import json
import time
from pathlib import Path

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

from schemas.pipeline_schemas import (
    EvaluateConfig,
    PredictConfig,
    SegmentationMetrics,
    TrainConfig,
)
from utils.send_msg import send_whatsapp_message


class MRIDataset(Dataset):
    def __init__(self, img_dir: Path, mask_dir: Path = None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.check_directories()
        self.images = sorted(self.img_dir.iterdir())
        self.masks = sorted(self.mask_dir.iterdir()) if mask_dir else None

    def check_directories(self):
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory {self.img_dir} does not exist.")
        if self.mask_dir and not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory {self.mask_dir} does not exist.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | None, str]:
        img_path = self.images[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img = torch.tensor(img).unsqueeze(0)

        if self.mask_dir:
            mask_path = self.masks[idx]
            mask = (
                cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            )
            mask = torch.tensor(mask).unsqueeze(0)
            return img, mask, img_path.name

        return img, img_path.name

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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
    """
    U-Net model for MRI segmentation tasks.
    This class handles training, evaluation, and prediction using a U-Net architecture.
    It supports training with k-fold cross-validation, evaluation on validation datasets,
    and prediction on test datasets.
    """
    def __init__(
        self,
        config: TrainConfig | EvaluateConfig | PredictConfig,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_name = "resnet50"
        # self.encoder_name = "mit_b5"
        self.config = config
        self.conf = 0.25

        if isinstance(config, TrainConfig):
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
            self.model_name = self.dst_path.name
            self.model_path = self.dst_path / "models" / f"{self.model_name}.pth"

            (self.dst_path / "models").mkdir(parents=True, exist_ok=True)

            self.train_loader = DataLoader(
                MRIDataset(
                    self.src_path / "images" / "train",
                    self.src_path / "labels" / "train",
                ),
                num_workers=4,
            )
            self.val_loader = DataLoader(
                MRIDataset(
                    self.src_path / "images" / "val",
                    self.src_path / "labels" / "val",
                ),
                num_workers=4,
            )

            if not self.use_kfold:
                self.train_loader = DataLoader(
                    MRIDataset(
                        self.src_path / "images" / "train",
                        self.src_path / "labels" / "train",
                    ),
                    batch_size=self.batch_size,
                    num_workers=4,
                )
                self.val_loader = DataLoader(
                    MRIDataset(
                        self.src_path / "images" / "val",
                        self.src_path / "labels" / "val",
                    ),
                    batch_size=self.batch_size,
                    num_workers=4,
                )

            self.model = smp.Unet(
                encoder_name=self.encoder_name, in_channels=1, classes=1, activation=None
            ).to(self.device)

            self.criterion = BCEDiceLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.1, patience=5
            )
            self.train_losses = []
            self.val_losses = []

        elif isinstance(config, EvaluateConfig):
            self.src_path = config.src_path
            self.model_path = config.model_path
            self.pred_path = config.pred_path
            self.gt_path = config.gt_path

            if not self.pred_path.exists():
                raise FileNotFoundError(
                    f"Prediction directory {self.pred_path} does not exist. Did you run the prediction step?"
                )

            self.model = smp.Unet(
                encoder_name=self.encoder_name, in_channels=1, classes=1, activation=None
            ).to(self.device)
            self._load_model()

            self.val_loader = DataLoader(
                MRIDataset(
                    self.src_path / "images" / "val",
                    self.src_path / "labels" / "val",
                )
            )

        elif isinstance(config, PredictConfig):
            self.src_path = config.src_path
            self.dst_path = config.dst_path
            self.model_path = config.model_path
            self.model = smp.Unet(
                encoder_name=self.encoder_name, in_channels=1, classes=1, activation=None
            ).to(self.device)
            self._load_model()
            self.test_loader = DataLoader(
                MRIDataset(self.src_path / "images" / "test"),
            )
        else:
            raise ValueError(f"Invalid configuration type: {type(config)}. Expected TrainConfig, EvaluateConfig, or PredictConfig.")

    def train(self) -> None:
        """
        Train the U-Net model using the specified training configuration.
        If k-fold cross-validation is enabled, it will perform k-fold training.
        Otherwise, it will train on the provided training and validation datasets.
        """
        logger.info(f"Training model {self.model_name}")

        if self.use_kfold:
            # combine train, val, test
            splits = ["train", "val", "test"]
            datasets = [
                MRIDataset(
                    self.src_path / "images" / s,
                    self.src_path / "labels" / s,
                ) for s in splits
            ]
            # build group labels from filenames (assumes IDs before underscore)
            group_labels = []
            for ds in datasets:
                for img_name in ds.images:
                    pid = img_name.name.split('_')[0]
                    group_labels.append(pid)

            dataset = ConcatDataset(datasets)
            gkf = GroupKFold(n_splits=self.kfold_n_splits)
            all_metrics = []
            all_train_hist = []
            all_val_hist   = []

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
                # Train the fold
                fold_trainer.train()

                # Collect training and validation losses
                all_train_hist.append(fold_trainer.train_losses)
                all_val_hist.append(fold_trainer.val_losses)

                # Predict the fold
                fold_trainer.predict()

                # Evaluate the fold
                metrics = fold_trainer.evaluate()
                all_metrics.append(metrics)

                send_whatsapp_message(
                    f"U-Net Fold {fold+1}/{self.kfold_n_splits} completed. "
                    f"Metrics: {metrics}"
                )
            self._plot_kfolds_curve(
                all_train_hist, all_val_hist, self.model_name
            )
            summary = self._compute_mean_std_metrics(all_metrics)
            self._write_summary(summary)
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
    def from_kfold(cls, config: TrainConfig, fold: int, train_loader, val_loader) -> "UNet":
        """
        Create a U-Net instance for k-fold training.
        Args:
            config (TrainConfig): Configuration for training.
            fold (int): Current fold number.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
        Returns:
            UNet: An instance of the U-Net model configured for k-fold training.
        """
        instance = cls.__new__(cls)
        instance.mode = "train"
        instance.conf = 0.25
        instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        instance.encoder_name = "resnet50"
        # instance.encoder_name = "mit_b5"
        instance.config = config
        instance.batch_size = config.batch_size
        instance.epochs = config.epochs
        instance.learning_rate = config.learning_rate
        instance.use_kfold = False  

        base = config.dst_path.name
        instance.dst_path = config.dst_path / f"{base}_fold_{fold}"
        instance.model_name = f"{base}_fold_{fold}"
        instance.model_path = instance.dst_path / "models" / f"{instance.model_name}.pth"
        (instance.dst_path / "models").mkdir(parents=True, exist_ok=True)

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

    def validate(self) -> float:
        """
        Validate the model on the validation dataset and return the average loss.
        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks, _ in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

    def evaluate(self) -> dict:
        """
        Evaluate the model on the validation dataset by computing metrics per image.
        Computes IoU, Dice, precision, recall, F1 score, specificity, and inference time for each image,
        then returns the average of each metric across all images. Results are saved in JSON format.
        """
        logger.info(f"Evaluating model {self.model_path}")

        # Load model weights and set to eval mode
        self.model.load_state_dict(
            torch.load(str(self.model_path), map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        # Lists to accumulate per-image metrics
        iou_list = []
        dice_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        specificity_list = []
        time_list = []

        with torch.no_grad():
            for images, masks, _ in tqdm(self.val_loader, desc="Evaluating"):
                # Move batch to device
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Run inference and measure time for the whole batch
                pred_masks, batch_time = self._infer_and_time(images)

                # Convert predictions and masks to CPU numpy arrays
                preds = pred_masks
                gts = masks.cpu().numpy()

                # Estimate per-image inference time
                batch_size = preds.shape[0]
                time_per_image = batch_time / batch_size if batch_size > 0 else 0.0

                # Compute metrics for each image in the batch
                for pred_mask, gt_mask in zip(preds, gts):
                    iou, dice, precision, recall, f1_score, specificity = self._compute_all_metrics(
                        pred_mask, gt_mask
                    )

                    # Accumulate metrics
                    iou_list.append(iou)
                    dice_list.append(dice)
                    precision_list.append(precision)
                    recall_list.append(recall)
                    f1_list.append(f1_score)
                    specificity_list.append(specificity)
                    time_list.append(time_per_image)

        # Compute average metrics across all images
        avg_metrics = {
            "iou": float(np.mean(iou_list)),
            "dice_score": float(np.mean(dice_list)),
            "precision": float(np.mean(precision_list)),
            "recall": float(np.mean(recall_list)),
            "f1_score": float(np.mean(f1_list)),
            "specificity": float(np.mean(specificity_list)),
            "inference_time": float(np.mean(time_list)),
        }

        # Save metrics to JSON
        logger.info(f"Metrics saved in {self.pred_path}/metrics.json")
        self._write_metrics(SegmentationMetrics(**avg_metrics), format="json")
        self.metrics = avg_metrics

        logger.info("Average metrics:")
        for k, v in avg_metrics.items():
            logger.info(f"{k}: {v:.4f}")

        return avg_metrics

    def predict(self) -> None:
        """
        Predict segmentation masks for images in the test dataset and save the results.
        This method loads the model weights, performs inference on the test dataset,
        and saves the predicted masks as images in the specified destination path.
        """
        logger.info(f"Predicting images in {self.src_path} and saving to {self.dst_path}")
        self.model.load_state_dict(
            torch.load(str(self.model_path), map_location=self.device)
        )
        self.model.eval()

        if not self.dst_path.exists():
            logger.info(f"Creating destination directory: {self.dst_path}")
            self.dst_path.mkdir(parents=True, exist_ok=True)

        for i, (image, image_name) in enumerate(tqdm(self.test_loader, desc="Predicting")):
            image = image.to(self.device)
            with torch.no_grad():
                pred_mask = torch.sigmoid(self.model(image)).cpu().numpy().squeeze()
            pred_img = ((pred_mask > self.conf) * 255).astype(np.uint8)
            output_path = self.dst_path / f"{image_name[0]}"
            cv2.imwrite(str(output_path), pred_img)

        logger.info(f"Predictions saved in test directory: {self.dst_path}")

    # ------------------------- Private Methods -------------------------
    def _load_model(self) -> None:
        """Load pre-trained model"""
        self.model.load_state_dict(
            torch.load(str(self.model_path), map_location=self.device)
        )
        self.model.eval()

    def _compute_all_metrics(
        self, pred: np.ndarray, mask: np.ndarray
    ) -> tuple[float, float, float, float, float, float]:
        """
        Compute segmentation metrics: IoU, Dice score, precision, recall, F1 score, and specificity.
        Args:
            pred (np.ndarray): Predicted mask.
            mask (np.ndarray): Ground truth mask.
        Returns:
            tuple: A tuple containing IoU, Dice score, precision, recall, F1 score, and specificity.
        """
        pred_bin = (pred > self.conf).astype(np.uint8)
        gt_bin = (mask > 0.5).astype(np.uint8)

        tp = np.logical_and(pred_bin, gt_bin).sum()
        fp = np.logical_and(pred_bin, np.logical_not(gt_bin)).sum()
        fn = np.logical_and(np.logical_not(pred_bin), gt_bin).sum()
        tn = np.logical_and(np.logical_not(pred_bin), np.logical_not(gt_bin)).sum()

        # Compute each metric per image
        iou         = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
        dice        = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
        precision   = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall      = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1_score    = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0

        return iou, dice, precision, recall, f1_score, specificity

    def _write_metrics(self, metrics: SegmentationMetrics, format: str = "json") -> None:
        """
        Write segmentation metrics to a file in the specified format (JSON or CSV).
        Args:
            metrics (SegmentationMetrics): The metrics to write.
            format (str): The format to save the metrics in ("json" or "csv").
        """
        if format == "csv":
            with open(self.pred_path / "metrics.csv", "w") as f:
                f.write("Metric,Value\n")
                for key, value in metrics.items():
                    f.write(f"{key},{value}\n")
        else:
            with open(self.pred_path / "metrics.json", "w") as f:
                json.dump(metrics.model_dump(), f, indent=4)

        logger.info(f"Metrics saved in {self.pred_path}/metrics.{format}")

    def _plot_loss_curve(self) -> None:
        """
        Plot and save the training and validation loss curves.
        This method generates a plot showing the loss over epochs for both training and validation datasets.
        The plot is saved in the destination path with a filename based on the model name.
        """
        title = self.model_name
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss Curve - {title}')
        plt.legend()
        plt.grid()
        plt.savefig(self.dst_path / f"loss_curve_{title}.png")
        logger.info(f"Loss curve saved at {self.dst_path}/loss_curve.png")

    def _plot_kfolds_curve(
        self, train_hist: list[list[float]], val_hist: list[list[float]], model_name: str
    ) -> None:
        plt.figure(figsize=(10,6))
        for i, (tr, val) in enumerate(zip(train_hist, val_hist), start=1):
            plt.plot(tr,      label=f'Fold {i} Train')
            plt.plot(val, '--', label=f'Fold {i} Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'K-Fold Cross-Validation Loss Curves - {model_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.dst_path / f"kfolds_loss_curve_{model_name}.png")
        logger.info(f"K-Fold loss curves saved at {self.dst_path}/kfolds_loss_curve_{model_name}.png")
        plt.close()

    def _infer_and_time(self, images) -> tuple[np.ndarray, float]:
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(images)
            pred_masks = torch.sigmoid(outputs)
        end_time = time.perf_counter()
        return pred_masks.cpu().numpy(), end_time - start_time

    def _compute_mean_std_metrics(self, metrics_list: list[dict]) -> dict:
        """
        Compute mean and standard deviation for each metric across all folds.
        Args:
            metrics_list (list[dict]): List of dictionaries containing metrics from each fold.
        Returns:
            dict: A dictionary with mean and std for each metric.
        """
        keys = metrics_list[0].keys()
        return {
            key: {
                "mean": float(np.mean([m[key] for m in metrics_list])),
                "std": float(np.std([m[key] for m in metrics_list]))
            } for key in keys
        }

    def _write_summary(self, metrics_dict: dict) -> None:
        """
        Write a summary of cross-validation metrics to a JSON file.
        Args:
            metrics_dict (dict): Dictionary containing mean and std for each metric.
        """
        summary_path = self.dst_path / "cv_summary.json"
        with open(summary_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        logger.info(f"Cross-validation summary saved at {summary_path}")
        print("\nCross-Validation Summary:\n")
        for key, val in metrics_dict.items():
            print(f"{key:15s}: {val['mean']:.4f} ± {val['std']:.4f}")
