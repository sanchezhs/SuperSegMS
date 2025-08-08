import json
import math
import time
from pathlib import Path
from typing import List, Type

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
from steps.evaluation.performance.metrics import MetricsCalculator
from utils.send_msg import send_whatsapp_message


class MRIDataset(Dataset):
    def __init__(self, img_dir: Path, mask_dir: Path = None, cache_size: int = 100) -> None:
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.cache_size = cache_size
        self.cache = {} # LRU cache
        self.check_directories()
        self.images = sorted(self.img_dir.iterdir())
        self.masks = sorted(self.mask_dir.iterdir()) if mask_dir else None

    def _get_image_shapes(self):
        """Pre-compute image shapes to avoid repeated cv2.imread calls"""
        shapes = {}
        for img_path in self.images:
            # Read just the header to get shape info
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            shapes[img_path.name] = img.shape
        return shapes

    def check_directories(self):
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory {self.img_dir} does not exist.")
        if self.mask_dir and not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory {self.mask_dir} does not exist.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        
        # Check cache first
        cache_key = f"img_{idx}"
        if cache_key in self.cache:
            img = self.cache[cache_key]
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            # Add to cache if under limit
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = img
        
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Specify dtype

        if self.mask_dir:
            mask_path = self.masks[idx]
            mask_cache_key = f"mask_{idx}"
            
            if mask_cache_key in self.cache:
                mask = self.cache[mask_cache_key]
            else:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                if len(self.cache) < self.cache_size:
                    self.cache[mask_cache_key] = mask
            
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            return img, mask, img_path.name

        return img, img_path.name

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE loss
        bce_loss = self.bce(inputs, targets)
        
        # Dice loss with optimized computation
        probs = torch.sigmoid(inputs)
        
        # Flatten tensors more efficiently
        probs_flat = probs.flatten(1)  # Keep batch dimension
        targets_flat = targets.flatten(1)
        
        # Vectorized dice computation
        intersection = torch.sum(probs_flat * targets_flat, dim=1)
        union = torch.sum(probs_flat, dim=1) + torch.sum(targets_flat, dim=1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss



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
        metrics_calculator_cls: Type[MetricsCalculator] = None,
    ) -> None:
        # Common initialization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_name = "resnet34"
        self.config = config
        self.conf = 0.5
        
        self.metrics_calculator_cls = metrics_calculator_cls
        self._metrics_calculator = None

        # Initialize based on config type
        config_handlers = {
            TrainConfig: self._init_training,
            EvaluateConfig: self._init_evaluation,
            PredictConfig: self._init_prediction
        }
        
        handler = config_handlers.get(type(config))
        if handler is None:
            raise ValueError(
                f"Invalid configuration type: {type(config)}. "
                f"Expected one of: {list(config_handlers.keys())}"
            )
        
        handler(config)

    @property
    def metrics_calculator(self) -> "MetricsCalculator":
        """Lazily instantiate and return the metrics calculator."""
        if self._metrics_calculator is None:
            if self._metrics_calculator_cls is None:
                self._metrics_calculator_cls = MetricsCalculator
            self._metrics_calculator = self._metrics_calculator_cls()
        return self._metrics_calculator

    def _create_model(self) -> smp.Unet:
        """Create and return a U-Net model."""
        return smp.Unet(
            encoder_name=self.encoder_name, 
            in_channels=1, 
            classes=1, 
            activation=None
        ).to(self.device)

    def _init_training(self, config: TrainConfig) -> None:
        """Initialize for training mode."""
        # Set training-specific attributes
        self.src_path = config.src_path
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.learning_rate = config.learning_rate
        self.dst_path = config.dst_path
        self.use_kfold = config.use_kfold
        self.kfold_n_splits = config.kfold_n_splits
        self.kfold_seed = config.kfold_seed
        
        # Derived attributes
        self.model_name = self.dst_path.name
        self.model_path = self.dst_path / "models" / f"{self.model_name}.pth"
        
        # Create directories
        (self.dst_path / "models").mkdir(parents=True, exist_ok=True)
        
        # Initialize data loaders (only if not using k-fold)
        if not self.use_kfold:
            self._setup_training_dataloaders()
        
        # Initialize model and training components
        self.model = self._create_model()
        self._setup_training_components()

    def _init_evaluation(self, config: EvaluateConfig) -> None:
        """Initialize for evaluation mode."""
        self.src_path = config.src_path
        self.model_path = config.model_path
        self.pred_path = config.pred_path
        self.gt_path = config.gt_path
        
        # Derived attributes
        self.plots_dir = self.pred_path.parent / "plots"
        self.metrics: SegmentationMetrics = None
        
        # Validation
        if not self.pred_path.exists():
            raise FileNotFoundError(
                f"Prediction directory {self.pred_path} does not exist. "
                "Did you run the prediction step?"
            )
        
        # Initialize model and load weights
        self.model = self._create_model()
        self._load_model()
        
        # Setup validation data loader
        self.val_loader = self._get_dataloader(
            MRIDataset(
                self.src_path / "images" / "val",
                self.src_path / "labels" / "val",
            ),
            batch_size=1
        )

    def _init_prediction(self, config: PredictConfig) -> None:
        """Initialize for prediction mode."""
        self.src_path = config.src_path
        self.dst_path = config.dst_path
        self.model_path = config.model_path
        
        # Initialize model and load weights
        self.model = self._create_model()
        self._load_model()
        
        # Setup test data loader
        self.test_loader = self._get_dataloader(
            MRIDataset(self.src_path / "images" / "test"),
            batch_size=1,
        )

    def _setup_training_dataloaders(self) -> None:
        """Setup training and validation data loaders."""
        self.train_loader = self._get_dataloader(
            MRIDataset(
                self.src_path / "images" / "train",
                self.src_path / "labels" / "train",
            ),
            batch_size=self.batch_size,
        )
        self.val_loader = self._get_dataloader(
            MRIDataset(
                self.src_path / "images" / "val",
                self.src_path / "labels" / "val",
            ),
            batch_size=self.batch_size,
        )

    def _setup_training_components(self) -> None:
        """Setup training components (optimizer, scheduler, etc.)."""
        self.criterion = BCEDiceLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=5
        )
        self.train_losses = []
        self.val_losses = []

    def train(self) -> None:
        """
        Train the U-Net model using the specified training configuration.
        If k-fold cross-validation is enabled, it will perform k-fold training.
        Otherwise, it will train on the provided training and validation datasets.
        """
        logger.info(f"Training model {self.model_name}")
        self.scaler = torch.GradScaler()

        return self._run_kfold() if self.use_kfold else self._run_single_training()

    def _run_kfold(self) -> dict:
        """Run k-fold cross-validation for training the U-Net model.
        This method initializes the dataset, performs k-fold splitting based on patient IDs,
        trains the model on each fold, evaluates it, and collects metrics.
        Returns:
            dict: A dictionary containing mean and standard deviation of metrics across all folds.
        """
        datasets = [
            MRIDataset(
                self.src_path / "images" / s,
                self.src_path / "labels" / s,
            ) for s in ["train", "val", "test"]
        ]
        dataset = ConcatDataset(datasets)

        # Extract group labels (e.g., patient IDs)
        group_labels = []
        for ds in datasets:
            for img in ds.images:
                pid = img.name.split("_")[0]
                group_labels.append(pid)

        gkf = GroupKFold(n_splits=self.kfold_n_splits)
        fold_metrics, fold_train_hist, fold_val_hist = [], [], []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(np.arange(len(dataset)), groups=group_labels)):
            logger.info(f"Starting fold {fold + 1}/{self.kfold_n_splits}")
            train_loader = self._get_dataloader(
                Subset(dataset, train_idx),
                batch_size=self.batch_size,
            )
            val_loader = self._get_dataloader(
                Subset(dataset, val_idx),
                batch_size=self.batch_size,
            )

            fold_model = UNet.from_kfold(self.config, fold, train_loader, val_loader)
            fold_model.scaler = self.scaler
            fold_model.train()

            fold_train_hist.append(fold_model.train_losses)
            fold_val_hist.append(fold_model.val_losses)

            fold_model.predict()
            metrics = fold_model.evaluate()
            fold_metrics.append(metrics)

            send_whatsapp_message(f"U-Net Fold {fold+1} done. Metrics: {metrics}")

        self._plot_kfolds_curve(fold_train_hist, fold_val_hist, self.model_name)
        summary = self.metrics_calculator.kfold_summary(fold_metrics)
        self.metrics_calculator.write_kfold_summary(
            summary, self.dst_path / "kfold_summary.json"
        )
        send_whatsapp_message(f"U-Net Cross-validation completed. Summary: {summary}")
        return summary


    def _run_single_training(self) -> None:
        """Run standard training without k-fold cross-validation.
        This method trains the U-Net model on the provided training and validation datasets,
        evaluates it after each epoch, and saves the best model based on validation loss.
        """
        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")

            for batch_idx, (images, masks, _) in enumerate(pbar):
                images, masks = images.to(self.device), masks.to(self.device)
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()

                if batch_idx % 10 == 0:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            self.model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    val_loss = self.validate()

            self.scheduler.step(val_loss)
            self.train_losses.append(epoch_loss / len(self.train_loader))
            self.val_losses.append(val_loss)

            logger.info(
                f"Epoch [{epoch+1}/{self.epochs}], "
                f"Train Loss: {self.train_losses[-1]:.4f}, Val Loss: {val_loss:.4f}"
            )

            if epoch == 0 or (not math.isnan(val_loss) and val_loss < best_val_loss):
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
        instance.conf = 0.5
        instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        instance.encoder_name = "resnet34"
        instance.config = config
        instance.batch_size = config.batch_size
        instance.epochs = config.epochs
        instance.learning_rate = config.learning_rate
        instance.use_kfold = False
        instance.metrics = None

        base = config.dst_path.name
        instance.dst_path = config.dst_path / f"{base}_fold_{fold}"
        instance.src_path = config.src_path
        instance.pred_path = instance.dst_path / "predictions"
        instance.plots_dir = instance.dst_path / "plots"
        instance.model_name = f"{base}_fold_{fold}"
        instance.model_path = instance.dst_path / "models" / f"{instance.model_name}.pth"
        (instance.dst_path / "models").mkdir(parents=True, exist_ok=True)
        instance.pred_path.mkdir(parents=True, exist_ok=True)
        instance.plots_dir.mkdir(parents=True, exist_ok=True)

        instance.train_loader = train_loader
        instance.val_loader = val_loader
        instance.test_loader = val_loader
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

    def evaluate(self) -> SegmentationMetrics:
        """
        Evaluate the model on the validation dataset by computing metrics per image.
        Computes IoU, Dice, precision, recall, F1 score, specificity, and inference time for each image,
        then returns the average of each metric across all images. Results are saved in JSON format.
        """
        logger.info(f"Evaluating model {self.model_path}")

        # Load model weights and set to eval mode
        self.model.load_state_dict(torch.load(str(self.model_path), map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        all_metrics = {
            "iou": [],
            "dice_score": [],
            "precision": [],
            "recall": [],
            "specificity": [],
            "inference_time": []
        }

        with torch.no_grad():
            for images, masks, *_ in tqdm(self.val_loader, desc="Evaluating"):
                # Move batch to device
                images, masks = images.to(self.device), masks.to(self.device)
                
                # Run inference and measure time
                pred_masks, batch_time = self._infer_and_time(images)
                
                # Compute per-image time and batch metrics
                time_per_image = batch_time / images.shape[0] if images.shape[0] > 0 else 0.0
                batch_metrics = self.metrics_calculator.evaluate_batch(
                    pred_masks, masks.cpu().numpy(), 
                    times=[time_per_image] * images.shape[0], 
                    threshold=self.conf
                )
                
                # Accumulate all metrics
                for key in all_metrics:
                    all_metrics[key].append(batch_metrics[key])

        # Compute averages and create metrics object
        self.metrics = SegmentationMetrics(
            **{key: float(np.mean(values)) for key, values in all_metrics.items()}
        )

        # Save results
        self.metrics.write_to_file(self.pred_path / "metrics.json")

        logger.info(f"Metrics saved in {self.pred_path}/metrics.json")
        logger.info(f"Average metrics:\n{self.metrics}")

        return self.metrics

    def predict(self) -> None:
        """
        Predict segmentation masks for images in the test dataset and save the results.
        This method loads the model weights, performs inference on the test dataset,
        and saves the predicted masks as images in the specified destination path.
        """
        logger.info(f"Predicting images in {self.src_path} and saving to {self.dst_path}")
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        self.model.eval()

        # Kfolds. TODO: IMPROVE
        if self.pred_path:
            self.dst_path = self.pred_path

        for i, batch in enumerate(tqdm(self.test_loader, desc="Predicting")):
            if len(batch) == 3:
                images, _, image_names = batch
            else:
                images, image_names = batch

            images = images.to(self.device)

            with torch.no_grad():
                preds = torch.sigmoid(self.model(images)).cpu().numpy()

            for j in range(images.shape[0]):
                pred_mask = preds[j, 0, :, :]
                pred_img = ((pred_mask > 0.5) * 255).astype(np.uint8)
                output_path = self.dst_path / image_names[j]
                cv2.imwrite(output_path, pred_img)

        logger.info(f"Predictions saved in test directory: {self.dst_path}")

    # ------------------------- Private Methods -------------------------
    def _load_model(self) -> None:
        """Load pre-trained model"""
        self.model.load_state_dict(
            torch.load(str(self.model_path), map_location=self.device)
        )
        self.model.eval()

    def _get_dataloader(self, dataset: Dataset, batch_size: int, shuffle=False, num_workers=None) -> DataLoader:
        """Create optimized DataLoader
        Args:
            dataset (Dataset): The dataset to load.
            batch_size (int): Size of each batch.
            shuffle (bool): Whether to shuffle the dataset.
            num_workers (int, optional): Number of subprocesses to use for data loading.
        """
        if num_workers is None:
            num_workers = min(4, torch.get_num_threads())
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,  # Faster GPU transfer
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2,  # Prefetch batches
            drop_last=True if shuffle else False,  # Consistent batch sizes
        )


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

    def _compute_mean_std_metrics(self, metrics_list: List[SegmentationMetrics]) -> dict:
        """
        Compute mean and standard deviation for each metric across all folds.
        Args:
            metrics_list (list[dict]): List of dictionaries containing metrics from each fold.
        Returns:
            dict: A dictionary with mean and std for each metric.
        """
        if not metrics_list:
            raise ValueError("metrics_list is empty. Cannot compute mean and std.")
        keys = metrics_list[0].model_dump().keys()
        return {
            key: {
                "mean": float(np.mean([m.model_dump()[key] for m in metrics_list])),
                "std": float(np.std([m.model_dump()[key] for m in metrics_list]))
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
