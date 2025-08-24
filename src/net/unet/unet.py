import json
import math
import time
from pathlib import Path
from typing import List, Optional, Type

import cv2
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from schemas.pipeline_schemas import (
    Config,
    EvaluateConfig,
    PredictConfig,
    SegmentationMetrics,
    TrainConfig,
)
from steps.evaluation.performance.metrics import MetricsCalculator
from utils.send_msg import send_whatsapp_message


class MRIDataset(Dataset):
    def __init__(self, img_dir: Path, mask_dir: Path|None = None, cache_size: int = 100) -> None:
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.cache_size = cache_size
        self.cache = {}  # simple cache
        self.check_directories()

        # Only load valid image files
        image_files = sorted([p for p in self.img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])

        if self.mask_dir is None:
            self.images = image_files
            self.masks = None
        else:
            # Align images and masks by filename (stem)
            masks_map = {p.name: p for p in self.mask_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
            pairs = [(img, masks_map.get(img.name, None)) for img in image_files]
            # keep only those with a mask
            pairs = [(img, m) for img, m in pairs if m is not None]
            self.images = [img for img, _ in pairs]
            self.masks  = [m for _, m in pairs]

    def check_directories(self):
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory {self.img_dir} does not exist.")
        if self.mask_dir and not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory {self.mask_dir} does not exist.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        cache_key = f"img_{idx}"

        if cache_key in self.cache:
            img = self.cache[cache_key]
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = img

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        if self.mask_dir:
            if not self.masks:
                raise ValueError(f"Mask directory {self.masks} has no valid masks (masks is empty)")

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
        metrics_calculator_cls: Type[MetricsCalculator] | None= None,
    ) -> None:
        # Common initialization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_name = "resnet34"
        self.config = config
        self.conf = 0.5
        
        self._metrics_calculator_cls = metrics_calculator_cls or MetricsCalculator
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
        if self._metrics_calculator is None:
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

    def _init_prediction(self, config: PredictConfig) -> None:
        """Initialize for prediction mode.

        Defaults to predicting on the 'test' split. You can optionally add
        `split` to PredictConfig (e.g., 'test', 'train', or 'val') to override.
        For CV use-cases, we usually skip the standalone predict step and
        use `from_kfold(...).predict()` on each fold's val subset.
        """
        self.src_path = config.src_path
        self.dst_path = config.dst_path
        self.model_path = config.model_path

        # Optional override (backward compatible if your schema doesn't have it)
        split = getattr(config, "split", "test")

        # Resolve images_dir based on split and what's materialized on disk
        images_dir = self.src_path / "images" / split
        if not images_dir.exists():
            # Try final_retrain val as a sensible fallback for ad-hoc checks
            fr_val = self.src_path / "final_retrain" / "images" / "val"
            if split == "test" and fr_val.exists():
                logger.warning(
                    "'images/test' not found. Falling back to final_retrain/images/val."
                )
                images_dir = fr_val
            else:
                # Last resort: train
                train_dir = self.src_path / "images" / "train"
                if train_dir.exists():
                    logger.warning(
                        f"Images dir for split '{split}' not found at {images_dir}. "
                        f"Falling back to 'images/train'."
                    )
                    images_dir = train_dir
                else:
                    raise FileNotFoundError(
                        f"Could not find images dir for split '{split}' at {images_dir}, "
                        f"and no fallback directory exists."
                    )

        # Initialize model and load weights
        self.model = self._create_model()
        self._load_model()

        # Build the loader for prediction (no masks needed)
        self.test_loader = self._get_dataloader(
            MRIDataset(images_dir),  # masks=None
            batch_size=1,
            shuffle=False,
        )

        # Ensure destination exists
        self.dst_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Prediction initialized. Split='{split}', images_dir={images_dir}, dst={self.dst_path}")


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
        """Initialize for evaluation mode on the split inferred from gt_path (e.g., 'test')."""
        self.src_path  = config.src_path
        self.model_path = config.model_path
        self.pred_path  = config.pred_path
        self.gt_path    = config.gt_path

        self.plots_dir = self.pred_path.parent / "plots"
        self.metrics: SegmentationMetrics | None = None

        split = Path(self.gt_path).name  # expects .../labels/<split>
        if (self.src_path / "images" / split).exists():
            images_dir = self.src_path / "images" / split
            labels_dir = self.src_path / "labels" / split
        else:
            raise FileNotFoundError(
                f"Could not infer images dir for split '{split}'. "
                f"Expected at {self.src_path / 'images' / split}"
            )

        self.model = self._create_model()
        self._load_model()

        self.val_loader = self._get_dataloader(
            MRIDataset(images_dir, labels_dir),
            batch_size=1,
            shuffle=False
        )

    def _setup_training_dataloaders(self) -> None:
        """
        Prefer 'final_retrain/{images,labels}/{train,val}' if present.
        Fallback to 'images/{train,val}' at dataset root.
        """
        fr_root = self.src_path / "final_retrain"
        if (fr_root / "images" / "train").exists() and (fr_root / "images" / "val").exists():
            train_img = fr_root / "images" / "train"
            train_lbl = fr_root / "labels" / "train"
            val_img   = fr_root / "images" / "val"
            val_lbl   = fr_root / "labels" / "val"
        else:
            train_img = self.src_path / "images" / "train"
            train_lbl = self.src_path / "labels" / "train"
            val_img   = self.src_path / "images" / "val"
            val_lbl   = self.src_path / "labels" / "val"

        self.train_loader = self._get_dataloader(
            MRIDataset(train_img, train_lbl),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.val_loader = self._get_dataloader(
            MRIDataset(val_img, val_lbl),
            batch_size=self.batch_size,
            shuffle=False,
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

    def train(self) -> Optional[dict]:
        """
        Train the U-Net model using the specified training configuration.
        If k-fold cross-validation is enabled, it will perform k-fold training.
        Otherwise, it will train on the provided training and validation datasets.
        """
        logger.info(f"Training model {self.model_name}")
        self.scaler = torch.GradScaler()

        return self._run_kfold() if self.use_kfold else self._run_single_training()

    def _run_kfold(self) -> dict:
        """
        If 'cv_folds/fold_*' exists under src_path, use the materialized folds.
        Otherwise, fallback to GroupKFold over images/train grouped by patient.
        """
        cv_root = self.src_path / "cv_folds"
        if cv_root.exists():
            logger.info(f"[U-Net] Using materialized folds at: {cv_root}")
            fold_metrics, fold_train_hist, fold_val_hist = [], [], []
            # Discover folds in numeric order
            fold_dirs = sorted([p for p in cv_root.iterdir() if p.is_dir() and p.name.startswith("fold_")],
                            key=lambda p: int(p.name.split("_")[-1]))
            if not fold_dirs:
                raise RuntimeError(f"No folds found in {cv_root}")

            for i, fold_dir in enumerate(fold_dirs, start=1):
                logger.info(f"[U-Net] Fold {i}: {fold_dir.name}")
                tr_img = fold_dir / "images" / "train"
                tr_lbl = fold_dir / "labels" / "train"
                va_img = fold_dir / "images" / "val"
                va_lbl = fold_dir / "labels" / "val"

                train_loader = self._get_dataloader(
                    MRIDataset(tr_img, tr_lbl),
                    batch_size=self.batch_size,
                    shuffle=True,
                )
                val_loader = self._get_dataloader(
                    MRIDataset(va_img, va_lbl),
                    batch_size=self.batch_size,
                    shuffle=False,
                )

                # Create a fold-specific instance writing under dst_path/fold_i
                fold_model = UNet.from_kfold(self.config, i, train_loader, val_loader)
                fold_model.scaler = self.scaler
                fold_model.train()

                fold_train_hist.append(fold_model.train_losses)
                fold_val_hist.append(fold_model.val_losses)

                fold_model.predict()      # runs on its val set
                metrics = fold_model.evaluate()
                fold_metrics.append(metrics)

                send_whatsapp_message(
                    f"U-Net Fold {i}/{len(fold_dirs)} done. Metrics: {metrics}"
                )

            self._plot_kfolds_curve(fold_train_hist, fold_val_hist, self.model_name)
            summary = self.metrics_calculator.kfold_summary(fold_metrics)
            self.metrics_calculator.write_kfold_summary(summary, self.dst_path / "kfold_summary.json")
            send_whatsapp_message(f"U-Net Cross-validation completed. Summary: {summary}")
            return summary

        # ---------- Fallback (previous patient-group KFold over images/train) ----------
        logger.info("[U-Net] Materialized folds not found. Falling back to GroupKFold over images/train.")
        train_ds = MRIDataset(self.src_path / "images" / "train", self.src_path / "labels" / "train")
        group_labels = [p.name.split("_")[0] for p in train_ds.images]
        if len(group_labels) != len(train_ds):
            raise RuntimeError("Group labels and dataset size mismatch.")
        gkf = GroupKFold(n_splits=self.kfold_n_splits)
        fold_metrics, fold_train_hist, fold_val_hist = [], [], []

        for fold, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_ds)), groups=group_labels), start=1):
            logger.info(f"[U-Net] Starting fold {fold}/{self.kfold_n_splits}")
            tr_subset  = Subset(train_ds, tr_idx.tolist())
            val_subset = Subset(train_ds, val_idx.tolist())
            train_loader = self._get_dataloader(tr_subset, batch_size=self.batch_size, shuffle=True)
            val_loader   = self._get_dataloader(val_subset, batch_size=self.batch_size, shuffle=False)
            fold_model = UNet.from_kfold(self.config, fold, train_loader, val_loader)
            fold_model.scaler = self.scaler
            fold_model.train()
            fold_train_hist.append(fold_model.train_losses)
            fold_val_hist.append(fold_model.val_losses)
            fold_model.predict()
            metrics = fold_model.evaluate()
            fold_metrics.append(metrics)
            send_whatsapp_message(f"U-Net Fold {fold}/{self.kfold_n_splits} done. Metrics: {metrics}")

        self._plot_kfolds_curve(fold_train_hist, fold_val_hist, self.model_name)
        summary = self.metrics_calculator.kfold_summary(fold_metrics)
        self.metrics_calculator.write_kfold_summary(summary, self.dst_path / "kfold_summary.json")
        send_whatsapp_message(f"U-Net Cross-validation completed. Summary: {summary}")
        return summary


    def _run_single_training(self) -> None:
        """Run standard training without k-fold cross-validation."""
        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = 10

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            valid_batches = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")

            for batch_idx, (images, masks, _) in enumerate(pbar):
                images, masks = images.to(self.device), masks.to(self.device)
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    
                    # Verificar loss válido antes de backprop
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"Invalid loss at epoch {epoch+1}, batch {batch_idx}: {loss.item()}")
                        continue

                self.scaler.scale(loss).backward()
                
                # Gradient clipping para estabilidad
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()
                valid_batches += 1

                if batch_idx % 10 == 0:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Calcular loss promedio del epoch
            avg_train_loss = epoch_loss / max(valid_batches, 1)
            
            # Validación
            self.model.eval()
            val_loss = self.validate()

            # Actualizar scheduler solo si val_loss es válido
            if not math.isnan(val_loss) and not math.isinf(val_loss):
                self.scheduler.step(val_loss)
            else:
                logger.warning(f"Invalid validation loss at epoch {epoch+1}: {val_loss}")
                val_loss = float('inf')  # Tratar como muy malo

            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)

            logger.info(
                f"Epoch [{epoch+1}/{self.epochs}], "
                f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            # Guardar mejor modelo con mejor lógica
            if val_loss < best_val_loss and not math.isnan(val_loss):
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)
                logger.info(f"New best model saved (val_loss: {val_loss:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        self._plot_loss_curve()


    @classmethod
    def from_kfold(cls, config: Config, fold: int, train_loader, val_loader) -> "UNet":
        assert isinstance(config, TrainConfig), "Invalid config class"

        instance = cls.__new__(cls)
        instance._metrics_calculator_cls = MetricsCalculator
        instance._metrics_calculator = None
        instance.conf = 0.5
        instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        instance.encoder_name = "resnet34"
        instance.config = config
        instance.batch_size = config.batch_size
        instance.epochs = config.epochs
        instance.learning_rate = config.learning_rate
        instance.use_kfold = False  # important: run single-training loop
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
        instance.val_loader   = val_loader
        instance.test_loader  = val_loader  # predict/eval on fold's validation subset

        instance.model = smp.Unet(
            encoder_name=instance.encoder_name,
            in_channels=1,
            classes=1,
            activation=None
        ).to(instance.device)

        instance.criterion = BCEDiceLoss()
        instance.optimizer = optim.Adam(instance.model.parameters(), lr=instance.learning_rate)
        instance.scheduler = optim.lr_scheduler.ReduceLROnPlateau(instance.optimizer, mode="min", factor=0.1, patience=5)
        instance.scaler = torch.cuda.amp.GradScaler()
        instance.train_losses, instance.val_losses = [], []
        return instance

    def validate(self) -> float:
        """Validate the model on the validation dataset and return the average loss."""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for images, masks, _ in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()
                    num_batches += 1
                else:
                    logger.warning(f"Invalid loss detected: {loss.item()}")
        return val_loss / max(num_batches, 1)

    def evaluate(self) -> SegmentationMetrics:
        """Evaluate model with consistent dimension handling."""
        logger.info(f"Evaluating model {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
            
        self.model.load_state_dict(torch.load(str(self.model_path), map_location=self.device))
        self.model.to(self.device).eval()

        all_per_image: list[dict] = []
        all_metrics = {k: [] for k in ["iou", "dice_score", "precision", "recall", "specificity", "inference_time"]}
        self.pred_path.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for images, masks, *_ in tqdm(self.val_loader, desc="Evaluating"):
                images, masks = images.to(self.device), masks.to(self.device)

                pred_masks, batch_time = self._infer_and_time(images)
                
                if pred_masks.ndim == 4 and pred_masks.shape[1] == 1:
                    pred_masks = pred_masks[:, 0, :, :]
                
                gt = masks.cpu().numpy()
                if gt.ndim == 4 and gt.shape[1] == 1:
                    gt = gt[:, 0, :, :]
                
                if pred_masks.shape != gt.shape:
                    logger.error(f"Shape mismatch: pred {pred_masks.shape} vs gt {gt.shape}")
                    continue

                time_per_image = batch_time / images.shape[0] if images.shape[0] > 0 else 0.0

                try:
                    avg_batch, per_img = self.metrics_calculator.evaluate_batch(
                        preds=pred_masks,
                        gts=gt,
                        times=[time_per_image] * images.shape[0],
                        threshold=0.5
                    )

                    areas = gt.reshape(gt.shape[0], -1).sum(axis=1).astype(float).tolist()
                    for i, per_img_metric in enumerate(per_img):
                        if i < len(areas):
                            per_img_metric["lesion_area_px"] = areas[i]

                    all_per_image.extend(per_img)
                    for key in all_metrics:
                        if key in avg_batch:
                            all_metrics[key].append(avg_batch[key])
                            
                except Exception as e:
                    logger.error(f"Error in batch evaluation: {e}")
                    continue

        final_metrics = {}
        for key, values in all_metrics.items():
            if values:
                final_metrics[key] = float(np.nanmean(values))
            else:
                logger.warning(f"No valid values for metric: {key}")
                final_metrics[key] = 0.0

        self.metrics = SegmentationMetrics(**final_metrics)

        out_dir = self.pred_path if getattr(self, "pred_path", None) else self.dst_path
        out_dir.mkdir(parents=True, exist_ok=True)

        self.metrics.write_to_file(out_dir / "metrics.json")
        
        if all_per_image:
            (out_dir / "per_image_metrics.json").write_text(json.dumps(all_per_image, indent=2))

        logger.info(f"Evaluation completed. Metrics: {self.metrics}")
        return self.metrics

    def predict(self) -> None:
        """
        Predict masks for the configured dataset loader.
        For CV, saves into self.pred_path. For final predict mode, into self.dst_path.
        """
        logger.info("Predicting...")
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

        # In CV (from_kfold), write into predictions dir
        out_dir = self.pred_path if getattr(self, "pred_path", None) else self.dst_path
        out_dir.mkdir(parents=True, exist_ok=True)

        for _, batch in enumerate(tqdm(self.test_loader, desc="Predicting")):
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
                cv2.imwrite(str(out_dir / image_names[j]), pred_img)

        logger.info(f"Predictions saved to: {out_dir}")

    # ------------------------- Private Methods -------------------------
    def _load_model(self) -> None:
        """Load pre-trained model"""
        self.model.load_state_dict(
            torch.load(str(self.model_path), map_location=self.device)
        )
        self.model.eval()

    def _get_dataloader(self, dataset: Dataset, batch_size: int, shuffle=False, num_workers=None) -> DataLoader:
        if num_workers is None:
            num_workers = min(4, max(0, torch.get_num_threads()))
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=False,  # keep all samples for stable metrics
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
