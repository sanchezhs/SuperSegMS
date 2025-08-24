import json
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Type

import cv2
import numpy as np
import yaml
from loguru import logger
from sklearn.model_selection import GroupKFold
from ultralytics import YOLO as uYOLO

from schemas.pipeline_schemas import (
    EvaluateConfig,
    Net,
    PredictConfig,
    SegmentationMetrics,
    TrainConfig,
)
from steps.evaluation.performance.metrics import MetricsCalculator
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
    def __init__(
            self, 
            config: TrainConfig | PredictConfig | EvaluateConfig,
            metrics_calculator_cls: Optional[Type[MetricsCalculator]] = None,
        ) -> None:
        self.config = config
        self.src_path = config.src_path

        self._metrics_calculator_cls = metrics_calculator_cls or MetricsCalculator
        self._metrics_calculator = None

        if not self.src_path.exists():
            raise FileNotFoundError(
                f"Source path {self.src_path} does not exist. Did you run the preprocess step?"
            )

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

    def _init_training(self, config: TrainConfig) -> None:
        """Initialize training configuration."""
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

    def _init_prediction(self, config: PredictConfig) -> None:
        """Initialize prediction configuration."""
        self.dst_path = config.dst_path
        self.model_path = config.model_path
        self.yaml_path = self.dst_path / "data.yaml"
        self.dst_path.mkdir(parents=True, exist_ok=True)
        self._create_yaml()

    def _init_evaluation(self, config: EvaluateConfig) -> None:
        """Initialize evaluation configuration."""
        self.pred_path = config.pred_path
        self.model_path = config.model_path
        self.gt_path = config.gt_path

        self.metrics = None
        self.pred_path.mkdir(parents=True, exist_ok=True)

        if not self.pred_path.exists():
            raise FileNotFoundError(
                f"Prediction path {self.pred_path} does not exist. Did you run the predict step?"
            )
        if not self.gt_path.exists():
            raise FileNotFoundError(
                f"Ground truth path {self.gt_path} does not exist. Did you run the preprocess step?"
            )

    def train(self) -> Optional[dict]:
        """Train the YOLO model using the provided configuration.
        If `use_kfold` is set to True, performs k-fold cross-validation.
        If `use_kfold` is False, performs a standard single-run training.
        """
        return self._run_kfold() if self.use_kfold else self._run_single_training()
        
    def _run_kfold(self) -> Dict:
        """
        Prefer materialized folds under <src_path>/cv_folds/fold_*/... .
        If not found, fallback to dynamic GroupKFold over <src_path>/images/train.
        All outputs (weights, predictions, metrics) are written under self.dst_path.
        """
        cv_root = self.src_path / "cv_folds"
        fold_metrics: list[SegmentationMetrics] = []

        if cv_root.exists():
            logger.info(f"[YOLO] Using materialized folds at: {cv_root}")
            fold_dirs = sorted(
                [p for p in cv_root.iterdir() if p.is_dir() and p.name.startswith("fold_")],
                key=lambda p: int(p.name.split("_")[-1])
            )
            if not fold_dirs:
                logger.warning(f"No folds found inside {cv_root}. Falling back to dynamic GroupKFold.")
                return self._run_kfold_dynamic()

            for i, fold_src in enumerate(fold_dirs, start=1):
                out_fold_dir = self.dst_path / fold_src.name
                out_fold_dir.mkdir(parents=True, exist_ok=True)

                # data.yaml points to SOURCE fold (data lives there)
                data_yaml = {
                    "path": str(fold_src),
                    "train": "images/train",
                    "val": "images/val",
                    "nc": 1,
                    "names": ["lesion"],
                    "task": "segment",
                }
                fold_yaml = out_fold_dir / "data.yaml"
                with open(fold_yaml, "w") as f:
                    yaml.dump(data_yaml, f, default_flow_style=False)

                # derive size from source fold train image
                train_imgs = sorted((fold_src / "images" / "train").iterdir())
                if not train_imgs:
                    raise RuntimeError(f"No training images in {fold_src / 'images' / 'train'}")
                img = cv2.imread(str(train_imgs[0]), cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError(f"Could not load image {train_imgs[0]}")
                h, w = img.shape[:2]

                # train -> outputs under out_fold_dir
                model = uYOLO(self.model_path)
                model.train(
                    data=fold_yaml,
                    epochs=self.epochs,
                    batch=self.batch_size,
                    save=True,
                    imgsz=(h, w),
                    project=out_fold_dir,
                    device="cuda",
                    verbose=True,
                    patience=10,
                )

                # predict + evaluate: read from SOURCE fold, write under OUT
                pred_dir = out_fold_dir / "predictions"
                best_weights = out_fold_dir / "train" / "weights" / "best.pt"

                eval_cfg = EvaluateConfig(
                    net=Net.YOLO,
                    src_path=fold_src,                 # data source
                    pred_path=pred_dir,                # outputs here
                    model_path=best_weights,
                    gt_path=fold_src / "labels" / "val",
                )
                evaluator = YOLO(eval_cfg)
                evaluator.predict(
                    image_dir=fold_src / "images" / "val",
                    pred_dir=pred_dir,
                )
                metrics = evaluator.evaluate()
                fold_metrics.append(metrics)

                send_whatsapp_message(
                    f"YOLO Fold {i}/{len(fold_dirs)} completed. "
                    f"Metrics: {metrics.model_dump_json()}"
                )

            summary = self.metrics_calculator.kfold_summary(fold_metrics)
            self.metrics_calculator.write_kfold_summary(summary, self.dst_path / "kfold_summary.json")
            send_whatsapp_message(f"YOLO Cross-validation completed. Summary: {json.dumps(summary, indent=4)}")
            return summary

        # ---------- Fallback dynamic K-Fold ----------
        logger.info("[YOLO] Materialized folds not found. Falling back to dynamic GroupKFold.")
        return self._run_kfold_dynamic()

    def _run_kfold_dynamic(self) -> Dict:
        """
        Build GroupKFold over <src_path>/images/train grouped by patient,
        materialize temporary fold data under <dst_path>/_dyn_folds/,
        and write all training artifacts under <dst_path>/fold_i/.
        """
        fold_metrics: list[SegmentationMetrics] = []

        img_dir = self.src_path / "images" / "train"
        mask_dir = self.src_path / "labels" / "train"
        if not img_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(f"Missing train split at {img_dir} or {mask_dir}")

        # collect samples (image, yolo_txt, gt_mask_png, group_id)
        samples = []
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            stem = img_path.stem
            txt = mask_dir / f"{stem}.txt"
            gt  = mask_dir / f"{stem}.png"
            if not txt.exists():
                raise FileNotFoundError(f"Missing YOLO txt for {img_path}")
            if not gt.exists():
                raise FileNotFoundError(f"Missing GT mask PNG for {img_path}")
            group = stem.split("_")[0]
            samples.append((img_path, txt, gt, group))

        if not samples:
            raise RuntimeError("No training samples to build dynamic CV folds.")

        all_imgs, all_txts, all_gts, groups = zip(*samples)

        gkf = GroupKFold(n_splits=self.kfold_n_splits)
        dyn_root = self.dst_path / "_dyn_folds"
        dyn_root.mkdir(parents=True, exist_ok=True)

        for i, (tr_idx, va_idx) in enumerate(gkf.split(all_imgs, groups=groups), start=1):
            # materialize fold data under dst_path/_dyn_folds/fold_i
            fold_src = dyn_root / f"fold_{i}"
            for sub in ("images/train", "images/val", "labels/train", "labels/val"):
                (fold_src / sub).mkdir(parents=True, exist_ok=True)

            # copy files
            for idx in tr_idx:
                shutil.copy2(all_imgs[idx], fold_src / "images" / "train" / all_imgs[idx].name)
                shutil.copy2(all_txts[idx], fold_src / "labels" / "train" / all_txts[idx].name)
                shutil.copy2(all_gts[idx],  fold_src / "labels" / "train" / all_gts[idx].name)
            for idx in va_idx:
                shutil.copy2(all_imgs[idx], fold_src / "images" / "val" / all_imgs[idx].name)
                shutil.copy2(all_txts[idx], fold_src / "labels" / "val" / all_txts[idx].name)
                shutil.copy2(all_gts[idx],  fold_src / "labels" / "val" / all_gts[idx].name)

            # outputs under dst_path/fold_i
            out_fold_dir = self.dst_path / f"fold_{i}"
            out_fold_dir.mkdir(parents=True, exist_ok=True)

            # data.yaml points to dynamic SOURCE fold we just materialized
            data_yaml = {
                "path": str(fold_src),
                "train": "images/train",
                "val": "images/val",
                "nc": 1,
                "names": ["lesion"],
                "task": "segment",
            }
            fold_yaml = out_fold_dir / "data.yaml"
            with open(fold_yaml, "w") as f:
                yaml.dump(data_yaml, f, default_flow_style=False)

            # image size
            tr_imgs = sorted((fold_src / "images" / "train").iterdir())
            if not tr_imgs:
                raise RuntimeError(f"No training images in {fold_src / 'images' / 'train'}")
            img = cv2.imread(str(tr_imgs[0]), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Could not load image {tr_imgs[0]}")
            h, w = img.shape[:2]

            # train
            model = uYOLO(self.model_path)
            model.train(
                data=fold_yaml,
                epochs=self.epochs,
                batch=self.batch_size,
                save=True,
                imgsz=(h, w),
                project=out_fold_dir,
                device="cuda",
                verbose=True,
                patience=10,
            )

            # predict + evaluate
            pred_dir = out_fold_dir / "predictions"
            best_weights = out_fold_dir / "train" / "weights" / "best.pt"

            eval_cfg = EvaluateConfig(
                net=Net.YOLO,
                src_path=fold_src,                # dynamic data we built
                pred_path=pred_dir,               # outputs here
                model_path=best_weights,
                gt_path=fold_src / "labels" / "val",
            )
            evaluator = YOLO(eval_cfg)
            evaluator.predict(
                image_dir=fold_src / "images" / "val",
                pred_dir=pred_dir,
            )
            metrics = evaluator.evaluate()
            fold_metrics.append(metrics)

            send_whatsapp_message(
                f"YOLO Fold {i}/{self.kfold_n_splits} completed. "
                f"Metrics: {metrics.model_dump_json()}"
            )

        summary = self.metrics_calculator.kfold_summary(fold_metrics)
        self.metrics_calculator.write_kfold_summary(summary, self.dst_path / "cv_summary.json")
        send_whatsapp_message(f"YOLO Cross-validation completed. Summary: {json.dumps(summary, indent=4)}")
        return summary

    def _run_single_training(self) -> None:
        """
        Single training run. If 'final_retrain/{train,val}' exists, use it;
        otherwise fall back to root images/{train,val}.
        """
        fr_root = self.src_path / "final_retrain"
        if (fr_root / "images" / "train").exists() and (fr_root / "images" / "val").exists():
            data_yaml = {
                "path": str(fr_root),
                "train": "images/train",
                "val": "images/val",
                "nc": 1,
                "names": ["lesion"],
                "task": "segment",
            }
            fold_yaml = self.dst_path / "data.yaml"
            with open(fold_yaml, "w") as f:
                yaml.dump(data_yaml, f, default_flow_style=False)
            yaml_to_use = fold_yaml
            # derive size from fr train
            train_imgs = sorted((fr_root / "images" / "train").iterdir())
            if not train_imgs:
                raise FileNotFoundError(f"No images found in {fr_root / 'images' / 'train'}")
            sample = str(train_imgs[0])
            img = cv2.imread(sample, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Could not load image {sample}")
            h, w = img.shape[:2]
            imgsz = (h, w)
        else:
            # fallback to root images/{train,val}
            self._create_yaml()
            yaml_to_use = self.yaml_path
            h, w = self._get_image_size()
            imgsz = (h, w)

        model = uYOLO(self.model_path)
        model.train(
            data=yaml_to_use,
            epochs=self.epochs,
            batch=self.batch_size,
            save=True,
            imgsz=imgsz,
            project=self.dst_path,
            device="cuda",
            verbose=True,
            patience=10,
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

        if pred_dir.exists():
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
        inference_times: Dict[str, float] = {}

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
                conf=0.5,
            )
            elapsed = time.perf_counter() - start
            inference_times[img_file.name] = elapsed

        with open(pred_dir / "inference_times.json", "w") as f:
            json.dump(inference_times, f, indent=4)

        logger.success(f"Inference times saved to {pred_dir / 'inference_times.json'}")

        self._draw_predictions(pred_dir=pred_dir, image_dir=image_dir)

    def evaluate(self) -> SegmentationMetrics:
        """
        Evaluate the YOLO predictions using the ground truth masks on a per-image basis.
        Computes metrics using the injected MetricsCalculator.
        """
        logger.info(f"Evaluating YOLO predictions in folder {self.pred_path}")

        # Load inference times
        times_path = self.pred_path / "inference_times.json"
        with open(times_path, 'r') as f:
            times_dict = json.load(f)

        # Collect predicted and ground truth masks
        mask_dir = self.pred_path / "masks"
        image_names = sorted(mask_dir.iterdir(), key=lambda x: x.name)

        preds = []
        gts = []
        times = []


        for name in image_names:
            pred_mask = cv2.imread(str(mask_dir / name.name), cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.imread(str(self.gt_path / name.name), cv2.IMREAD_GRAYSCALE)

            preds.append(pred_mask)
            gts.append(gt_mask)
            times.append(times_dict.get(name.name, 0.0))

        # Stack into (N, H, W) arrays
        preds_np = np.stack(preds)
        gts_np = np.stack(gts)

        # Compute metrics
        avg_metrics, per_image = self.metrics_calculator.evaluate_batch(
            preds=preds_np,
            gts=gts_np,
            times=times,
            threshold=127  # 8-bit (0-255)
        )

        # Add lesion area in pixels to per_image metrics
        areas = (gts_np > 0.5).sum(axis=(1,2))
        for i, a in enumerate(areas.tolist()):
            per_image[i]["lesion_area_px"] = float(a)

        # Save to JSON
        self.metrics = SegmentationMetrics(
            **{k: float(v) for k, v in avg_metrics.items()}
        )
        metrics_path = self.pred_path / "metrics.json"
        self.metrics.write_to_file(metrics_path)
        (self.pred_path / "per_image_metrics.json").write_text(json.dumps(per_image, indent=2))

        logger.info(f"Metrics saved at {self.pred_path / 'metrics.json'}")
        logger.info(f"Per-image metrics saved at {self.pred_path / 'per_image_metrics.json'}")
        logger.info(f"Average metrics:\n{self.metrics}")

        return self.metrics

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
        """
        Draw predictions on images and save them as binary masks.
        """
        output_dir = pred_dir / "masks"
        output_dir.mkdir(parents=True, exist_ok=True)

        prediction_dir = pred_dir / "labels"
        if not prediction_dir.exists():
            raise FileNotFoundError(
                f"Prediction directory {prediction_dir} does not exist. Did you run the predict step?"
            )

        image_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
        img_h, img_w = self._get_image_size()

        for image_file in image_files:
            base_name = image_file.stem
            prediction_path = prediction_dir / f"{base_name}.txt"

            img = cv2.imread(str(image_dir / image_file.name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Could not load image {image_file}")
                continue

            # resize target shape for mask
            mask = np.zeros((img_h, img_w), dtype=np.uint8)

            if not prediction_path.exists():
                logger.warning(f"Prediction file not found for {image_file}. Saving empty mask.")
            else:
                with open(prediction_path, "r") as f:
                    lines = f.readlines()

                if not lines:
                    logger.info(f"No predictions for {image_file}. Saving empty mask.")
                else:
                    for line in lines:
                        values = list(map(float, line.split()))
                        if len(values) > 2:
                            coords = values[1:]
                            if len(coords) % 2 != 0:
                                coords = coords[:-1]
                            if len(coords) >= 4:
                                pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
                                pts[:, 0] *= img_w
                                pts[:, 1] *= img_h
                                pts = pts.astype(np.int32)
                                cv2.fillPoly(mask, [pts], color=255)

            cv2.imwrite(str(output_dir / f"{base_name}.png"), mask)

        logger.success(f"Predictions drawn and saved correctly to {output_dir}")

    def _get_image_size(self) -> tuple[int, int]:
        """
        Return (H, W) from a sample training image.
        Prefer final_retrain/images/train if present; else root images/train.
        """
        base = self.src_path
        fr_train = base / "final_retrain" / "images" / "train"
        train_path = fr_train if fr_train.exists() else base / "images" / "train"
        image_files = [p for p in train_path.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        if not image_files:
            raise FileNotFoundError(f"No images found in {train_path}")
        img = cv2.imread(str(image_files[0]), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image {image_files[0]}")
        h, w = img.shape[:2]
        return (h, w)
