import os
import sys
from pathlib import Path
from typing import Callable, Optional

import nibabel as nib
import numpy as np
from loguru import logger
from tqdm import tqdm

from interfaces.interfaces import (
    ImageSaver,
    PatientSplitter,
    SliceSelector,
    Transformer,
)
from schemas.pipeline_schemas import (
    Net,
    PreprocessConfig,
    Strategy,
    KFoldPrep,
    SuperScale,
)
from steps.preprocessing.savers.savers import PNGSaver, YOLOSegSaver
from steps.preprocessing.slicers.selectors import (
    AllSlicesSelector,
    BrainOnlySelector,
    LesionOnlySelector,
    TopCenteredBlockSelector,
)
from steps.preprocessing.splitters.holdout_splitter import HoldoutPatientSplitter, RandomPatientSplitter
from steps.preprocessing.stats.slice_stats import StatsCollector, compute_slice_stats
from steps.preprocessing.transformers.transformers import (
    ComposeTransformer,
    FlairNormalizeTransformer,
    MaskBinarizeTransformer,
    ResizeTransformer,
    SuperResolutionTransformer,
)
from steps.preprocessing.utils.folds import build_kfolds, split_minival
from steps.preprocessing.utils.linking import hardlink_or_copy


class PreprocessingPipeline:
    def __init__(
        self,
        src_train_root: Path,
        dst_root: Path,
        splitter: PatientSplitter,
        selector: SliceSelector,
        transformer: Transformer,
        saver: ImageSaver,
        kfold: Optional[KFoldPrep] = None,
        compute_stats: Optional[Callable] = None,
        super_scale: Optional[SuperScale] = None
    ):
        self.src_train_root = src_train_root
        self.dst_root = dst_root
        self.splitter = splitter
        self.selector = selector
        self.transformer = transformer
        self.saver = saver
        self.kfold = kfold
        self.super_scale = super_scale

        self.stats = StatsCollector()
        self.compute_stats = compute_stats
        self.strategy_name = type(self.selector).__name__
        self.preproc_signature = "resize{}_sr{}".format(
            getattr(self.transformer, "target_size", None),
            getattr(self.transformer, "super_scale", None)
        )

    def run(self):
        """
        Executes the preprocessing pipeline for the dataset.

        This method performs the following steps:
        1. Retrieves and sorts all patient directories from the source training root.
        2. Splits the patient IDs into training and testing sets using the configured splitter.
        3. Creates directories for the training and testing splits.
        4. Processes the training and testing splits by generating the necessary images and labels.
        5. If k-fold cross-validation is enabled, materializes mini-validation sets and folds based on the configuration.
        6. Computes and writes statistical summaries if the `compute_stats` flag is set.

        Steps involving k-fold cross-validation and statistics computation are optional and depend on the configuration.

        Attributes:
            self.src_train_root (Path): The root directory containing the source training data.
            self.splitter (Splitter): The object responsible for splitting the dataset into training, validation, and testing sets.
            self.kfold (KFoldConfig): Configuration for k-fold cross-validation.
            self.compute_stats (bool): Flag indicating whether to compute and save dataset statistics.
            self.dst_root (Path): The root directory where processed data and statistics will be saved.
            self.stats (Stats): Object responsible for computing and saving dataset statistics.

        Raises:
            ValueError: If any required configuration or directory is missing.
        """
        all_patients = sorted([d.name for d in self.src_train_root.iterdir() if d.is_dir()])
        train_ids, _, test_ids = self.splitter.split(all_patients)  # discard val_ids

        split_dirs = self._make_split_dirs(["train", "test"])

        for split_name, pat_ids in [("train", train_ids), ("test", test_ids)]:
            img_dir, lbl_dir = split_dirs[split_name]
            self._process_split(split_name, pat_ids, img_dir, lbl_dir)

        kcfg = self.kfold
        if kcfg and kcfg.enable:
            self._materialize_minival_and_folds(
                final_retrain=kcfg.final_retrain,
                mini_val_frac=kcfg.mini_val_frac,
                n_splits=kcfg.n_splits,
                seed=kcfg.seed,
                link_mode=kcfg.link_mode,
            )

        if self.compute_stats:
            stats_dir = self.dst_root / "stats"
            self.stats.write_json(stats_dir / "per_slice.json")
            self.stats.write_summary_json(stats_dir / "summary.json")

    def _make_split_dirs(self, split_names: list[str]) -> dict[str, tuple[Path, Path]]:
        """
        Creates directories for image and label splits based on the provided split names.

        This method generates subdirectories for each split name under the `images` and `labels`
        directories within the destination root (`dst_root`). If the directories already exist,
        they will not be recreated.

        Args:
            split_names (list[str]): A list of split names (e.g., "train", "val", "test") for which
                                     directories will be created.

        Returns:
            dict[str, tuple[Path, Path]]: A dictionary where each key is a split name, and the value
                                          is a tuple containing the paths to the corresponding image
                                          and label directories.
        """
        splits: dict[str, tuple[Path, Path]] = {}
        for name in split_names:
            img_dir = self.dst_root / "images" / name
            lbl_dir = self.dst_root / "labels" / name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            splits[name] = (img_dir, lbl_dir)
        return splits

    def _materialize_minival_and_folds(self, final_retrain: bool, mini_val_frac: float, n_splits: int, seed: int, link_mode: str):
        """
        Materializes mini-validation sets and K-Folds for training and evaluation.
        This method creates the necessary directory structure and links or copies 
        image and label files for mini-validation and K-Fold cross-validation splits. 
        It also generates manifest files to document the splits.
        Args:
            final_retrain (bool): If True, creates a mini-validation set for final retraining.
            mini_val_frac (float): Fraction of the training data to use for the mini-validation set.
            n_splits (int): Number of splits for K-Fold cross-validation.
            seed (int): Random seed for reproducibility in data splitting.
            link_mode (str): Mode for linking or copying files. Options include "hardlink" or "copy".
        Directory Structure:
            - For mini-validation (if `final_retrain` is True):
                dst_root/final_retrain/
                    ├── images/
                    │   ├── train/
                    │   └── val/
                    └── labels/
                        ├── train/
                        └── val/
            - For K-Folds:
                dst_root/cv_folds/
                    ├── fold_1/
                    │   ├── images/
                    │   │   ├── train/
                    │   │   └── val/
                    │   └── labels/
                    │       ├── train/
                    │       └── val/
                    ├── fold_2/
                    └── ...
        Manifest Files:
            - `minival_manifest.json`: Contains metadata about the mini-validation split, 
              including the fraction, seed, and the list of training and validation images.
            - `folds_manifest.json`: Contains metadata about the K-Fold splits, including 
              the number of splits and the list of training and validation images for each fold.
        Dependencies:
            - `split_minival`: Function to split the training data into mini-validation sets.
            - `build_kfolds`: Function to generate K-Fold splits.
            - `hardlink_or_copy`: Utility to link or copy files based on the specified mode.
        Raises:
            ValueError: If invalid arguments are provided for `mini_val_frac`, `n_splits`, or `link_mode`.
        """
        train_img = self.dst_root / "images" / "train"
        train_lbl = self.dst_root / "labels" / "train"

        # ---- Mini-val for final retrain
        if final_retrain:
            fr_root = self.dst_root / "final_retrain"
            for sub in ("images/train", "images/val", "labels/train", "labels/val"):
                (fr_root / sub).mkdir(parents=True, exist_ok=True)
            
            tr_imgs, va_imgs = split_minival(train_img, frac=mini_val_frac, seed=seed)

        def link_pair(img_path: Path, split: str, base_root: Path):
            stem = img_path.stem
            yolo_txt = train_lbl / f"{stem}.txt"    # for YOLO (may not exist for UNet)
            gt_png   = train_lbl / f"{stem}.png"    # for UNet/YOLO evaluator
            hardlink_or_copy(img_path, base_root / "images" / split / img_path.name, mode=link_mode)
            if yolo_txt.exists():
                hardlink_or_copy(yolo_txt, base_root / "labels" / split / yolo_txt.name, mode=link_mode)
            if gt_png.exists():
                hardlink_or_copy(gt_png, base_root / "labels" / split / gt_png.name, mode=link_mode)
        
        if final_retrain:
            for p in tr_imgs:
                link_pair(p, "train", fr_root)
            for p in va_imgs:
                link_pair(p, "val", fr_root)
            
            # Write manifests
            import json
            (self.dst_root / "minival_manifest.json").write_text(json.dumps({
                "mini_val_frac": mini_val_frac,
                "seed": seed,
                "train_images": [p.name for p in tr_imgs],
                "val_images":   [p.name for p in va_imgs],
            }, indent=2))

        # ---- K-Folds (cv_folds/fold_i)
        cv_root = self.dst_root / "cv_folds"
        folds = build_kfolds(train_img, n_splits=n_splits)
        for i, (tr, va) in enumerate(folds, start=1):
            fold_dir = cv_root / f"fold_{i}"
            for sub in ("images/train", "images/val", "labels/train", "labels/val"):
                (fold_dir / sub).mkdir(parents=True, exist_ok=True)
            for p in tr:
                link_pair(p, "train", fold_dir)
            for p in va:
                link_pair(p, "val", fold_dir)

        (self.dst_root / "folds_manifest.json").write_text(json.dumps({
            "n_splits": n_splits,
            "folds": [
                {
                    "train_images": [p.name for p in tr],
                    "val_images": [p.name for p in va],
                } for tr, va in folds
            ]
        }, indent=2))

    def _process_split(
        self,
        split_name: str,
        patient_set: set[str],
        images_dir: Path,
        labels_dir: Path,
    ):
        """
        Processes a specific data split by iterating through patient folders, extracting 
        relevant slices from medical imaging data, and applying transformations and saving 
        the results.

        Args:
            split_name (str): The name of the data split (e.g., "train", "validation", "test").
            patient_set (set[str]): A set of patient IDs to process.
            images_dir (Path): The directory where processed image slices will be saved.
            labels_dir (Path): The directory where processed label slices will be saved.

        Workflow:
            1. Iterates through each patient in the `patient_set`.
            2. Checks if the patient's folder exists; logs a warning if not.
            3. For each timepoint in the patient's folder:
                - Loads the FLAIR and MASK NIfTI files.
                - Selects relevant slices using the `selector` object.
                - Applies transformations to the slices using the `transformer` object.
                - Saves the transformed slices using the `saver` object.
                - Computes and stores statistics if `compute_stats` is enabled.

        Notes:
            - Skips processing if required files (FLAIR or MASK) are missing.
            - Ensures slice indices are within valid bounds.
            - Saves empty masks if the transformed mask has no positive values.

        Raises:
            Any exceptions raised by file I/O, NIfTI loading, or transformations 
            will propagate to the caller.
        """
        for pat_id in tqdm(patient_set, desc=f"Split {split_name}"):
            pat_folder = self.src_train_root / pat_id
            if not pat_folder.exists():
                logger.warning(f"Patient folder {pat_folder} not found.")
                continue

            for timepoint in os.listdir(pat_folder):
                tp_path = pat_folder / timepoint
                flair_nifti = tp_path / f"{pat_id}_{timepoint}_FLAIR.nii.gz"
                mask_nifti  = tp_path / f"{pat_id}_{timepoint}_MASK.nii.gz"
                if not flair_nifti.exists() or not mask_nifti.exists():
                    continue

                flair_vol = nib.load(str(flair_nifti)).get_fdata()
                mask_vol  = nib.load(str(mask_nifti)).get_fdata()

                slice_idxs = self.selector.select_slices(flair_vol, mask_vol)
                for idx in slice_idxs:
                    if idx < 0 or idx >= flair_vol.shape[2]:
                        continue

                    flair_slice = flair_vol[:, :, idx]
                    mask_slice  = mask_vol[:, :, idx]

                    base_name = f"{pat_id}_{timepoint}_{idx}.png"
                    out_flair = images_dir / base_name
                    out_mask  = labels_dir / base_name

                    t_flair = self.transformer.transform(flair_slice, is_flair=True)
                    t_mask  = self.transformer.transform(mask_slice, is_flair=False)

                    self.saver.save_flair(t_flair, out_flair)

                    if np.max(t_mask) > 0:
                        self.saver.save_mask(t_mask, out_mask)
                    else:
                        self.saver.save_mask(np.zeros_like(t_mask), out_mask)

                    if self.compute_stats:
                        m_bin = (t_mask > 0).astype(np.uint8) * 255
                        s = self.compute_stats(
                            t_flair.astype(np.uint8), m_bin, pat_id, timepoint, idx, split_name, self.strategy_name, self.preproc_signature
                        )
                        self.stats.add(s.model_dump())

# --------------------- Entry Point ---------------------
def preprocess(config: PreprocessConfig) -> None:
    """
    Preprocesses medical imaging data based on the provided configuration.

    This function orchestrates the preprocessing pipeline, which includes:
    - Validating the source path.
    - Splitting the dataset into training and testing sets using either random or holdout strategies.
    - Selecting slices based on the specified strategy (e.g., all slices, lesion-only, brain-only).
    - Applying a series of transformations such as resizing, normalization, and super-resolution.
    - Saving the preprocessed data in the format required by the specified neural network (e.g., UNet, YOLO).

    Args:
        config (PreprocessConfig): Configuration object containing all parameters 
            required for preprocessing, including paths, strategies, and transformations.

    Raises:
        ValueError: If an invalid neural network type is specified in the configuration.

    Side Effects:
        - Writes preprocessed data to the destination path specified in the configuration.
        - Logs errors and exits if the source path does not exist.
        - Writes the configuration to a file after preprocessing.

    Note:
        - The function supports k-fold cross-validation if enabled in the configuration.
        - The preprocessing pipeline computes statistics for each slice if required.
    """
    if not os.path.exists(config.src_path):
        logger.error(f"Source path {config.src_path} does not exist. Exiting.")
        sys.exit(1)

    if not config.kfold or not config.kfold.enable:
        splitter = RandomPatientSplitter(train_fac=config.split, seed=config.seed)
    else:
        splitter = HoldoutPatientSplitter(test_frac=1.0 - config.split, seed=config.seed)

    match config.strategy:
        case Strategy.ALL_SLICES:
            selector = AllSlicesSelector()
        case Strategy.LESION:
            selector = LesionOnlySelector(min_area=config.threshold or 0)
        case Strategy.LESION_BLOCK:
            selector = TopCenteredBlockSelector(block_size=config.block_size, min_area=config.threshold)
        case Strategy.BRAIN:
            selector = BrainOnlySelector()

    transformers = [
        ResizeTransformer(target_size=config.resize, method=config.resize_method, super_scale=config.super_scale),
        SuperResolutionTransformer(
            super_scale=config.super_scale,      # if NONE, no-op
            orig_size=config.resize,
        ),
        FlairNormalizeTransformer(),
        MaskBinarizeTransformer(),
    ]

    transformer = ComposeTransformer(transformers)

    if config.net == Net.UNET:
        saver: ImageSaver = PNGSaver()
    elif config.net == Net.YOLO:
        saver = YOLOSegSaver(class_id=0)
    else:
        raise ValueError(f"Invalid net {config.net}")

    pipeline = PreprocessingPipeline(
        src_train_root=Path(config.src_path) / "train",
        dst_root=Path(config.dst_path),
        splitter=splitter,
        selector=selector,
        transformer=transformer,
        saver=saver,
        kfold=config.kfold,
        compute_stats=compute_slice_stats,
        super_scale=config.super_scale,
    )
    pipeline.run()
    config.write_config()
