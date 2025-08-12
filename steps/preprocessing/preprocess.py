import os
from pathlib import Path
import sys

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
    SuperScale,
    KFoldPrep,
)
from steps.preprocessing.savers.savers import PNGSaver, YOLOSegSaver
from steps.preprocessing.slicers.selectors import (
    AllSlicesSelector,
    LesionOnlySelector,
    TopCenteredBlockSelector,
)
from steps.preprocessing.splitters.holdout_splitter import HoldoutPatientSplitter
# from steps.preprocessing.splitters.random_splitter import RandomPatientSplitter
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
        kfold: KFoldPrep | None = None,
    ):
        self.src_train_root = src_train_root
        self.dst_root = dst_root
        self.splitter = splitter
        self.selector = selector
        self.transformer = transformer
        self.saver = saver
        self.kfold = kfold

    def run(self):
        all_patients = sorted([d.name for d in self.src_train_root.iterdir() if d.is_dir()])
        train_ids, _, test_ids = self.splitter.split(all_patients)  # discard val_ids

        split_dirs = self._make_split_dirs(["train", "test"])

        for split_name, pat_ids in [("train", train_ids), ("test", test_ids)]:
            img_dir, lbl_dir = split_dirs[split_name]
            self._process_split(split_name, pat_ids, img_dir, lbl_dir)

        kcfg = self.kfold
        if kcfg and kcfg.enable:
            self._materialize_minival_and_folds(
                mini_val_frac=kcfg.mini_val_frac,
                n_splits=kcfg.n_splits,
                seed=kcfg.seed,
                link_mode=kcfg.link_mode,
            )

    def _make_split_dirs(self, split_names: list[str]) -> dict[str, tuple[Path, Path]]:
        splits: dict[str, tuple[Path, Path]] = {}
        for name in split_names:
            img_dir = self.dst_root / "images" / name
            lbl_dir = self.dst_root / "labels" / name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            splits[name] = (img_dir, lbl_dir)
        return splits

    def _materialize_minival_and_folds(self, mini_val_frac: float, n_splits: int, seed: int, link_mode: str):
        train_img = self.dst_root / "images" / "train"
        train_lbl = self.dst_root / "labels" / "train"

        # ---- Mini-val for final retrain
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
        for pat_id in tqdm(patient_set, desc=f"Split {split_name}"):
            pat_folder = self.src_train_root / pat_id
            if not pat_folder.exists():
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
                    out_flair = str(images_dir / base_name)
                    out_mask  = str(labels_dir / base_name)

                    t_flair = self.transformer.transform(flair_slice, is_flair=True)
                    t_mask  = self.transformer.transform(mask_slice, is_flair=False)

                    self.saver.save_flair(t_flair, out_flair)
                    if np.max(t_mask) > 0:
                        self.saver.save_mask(t_mask, out_mask)
                    else:
                        self.saver.save_mask(np.zeros_like(t_mask), out_mask)

# --------------------- Entry Point ---------------------
def preprocess(config: PreprocessConfig) -> None:
    if not os.path.exists(config.src_path):
        logger.error(f"Source path {config.src_path} does not exist. Exiting.")
        sys.exit(1)

    splitter = HoldoutPatientSplitter(test_frac=1.0 - config.split, seed=config.seed)

    match config.strategy:
        case Strategy.ALL_SLICES:
            selector = AllSlicesSelector()
        case Strategy.LESION:
            selector = LesionOnlySelector(min_area=config.threshold)
        case Strategy.LESION_BLOCK:
            selector = TopCenteredBlockSelector(block_size=config.block_size, min_area=config.threshold)
        case _:
            raise ValueError(f"Invalid strategy {config.strategy}")

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
    )
    pipeline.run()
    config.write_config()