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
)
from steps.preprocessing.savers.savers import PNGSaver, YOLOSegSaver
from steps.preprocessing.slicers.selectors import (
    AllSlicesSelector,
    LesionOnlySelector,
    TopCenteredBlockSelector,
)
from steps.preprocessing.splitters.random_splitter import RandomPatientSplitter
from steps.preprocessing.transformers.transformers import (
    ComposeTransformer,
    FlairNormalizeTransformer,
    MaskBinarizeTransformer,
    ResizeTransformer,
    SuperResolutionTransformer,
)


class PreprocessingPipeline:
    def __init__(
        self,
        src_train_root: Path,
        dst_root: Path,
        splitter: PatientSplitter,
        selector: SliceSelector,
        transformer: Transformer,
        saver: ImageSaver,
    ):
        self.src_train_root = src_train_root
        self.dst_root = dst_root
        self.splitter = splitter
        self.selector = selector
        self.transformer = transformer
        self.saver = saver

    def run(self):
        all_patients = [d.name for d in self.src_train_root.iterdir() if d.is_dir()]
        train_ids, val_ids, test_ids = self.splitter.split(all_patients)
        split_dirs = self._make_split_dirs()

        for split_name, pat_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            img_dir, lbl_dir = split_dirs[split_name]
            self._process_split(split_name, pat_ids, img_dir, lbl_dir)

    def _make_split_dirs(self) -> dict[str, tuple[Path, Path]]:
        splits: dict[str, tuple[Path, Path]] = {}
        for name in ("train", "val", "test"):
            img_dir = self.dst_root / "images" / name
            lbl_dir = self.dst_root / "labels" / name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            splits[name] = (img_dir, lbl_dir)
        return splits

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

                    # 1) Transform arrays
                    t_flair = self.transformer.transform(flair_slice, is_flair=True)
                    t_mask  = self.transformer.transform(mask_slice, is_flair=False)

                    # 2) Save to disk
                    self.saver.save_flair(t_flair, out_flair)
                    if np.max(t_mask) > 0:
                        self.saver.save_mask(t_mask, out_mask)
                    else:
                        # write empty masks
                        self.saver.save_mask(np.zeros_like(t_mask), out_mask)

# --------------------- Entry Point ---------------------
def preprocess(config: PreprocessConfig) -> None:
    """
    Preprocess images based on the provided configuration.
    Args:
        config: PreprocessConfig object containing all necessary parameters.
    """

    if not os.path.exists(config.src_path):
        logger.error(f"Source path {config.src_path} does not exist. Exiting.")
        sys.exit(1)

    splitter = RandomPatientSplitter(train_frac=config.split, seed=config.seed)

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
        FlairNormalizeTransformer(),
        MaskBinarizeTransformer(),
    ]
    
    if config.super_scale != SuperScale.NONE:
        transformers.insert(
            1, SuperResolutionTransformer(super_scale=config.super_scale)
        )

    transformer = ComposeTransformer(transformers)

    if config.net == Net.UNET:
        saver: ImageSaver = PNGSaver()
    elif config.net == Net.YOLO:
        saver = YOLOSegSaver(class_id=0)
    else:
        raise ValueError(f"Invalid net {config.net}")
    
    logger.info(f"Starting preprocessing with config: {config}")

    if not os.path.exists(config.src_path):
        raise FileNotFoundError(f"Source path {config.src_path} does not exist.")

    pipeline = PreprocessingPipeline(
        src_train_root=Path(config.src_path) / "train",
        dst_root=Path(config.dst_path),
        splitter=splitter,
        selector=selector,
        transformer=transformer,
        saver=saver,
    )
    pipeline.run()
    config.write_config()
