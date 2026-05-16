import os
import tarfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from config import (
    DATASET_REPO, TRAIN_ARCHIVE, VAL_ARCHIVE,
    DATA_DIR, NUM_TEST_CHIPS,
)
from data_processing import (
    build_hls_bands, normalize_hls, build_sar_proxy,
    compute_spectral_indices, compute_pseudo_gt,
)


def _download_and_extract(archive: str, data_dir: str = DATA_DIR) -> Path:
    from huggingface_hub import hf_hub_download
    os.makedirs(data_dir, exist_ok=True)
    tgz         = hf_hub_download(repo_id=DATASET_REPO, filename=archive, repo_type="dataset")
    extract_dir = Path(data_dir) / archive.replace(".tgz", "_extracted")
    if not extract_dir.exists():
        print(f"  Extracting {archive} → {extract_dir} …")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tgz, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        print("  Done.")
    else:
        print(f"  Cached: {extract_dir}")
    return extract_dir


def load_chip_paths(extract_dir: Path) -> List[Tuple[Path, Optional[Path]]]:
    merged = sorted(
        p for p in extract_dir.rglob("*_merged.tif")
        if not p.name.startswith("._")
    )
    pairs = []
    for mp in merged:
        mask_p = Path(str(mp).replace("_merged.tif", "_mask.tif"))
        valid  = mask_p.exists() and not mask_p.name.startswith("._")
        pairs.append((mp, mask_p if valid else None))
    return pairs


def download_hls_chips(
    n:        int  = NUM_TEST_CHIPS,
    data_dir: str  = DATA_DIR,
    archive:  str  = VAL_ARCHIVE,
) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]], List[str]]:
    import rasterio
    extract_dir = _download_and_extract(archive, data_dir)
    pairs       = load_chip_paths(extract_dir)
    chips, masks, names = [], [], []
    for i, (mp, mask_p) in enumerate(pairs[:n]):
        with rasterio.open(str(mp)) as src:
            data = src.read().astype(np.float32)
        chip = data[:6]
        chips.append(chip)
        if mask_p is not None:
            with rasterio.open(str(mask_p)) as src:
                masks.append(src.read(1).astype(np.int32))
        else:
            masks.append(None)
        names.append(mp.name)
        print(f"  ✓  [{i+1}/{n}]  {mp.name}  shape={chip.shape}")
    return chips, masks, names


class ChipDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Optional[Path]]], augment: bool = True):
        import rasterio as _rio
        self.pairs   = pairs
        self.augment = augment
        self._rio    = _rio

    def __len__(self) -> int:
        return len(self.pairs)

    def _augment(self, *arrays):
        if np.random.rand() > 0.5:
            arrays = tuple(np.flip(a, axis=-1).copy() for a in arrays)
        if np.random.rand() > 0.5:
            arrays = tuple(np.flip(a, axis=-2).copy() for a in arrays)
        k = np.random.randint(0, 4)
        if k:
            arrays = tuple(np.rot90(a, k=k, axes=(-2, -1)).copy() for a in arrays)
        return arrays

    def __getitem__(self, idx: int):
        mp, _ = self.pairs[idx]
        with self._rio.open(str(mp)) as src:
            data = src.read().astype(np.float32)
        chip     = data[:6]
        hls      = build_hls_bands(chip)
        sar      = build_sar_proxy((hls.shape[1], hls.shape[2]))
        indices  = compute_spectral_indices(hls)
        hls_norm = normalize_hls(hls)
        gt       = compute_pseudo_gt(chip)
        gt_stack = np.stack([gt["chl"], gt["nit"], gt["bio"], gt["loss"]])
        if self.augment:
            hls_norm, sar, indices, gt_stack = self._augment(hls_norm, sar, indices, gt_stack)
        return (
            torch.from_numpy(hls_norm).float(),
            torch.from_numpy(sar).float(),
            torch.from_numpy(indices).float(),
            torch.from_numpy(gt_stack).float(),
        )


def build_dataloaders(data_dir: str = DATA_DIR, batch_size: int = 8, num_workers: int = 2):
    from torch.utils.data import DataLoader
    train_dir   = _download_and_extract(TRAIN_ARCHIVE, data_dir)
    val_dir     = _download_and_extract(VAL_ARCHIVE,   data_dir)
    train_pairs = load_chip_paths(train_dir)
    val_pairs   = load_chip_paths(val_dir)
    print(f"  Train chips : {len(train_pairs)}")
    print(f"  Val chips   : {len(val_pairs)}")
    train_ds = ChipDataset(train_pairs, augment=True)
    val_ds   = ChipDataset(val_pairs,   augment=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl
