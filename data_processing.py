from typing import Dict, List, Tuple

import cv2
import numpy as np

from config import (
    HLS_MEAN, HLS_STD, TILE_SIZE, TILE_STRIDE,
    MAX_CHL_UG_CM2, MAX_N_PERCENT, MAX_BIOMASS_MGHA,
    VEG_NDVI_THRESH,
)


def build_hls_bands(image: np.ndarray) -> np.ndarray:
    C = image.shape[0]
    if C >= 6:
        return image[:6].copy().astype(np.float32)
    if C == 3:
        R, G, B = image[0], image[1], image[2]
        RE1 = R * 0.55 + G * 0.30 + B * 0.15
        RE2 = R * 0.60 + G * 0.25 + B * 0.15
        RE3 = R * 0.50 + G * 0.30 + B * 0.20
        return np.stack([B, G, R, RE1, RE2, RE3]).astype(np.float32)
    if C == 1:
        return np.tile(image, (6, 1, 1)).astype(np.float32)
    pad = np.zeros((6 - C, image.shape[1], image.shape[2]), dtype=np.float32)
    return np.concatenate([image[:C], pad], axis=0).astype(np.float32)


def normalize_hls(hls: np.ndarray) -> np.ndarray:
    out = hls.copy()
    for i in range(6):
        out[i] = (out[i] - HLS_MEAN[i]) / (HLS_STD[i] + 1e-8)
    return out


def build_sar_proxy(spatial_shape: Tuple[int, int]) -> np.ndarray:
    H, W = spatial_shape
    rng  = np.random.default_rng(seed=0)
    vv   = rng.uniform(-15.0, -5.0,  (H, W)).astype(np.float32)
    vh   = (vv - rng.uniform(5.0, 12.0, (H, W))).astype(np.float32)
    rvi  = (4.0 * 10 ** (vh / 10.0)) / (10 ** (vv / 10.0) + 10 ** (vh / 10.0) + 1e-8)
    coh  = rng.uniform(0.3, 0.9, (H, W)).astype(np.float32)
    return np.stack([vv, vh, rvi.astype(np.float32), coh], axis=0)


def compute_spectral_indices(hls: np.ndarray) -> np.ndarray:
    B02, B03, B04 = hls[0], hls[1], hls[2]
    B05, B06, B07 = hls[3], hls[4], hls[5]
    eps  = 1e-8
    NIR, Red = B07, B04
    ndvi  = (NIR - Red)  / (NIR + Red  + eps)
    ndre  = (NIR - B05)  / (NIR + B05  + eps)
    evi   = 2.5 * (NIR - Red) / (NIR + 6 * Red - 7.5 * B02 + 1 + eps)
    cire  = (NIR / (B05 + eps)) - 1.0
    ndwi  = (B03 - NIR) / (B03 + NIR + eps)
    mndwi = (B03 - B06) / (B03 + B06 + eps)
    savi  = 1.5 * (NIR - Red) / (NIR + Red + 0.5 + eps)
    rvi   = NIR / (Red + eps)
    return np.stack([ndvi, ndre, evi, cire, ndwi, mndwi, savi, rvi]).astype(np.float32)


def tile_image(
    hls_norm: np.ndarray,
    sar:      np.ndarray,
    indices:  np.ndarray,
    tile_size: int = TILE_SIZE,
    stride:    int = TILE_STRIDE,
) -> List[Dict]:
    _, H, W = hls_norm.shape
    tiles   = []
    y = 0
    while True:
        ey = min(y + tile_size, H)
        sy = max(0, ey - tile_size)
        x  = 0
        while True:
            ex = min(x + tile_size, W)
            sx = max(0, ex - tile_size)
            ch, cw = ey - sy, ex - sx
            hls_t = np.zeros((hls_norm.shape[0], tile_size, tile_size), dtype=np.float32)
            sar_t = np.zeros((sar.shape[0],       tile_size, tile_size), dtype=np.float32)
            idx_t = np.zeros((indices.shape[0],   tile_size, tile_size), dtype=np.float32)
            hls_t[:, :ch, :cw] = hls_norm[:, sy:ey, sx:ex]
            sar_t[:, :ch, :cw] = sar[:,     sy:ey, sx:ex]
            idx_t[:, :ch, :cw] = indices[:, sy:ey, sx:ex]
            tiles.append({
                "hls": hls_t, "sar": sar_t, "indices": idx_t,
                "y_start": sy, "x_start": sx, "y_end": ey, "x_end": ex,
                "valid_h": ch, "valid_w": cw,
            })
            if ex == W: break
            x += stride
        if ey == H: break
        y += stride
    return tiles


def compute_gt_proxies(chip: np.ndarray) -> Dict:
    B02, B03, B04 = chip[0], chip[1], chip[2]
    B05, B06, B07 = chip[3], chip[4], chip[5]
    eps  = 1e-8
    ndvi = (B07 - B04) / (B07 + B04 + eps)
    ndre = (B07 - B05) / (B07 + B05 + eps)
    cire = (B07 / (B05 + eps)) - 1.0
    chl_gt   = np.clip(cire / 10.0, 0, 1) * MAX_CHL_UG_CM2
    n_gt_pct = np.clip(ndre * 2.5 + 0.5, 0, MAX_N_PERCENT)
    agb_gt   = (0.7 * np.clip(ndvi, 0, 1) + 0.3 * (B07 / (B07.max() + eps))) * 120.0
    veg_mask = ndvi > VEG_NDVI_THRESH
    return {
        "chl_gt_ug_cm2": chl_gt,
        "n_gt_pct":      n_gt_pct,
        "agb_gt_mgha":   agb_gt,
        "ndvi":          ndvi,
        "ndre":          ndre,
        "cire":          cire,
        "veg_mask":      veg_mask,
        "veg_pct":       100.0 * veg_mask.sum() / veg_mask.size,
    }


def compute_pseudo_gt(chip: np.ndarray) -> Dict:
    B02, B03, B04 = chip[0], chip[1], chip[2]
    B05, B06, B07 = chip[3], chip[4], chip[5]
    eps    = 1e-8
    ndvi   = (B07 - B04) / (B07 + B04 + eps)
    ndre   = (B07 - B05) / (B07 + B05 + eps)
    cire   = (B07 / (B05 + eps)) - 1.0
    ndvi_c = np.clip(ndvi, 0, 1)
    chl    = np.clip(cire / 10.0, 0, 1)
    nit    = np.clip(ndre * 2.5 + 0.5, 0, MAX_N_PERCENT) / MAX_N_PERCENT
    bio    = np.clip((0.7 * ndvi_c + 0.3 * (B07 / (B07.max() + eps))) * 120.0 / MAX_BIOMASS_MGHA, 0, 1)
    loss   = np.clip(1.0 - ndvi_c, 0, 1)
    return {"chl": chl, "nit": nit, "bio": bio, "loss": loss}


def prepare_chip(chip_raw: np.ndarray):
    hls      = build_hls_bands(chip_raw)
    H, W     = hls.shape[1], hls.shape[2]
    sar      = build_sar_proxy((H, W))
    indices  = compute_spectral_indices(hls)
    hls_norm = normalize_hls(hls)
    return hls, hls_norm, sar, indices
