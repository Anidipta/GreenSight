import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from config import (
    TILE_SIZE, TILE_STRIDE, DEVICE, ABLATION,
    MAX_CHL_UG_CM2, MAX_N_PERCENT, MAX_BIOMASS_MGHA, HEALTHY_BIOMASS_REF,
    VEG_NDVI_THRESH, STRESS_THRESHOLD, BIOMASS_LOSS_THRESH,
)
from data_processing import (
    build_hls_bands, build_sar_proxy, compute_spectral_indices,
    normalize_hls, tile_image, compute_gt_proxies,
)


def process_chip(
    chip_raw: np.ndarray,
    model,
    device:   str = DEVICE,
    ablation: bool = ABLATION,
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, Optional[object]]:
    hls      = build_hls_bands(chip_raw)
    H, W     = hls.shape[1], hls.shape[2]
    sar      = build_sar_proxy((H, W))
    indices  = compute_spectral_indices(hls)
    hls_norm = normalize_hls(hls)
    tiles    = tile_image(hls_norm, sar, indices, TILE_SIZE, TILE_STRIDE)
    results  = []
    abl_out  = None

    for tile in tiles:
        hls_t = torch.from_numpy(tile["hls"]).float().unsqueeze(0).to(device)
        sar_t = torch.from_numpy(tile["sar"]).float().unsqueeze(0).to(device)
        idx_t = torch.from_numpy(tile["indices"]).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(hls_t, sar_t, idx_t, ablation=ablation)
        if ablation:
            baseline = out.variants["Full model (baseline)"]
            results.append({
                **tile,
                "chl_map":     baseline.chl_map.squeeze().cpu().numpy(),
                "nitro_map":   baseline.nitro_map.squeeze().cpu().numpy(),
                "biomass_map": baseline.biomass_map.squeeze().cpu().numpy(),
                "loss_map":    baseline.loss_map.squeeze().cpu().numpy(),
            })
            if abl_out is None:
                abl_out = out
        else:
            c, n, b, l = out
            results.append({
                **tile,
                "chl_map":     c.squeeze().cpu().numpy(),
                "nitro_map":   n.squeeze().cpu().numpy(),
                "biomass_map": b.squeeze().cpu().numpy(),
                "loss_map":    l.squeeze().cpu().numpy(),
            })

    maps = stitch_maps(results, hls.shape, TILE_SIZE)
    return maps, hls, sar, indices, abl_out


def stitch_maps(
    results:     List[Dict],
    image_shape: Tuple,
    tile_size:   int = TILE_SIZE,
) -> Dict[str, np.ndarray]:
    _, H, W = image_shape
    acc     = {k: np.zeros((H, W), dtype=np.float32)
               for k in ("chlorophyll", "nitrogen", "biomass", "biomass_loss")}
    weight  = np.zeros((H, W), dtype=np.float32)
    key_map = {
        "chlorophyll": "chl_map",
        "nitrogen":    "nitro_map",
        "biomass":     "biomass_map",
        "biomass_loss":"loss_map",
    }
    for r in results:
        sy, ey = r["y_start"], r["y_end"]
        sx, ex = r["x_start"], r["x_end"]
        vh, vw = r["valid_h"], r["valid_w"]
        for out_k, in_k in key_map.items():
            patch = r[in_k]
            if patch.shape != (vh, vw):
                patch = cv2.resize(patch, (vw, vh), interpolation=cv2.INTER_LINEAR)
            acc[out_k][sy:ey, sx:ex] += patch
        weight[sy:ey, sx:ex] += 1.0
    w = np.maximum(weight, 1.0)
    return {k: v / w for k, v in acc.items()} | {"weight": weight}


def compute_metrics(
    maps:         Dict[str, np.ndarray],
    indices_full: Optional[np.ndarray] = None,
) -> Dict:
    chl  = maps["chlorophyll"]
    nit  = maps["nitrogen"]
    bio  = maps["biomass"]
    loss = maps["biomass_loss"]
    H, W = chl.shape
    ndvi     = indices_full[0] if indices_full is not None else np.zeros_like(chl)
    veg_mask = ndvi > VEG_NDVI_THRESH
    total_px = H * W
    stressed_mask = (chl  < STRESS_THRESHOLD)   & veg_mask
    bio_loss_mask = (loss > BIOMASS_LOSS_THRESH) & veg_mask
    veg_pct      = 100.0 * veg_mask.sum() / total_px
    stressed_pct = 100.0 * stressed_mask.sum() / max(veg_mask.sum(), 1)
    bio_loss_pct = 100.0 * bio_loss_mask.sum() / max(veg_mask.sum(), 1)
    def _vm(arr): return float(arr[veg_mask].mean()) if veg_mask.any() else 0.0
    mean_chl_ug   = _vm(chl  * MAX_CHL_UG_CM2)
    mean_n_pct    = _vm(nit  * MAX_N_PERCENT)
    mean_agb      = _vm(bio  * MAX_BIOMASS_MGHA)
    mean_loss_agb = _vm(loss * HEALTHY_BIOMASS_REF)
    mean_chl_raw  = _vm(chl)
    severity = "SEVERE" if stressed_pct > 50 else "MODERATE" if stressed_pct > 20 else "MILD"
    return {
        "image_size_px":            [H, W],
        "total_pixels":             total_px,
        "vegetation_coverage_pct":  round(veg_pct, 2),
        "stressed_area_pct":        round(stressed_pct, 2),
        "biomass_loss_area_pct":    round(bio_loss_pct, 2),
        "chlorophyll_ug_cm2":       round(mean_chl_ug, 3),
        "chlorophyll_pct_healthy":  round(min(100.0, 100.0 * mean_chl_ug / (MAX_CHL_UG_CM2 + 1e-8)), 2),
        "chlorophyll_stress_pct":   round((1 - mean_chl_raw) * 100, 2),
        "n_concentration_pct":      round(mean_n_pct, 3),
        "n_normalized_pct":         round(min(100.0, 100.0 * mean_n_pct / (MAX_N_PERCENT + 1e-8)), 2),
        "biomass_agb_mgha":         round(mean_agb, 2),
        "biomass_pct_of_max":       round(min(100.0, 100.0 * mean_agb / (MAX_BIOMASS_MGHA + 1e-8)), 2),
        "biomass_loss_mgha":        round(mean_loss_agb, 2),
        "biomass_loss_pct":         round(min(100.0, 100.0 * mean_loss_agb / (HEALTHY_BIOMASS_REF + 1e-8)), 2),
        "stress_severity":          severity,
    }


def compute_error_metrics(
    pred_map: np.ndarray,
    gt_map:   np.ndarray,
    mask:     np.ndarray,
) -> Dict:
    if not mask.any():
        return {"mae": None, "rmse": None, "r": None, "bias_pct": None}
    p    = pred_map[mask].ravel()
    g    = gt_map[mask].ravel()
    mae  = float(np.abs(p - g).mean())
    rmse = float(np.sqrt(((p - g) ** 2).mean()))
    r    = float(np.corrcoef(p, g)[0, 1]) if p.std() > 0 and g.std() > 0 else None
    bias = float((p - g).mean() / (g.mean() + 1e-8) * 100)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4),
            "r": round(r, 4) if r is not None else None,
            "bias_pct": round(bias, 2)}


def run_chip_analysis(
    chip_raw: np.ndarray,
    model,
    device:   str  = DEVICE,
    ablation: bool = ABLATION,
) -> Dict:
    gt = compute_gt_proxies(chip_raw)
    maps, hls, sar, indices, abl_out = process_chip(chip_raw, model, device, ablation)
    H, W   = maps["chlorophyll"].shape
    vm     = gt["veg_mask"][:H, :W]
    pred_m = compute_metrics(maps, indices)
    chl_pred_n = maps["chlorophyll"][:H, :W] * MAX_CHL_UG_CM2
    bio_pred_n = maps["biomass"][:H, :W]     * MAX_BIOMASS_MGHA
    err_chl    = compute_error_metrics(chl_pred_n, gt["chl_gt_ug_cm2"][:H, :W], vm)
    err_bio    = compute_error_metrics(bio_pred_n, gt["agb_gt_mgha"][:H, :W],   vm)
    result = {
        "metrics":           pred_m,
        "error_chlorophyll": err_chl,
        "error_biomass":     err_bio,
        "gt_proxies": {
            "chl_gt_ug_cm2": round(float(gt["chl_gt_ug_cm2"][gt["veg_mask"]].mean()), 3)
                              if gt["veg_mask"].any() else 0.0,
            "agb_gt_mgha":   round(float(gt["agb_gt_mgha"][gt["veg_mask"]].mean()), 2)
                              if gt["veg_mask"].any() else 0.0,
            "vegetation_pct": round(gt["veg_pct"], 2),
        },
        "maps": maps,
        "ablation": abl_out.to_dict() if abl_out is not None else None,
    }
    return result
