import os
import time
from typing import Dict, List

import numpy as np
import torch

from config import DEVICE, ABLATION_VARIANTS as _CFG, OUTPUT_DIR
from ablation import ABLATION_VARIANTS, AblationOutput
from model import DualTaskCropHealthModel
from data_processing import (
    build_hls_bands, build_sar_proxy, compute_spectral_indices,
    normalize_hls, tile_image,
)
from config import TILE_SIZE, TILE_STRIDE
from utils import _section, _fmt


def run_ablation_on_chip(
    chip_raw: np.ndarray,
    model:    DualTaskCropHealthModel,
    device:   str = DEVICE,
) -> AblationOutput:
    hls      = build_hls_bands(chip_raw)
    H, W     = hls.shape[1], hls.shape[2]
    sar      = build_sar_proxy((H, W))
    indices  = compute_spectral_indices(hls)
    hls_norm = normalize_hls(hls)
    tiles    = tile_image(hls_norm, sar, indices, TILE_SIZE, TILE_STRIDE)

    abl_out = None
    for tile in tiles:
        hls_t = torch.from_numpy(tile["hls"]).float().unsqueeze(0).to(device)
        sar_t = torch.from_numpy(tile["sar"]).float().unsqueeze(0).to(device)
        idx_t = torch.from_numpy(tile["indices"]).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(hls_t, sar_t, idx_t, ablation=True)
        if abl_out is None:
            abl_out = out
    return abl_out


def run_ablation_suite(
    chips:  List[np.ndarray],
    names:  List[str],
    model:  DualTaskCropHealthModel,
    device: str = DEVICE,
) -> Dict[str, AblationOutput]:
    _section(f"ABLATION SUITE  ·  {len(chips)} chips  ·  {len(ABLATION_VARIANTS)} variants")
    results = {}
    for i, (chip, name) in enumerate(zip(chips, names)):
        print(f"\n  ── Chip {i+1}/{len(chips)}  ·  {name}")
        t0  = time.time()
        out = run_ablation_on_chip(chip, model, device)
        print(f"  {out.summary_table()}")
        print(f"  Time: {time.time()-t0:.1f}s")
        results[name] = out
    return results


def ablation_to_report(ablation_results: Dict[str, AblationOutput]) -> Dict:
    report = {}
    for chip_name, abl_out in ablation_results.items():
        report[chip_name] = abl_out.to_dict()
    return report


if __name__ == "__main__":
    from data_loading import download_hls_chips
    from model import build_model

    _section("ABLATION STUDY  ·  Standalone Run")
    chips, _, names = download_hls_chips()
    model = build_model(DEVICE)
    model.eval()
    results = run_ablation_suite(chips, names, model, DEVICE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import json
    report_path = os.path.join(OUTPUT_DIR, "ablation_report.json")
    with open(report_path, "w") as f:
        json.dump(ablation_to_report(results), f, indent=2, default=str)
    print(f"\n  ✓  Report saved → {os.path.abspath(report_path)}")
