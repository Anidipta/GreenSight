import os
import math
from typing import Dict, List

import cv2
import numpy as np

from config import (
    VEG_NDVI_THRESH, MAX_CHL_UG_CM2, MAX_N_PERCENT,
    MAX_BIOMASS_MGHA, OUTPUT_DIR,
)


def _fmt(label: str, value, width: int = 38) -> str:
    return f"  {label:<{width}}{value}"

def _section(title: str):
    w = 72
    print(f"\n{'━'*w}\n  {title}\n{'━'*w}")

def _norm_band(band: np.ndarray) -> np.ndarray:
    valid = band[band > 0]
    if valid.size < 10:
        return np.zeros_like(band, dtype=np.float32)
    lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
    if hi <= lo:
        return np.zeros_like(band, dtype=np.float32)
    return np.clip((band.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def print_chip_result(
    name:    str,
    idx:     int,
    gt:      Dict,
    pred_m:  Dict,
    err_chl: Dict,
    err_bio: Dict,
    elapsed: float,
):
    chl_pred_ug = pred_m["chlorophyll_pct_healthy"] * MAX_CHL_UG_CM2 / 100
    chl_gt_m    = float(gt["chl_gt_ug_cm2"][gt["veg_mask"]].mean()) if gt["veg_mask"].any() else 0
    bio_gt_m    = float(gt["agb_gt_mgha"][gt["veg_mask"]].mean())   if gt["veg_mask"].any() else 0
    r_chl = f"{err_chl['r']:.3f}" if err_chl["r"] is not None else "N/A"
    r_bio = f"{err_bio['r']:.3f}" if err_bio["r"] is not None else "N/A"
    print(f"\n  ┌─ Chip {idx+1}  ·  {name}")
    print(f"  │  Vegetation coverage    : {gt['veg_pct']:.1f}%")
    print(f"  │")
    print(f"  │  ── Chlorophyll ──────────────────────────────────")
    print(f"  │  GT  (CIre proxy)       : {chl_gt_m:.2f} μg/cm²")
    print(f"  │  Pred (SCAD)            : {chl_pred_ug:.2f} μg/cm²")
    print(f"  │  MAE / RMSE             : {err_chl['mae']:.4f}  /  {err_chl['rmse']:.4f}")
    print(f"  │  Pearson r / Bias       : {r_chl}  /  {err_chl['bias_pct']:.1f}%")
    print(f"  │  Stress level           : {pred_m['chlorophyll_stress_pct']:.1f}%  [{pred_m['stress_severity']}]")
    print(f"  │")
    print(f"  │  ── Biomass ───────────────────────────────────────")
    print(f"  │  GT  (NDVI-AGB proxy)   : {bio_gt_m:.1f} Mg/ha")
    print(f"  │  Pred (PMFD)            : {pred_m['biomass_agb_mgha']:.1f} Mg/ha")
    print(f"  │  MAE / RMSE             : {err_bio['mae']:.2f}  /  {err_bio['rmse']:.2f}")
    print(f"  │  Pearson r / Bias       : {r_bio}  /  {err_bio['bias_pct']:.1f}%")
    print(f"  │  Biomass loss area      : {pred_m['biomass_loss_area_pct']:.1f}% of veg")
    print(f"  │  N concentration        : {pred_m['n_concentration_pct']:.3f}%")
    print(f"  └─ Inference time         : {elapsed:.2f}s")


def print_aggregate_table(rows: List[Dict]):
    _section("AGGREGATE TEST RESULTS")
    hdr = (f"  {'Chip':<22} {'Veg%':>6} {'ChlStr%':>8} {'ChlGT':>9} {'ChlPr':>9}"
           f" {'N%':>6} {'AGB_GT':>7} {'AGB_Pr':>7} {'BioLoss%':>9} {'Sev':>8}  {'t(s)':>5}")
    sep = "  " + "─" * (len(hdr) - 2)
    print(hdr); print(sep)
    for r in rows:
        print(f"  {r['name'][:20]:<22} {r['veg_cov_pct']:>5.1f}% {r['chl_stress_pct']:>7.1f}%"
              f" {r['chl_gt_ug']:>8.2f} {r['chl_pred_ug']:>8.2f}"
              f" {r['n_pct']:>5.3f}% {r['bio_gt_mh']:>6.1f} {r['bio_pred_mh']:>6.1f}"
              f" {r['bio_loss_pct']:>8.1f}% {r['severity']:>8}  {r['time_s']:>4.1f}s")
    print(sep)
    def avg(k): return float(np.nanmean([r[k] for r in rows]))
    print(f"\n  Chlorophyll  MAE : {avg('chl_mae'):.4f} μg/cm²   RMSE : {avg('chl_rmse'):.4f}   r : {avg('chl_r'):.3f}")
    print(f"  Biomass      MAE : {avg('bio_mae'):.2f} Mg/ha    RMSE : {avg('bio_rmse'):.2f}    r : {avg('bio_r'):.3f}")
    total_t = sum(r["time_s"] for r in rows)
    print(f"  Total time       : {total_t:.1f}s   ({total_t/len(rows):.2f}s / chip)")


def render_chip_panel(
    chips_raw:  List[np.ndarray],
    gt_list:    List[Dict],
    names:      List[str],
    output_dir: str = OUTPUT_DIR,
):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    n    = len(chips_raw)
    cols = 3
    fig, axes = plt.subplots(n, cols, figsize=(cols * 3.6, n * 3.2),
                             facecolor="#0d1117", squeeze=False)
    fig.suptitle("Crop Health Monitor  ·  Prithvi-EO-1.0-100M",
                 color="white", fontsize=12, fontweight="bold", y=1.002)
    ndvi_cmap = LinearSegmentedColormap.from_list("ndvi", ["#8B4513", "#F5DEB3", "#228B22"])

    for ci in range(n):
        chip = chips_raw[ci]
        gt   = gt_list[ci]
        name = names[ci]
        H, W = chip.shape[1], chip.shape[2]
        natural = np.stack([_norm_band(chip[2]), _norm_band(chip[1]), _norm_band(chip[0])], axis=-1).clip(0, 1)
        falsecolor = np.stack([_norm_band(chip[5]), _norm_band(chip[3]), _norm_band(chip[2])], axis=-1).clip(0, 1)
        ndvi_disp = np.clip(gt["ndvi"][:H, :W].astype(np.float32), -0.2, 0.9)
        lbl = name.replace("_merged.tif", "").replace("_", " ")
        for ax in axes[ci]:
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("#0d1117")
            for sp in ax.spines.values(): sp.set_edgecolor("#222233")
        axes[ci][0].imshow(natural, interpolation="bilinear")
        axes[ci][0].set_ylabel(lbl, color="#aaaacc", fontsize=7,
                               rotation=0, labelpad=4, ha="right", va="center")
        axes[ci][1].imshow(falsecolor, interpolation="bilinear")
        im = axes[ci][2].imshow(ndvi_disp, cmap=ndvi_cmap, vmin=-0.2, vmax=0.9, interpolation="bilinear")
        cb = fig.colorbar(im, ax=axes[ci][2], fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelcolor="#aaaaaa", labelsize=5, length=2)
        cb.set_label("NDVI", color="#aaaaaa", fontsize=6)

    for c, t in enumerate(["Natural Color\n(B04-B03-B02)",
                            "False Color\n(B07-B05-B04)",
                            "NDVI\n(vegetation index)"]):
        axes[0][c].set_title(t, color="#ccccff", fontsize=8, fontweight="bold", pad=5)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "panel_maps.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓  Panel saved  →  {os.path.abspath(out)}")


def render_metrics_bar_chart(rows: List[Dict], output_dir: str = OUTPUT_DIR):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    names      = [r["name"][:18]       for r in rows]
    chl_stress = [r["chl_stress_pct"]  for r in rows]
    bio_loss   = [r["bio_loss_pct"]    for r in rows]
    veg_cov    = [r["veg_cov_pct"]     for r in rows]
    n_level    = [r["n_norm_pct"]      for r in rows]
    x, w       = np.arange(len(names)), 0.20
    fig, ax = plt.subplots(figsize=(max(10, 2.2 * len(names)), 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.bar(x - 1.5*w, veg_cov,    w, label="Veg Coverage %", color="#2ecc71", alpha=0.85)
    ax.bar(x - 0.5*w, chl_stress, w, label="Chl Stress %",   color="#e74c3c", alpha=0.85)
    ax.bar(x + 0.5*w, n_level,    w, label="N Level %",       color="#3498db", alpha=0.85)
    ax.bar(x + 1.5*w, bio_loss,   w, label="Biomass Loss %",  color="#e67e22", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", color="#cccccc", fontsize=8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.tick_params(colors="#aaaaaa")
    ax.set_ylabel("Percentage (%)", color="#aaaaaa")
    ax.set_title("Per-Chip Metrics  ·  Prithvi-EO-1.0-100M + SCAD + PMFD",
                 color="white", fontweight="bold", fontsize=11)
    ax.legend(loc="upper right", facecolor="#1a1a2e", edgecolor="#555555",
              labelcolor="white", fontsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#333344")
    ax.grid(axis="y", color="#222233", linewidth=0.6)
    plt.tight_layout()
    out = os.path.join(output_dir, "metrics_bar.png")
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓  Bar chart saved  →  {os.path.abspath(out)}")


def render_correlation_scatter(rows: List[Dict], output_dir: str = OUTPUT_DIR):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chl_pred = [r["chl_pred_ug"] for r in rows]
    chl_gt   = [r["chl_gt_ug"]   for r in rows]
    bio_pred = [r["bio_pred_mh"] for r in rows]
    bio_gt   = [r["bio_gt_mh"]   for r in rows]
    colors   = plt.cm.plasma(np.linspace(0.2, 0.9, len(chl_pred)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#111122")
        for sp in ax.spines.values(): sp.set_edgecolor("#333344")
        ax.tick_params(colors="#aaaaaa")
    for ax, pred, gt_v, title, xlabel, ylabel in [
        (axes[0], chl_pred, chl_gt,
         "Chlorophyll: Pred vs GT", "GT Chlorophyll (μg/cm²)", "Pred (SCAD)"),
        (axes[1], bio_pred, bio_gt,
         "Biomass: Pred vs GT", "GT Biomass (Mg/ha)", "Pred (PMFD)"),
    ]:
        ax.scatter(gt_v, pred, c=colors, s=90, zorder=5, edgecolors="white", linewidths=0.6)
        lim = max(max(gt_v + pred) * 1.1, 1)
        ax.plot([0, lim], [0, lim], "--", color="#666688", lw=1.2, label="1:1 line")
        ax.set_xlabel(xlabel, color="#aaaaaa")
        ax.set_ylabel(ylabel, color="#aaaaaa")
        ax.set_title(title, color="white", fontweight="bold")
        ax.legend(facecolor="#1a1a2e", edgecolor="#555555", labelcolor="white")
        if len(pred) > 1:
            try:
                r_val = np.corrcoef(gt_v, pred)[0, 1]
                ax.annotate(f"r = {r_val:.3f}", xy=(0.05, 0.88), xycoords="axes fraction",
                            color="#00ff88", fontsize=11, fontweight="bold")
            except Exception:
                pass
    plt.suptitle("Pred vs Spectral GT Proxies", color="white", fontweight="bold", fontsize=12)
    plt.tight_layout()
    out = os.path.join(output_dir, "correlation_scatter.png")
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓  Scatter saved  →  {os.path.abspath(out)}")
