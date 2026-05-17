import io
import json
import os
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

load_dotenv()

DEVICE        = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
API_HOST      = os.getenv("API_HOST", "0.0.0.0")
API_PORT      = int(os.getenv("API_PORT", "5000"))
API_DEBUG     = os.getenv("API_DEBUG", "false").lower() == "true"
SAMPLE_DIR    = Path(os.getenv("SAMPLE_DIR", "sample_data"))
BEST_CKPT     = os.getenv("BEST_CKPT", "output/best_model.pt")
CKPT_FALLBACK = "output/trained_model.pt"
API_VERSION   = "v1"
MAX_TILES_AOI = 64

from config import (
    MAX_CHL_UG_CM2, MAX_N_PERCENT, MAX_BIOMASS_MGHA,
    HEALTHY_BIOMASS_REF, VEG_NDVI_THRESH, ABLATION,
)
from model import build_model, load_model
from inference import stitch_maps, compute_metrics, compute_error_metrics
from data_processing import (
    build_hls_bands, build_sar_proxy, compute_spectral_indices,
    normalize_hls, tile_image, compute_gt_proxies,
)

app = Flask(__name__, static_folder="frontend", template_folder="frontend")
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024

_model:     object = None
_model_lock        = threading.Lock()
_model_meta: Dict  = {}
_jobs: Dict[str, Dict] = {}
_manifest: List[Dict]  = []


def _load_manifest():
    global _manifest
    mp = SAMPLE_DIR / "manifest.json"
    if mp.exists():
        with open(mp) as f:
            _manifest = json.load(f)
        print(f"  [API] Manifest loaded: {len(_manifest)} chips")
    else:
        print(f"  [API] manifest.json not found at {mp}  —  run temp.py first")


def _get_model():
    global _model, _model_meta
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        ckpt = BEST_CKPT if os.path.exists(BEST_CKPT) else (
               CKPT_FALLBACK if os.path.exists(CKPT_FALLBACK) else None)
        _model = load_model(ckpt, DEVICE) if ckpt else build_model(DEVICE)
        _model.eval()
        pc = _model.param_counts
        _model_meta = {
            "checkpoint":        ckpt,
            "device":            DEVICE,
            "params_total":      pc["total"],
            "params_backbone":   pc["backbone"],
            "params_scad":       pc["scad"],
            "params_pmfd":       pc["pmfd"],
            "backbone_strategy": _model.backbone._strategy,
        }
        print(f"  [API] Model ready  ({pc['total']:,} params)  ckpt={ckpt}")
    return _model


def _bbox_intersects(a: Dict, b: Dict) -> bool:
    return (
        a["west"]  < b["east"]  and a["east"]  > b["west"] and
        a["south"] < b["north"] and a["north"] > b["south"]
    )


def _bbox_intersection(a: Dict, b: Dict) -> Dict:
    return {
        "west":  max(a["west"],  b["west"]),
        "south": max(a["south"], b["south"]),
        "east":  min(a["east"],  b["east"]),
        "north": min(a["north"], b["north"]),
    }


def _read_chip_aoi(chip_path: str, aoi_bbox: Dict) -> Optional[np.ndarray]:
    import rasterio
    import rasterio.windows
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds as win_from_bounds
    try:
        with rasterio.open(chip_path) as src:
            aoi_src = transform_bounds(
                "EPSG:4326", src.crs,
                aoi_bbox["west"], aoi_bbox["south"],
                aoi_bbox["east"], aoi_bbox["north"],
            )
            win = win_from_bounds(*aoi_src, src.transform)
            win = win.intersection(
                rasterio.windows.Window(0, 0, src.width, src.height)
            )
            if win.width < 1 or win.height < 1:
                return None
            data = src.read(window=win).astype(np.float32)
        return data[:6] if data.shape[0] >= 6 else data
    except Exception as e:
        print(f"  [read_chip_aoi] {e}")
        return None


def _push(job_id: str, event: Dict):
    if job_id in _jobs:
        _jobs[job_id]["events"].append(event)


def _severity_color(sev: str) -> str:
    return {"MILD": "#22c55e", "MODERATE": "#f59e0b", "SEVERE": "#ef4444"}.get(sev, "#6b7280")


def _process_aoi(job_id: str, aoi_bbox: Dict, ablation: bool):
    try:
        _push(job_id, {"type": "status", "message": "Searching for chips in AOI …"})

        hits = [c for c in _manifest if _bbox_intersects(c["bounds"], aoi_bbox)]
        if not hits:
            _push(job_id, {
                "type":    "error",
                "message": "No sample chips found in that area. Draw your rectangle over a highlighted green region.",
            })
            return

        _push(job_id, {"type": "status", "message": f"Found {len(hits)} chip(s). Loading model …"})
        model = _get_model()

        for ci, chip in enumerate(hits):
            inter = _bbox_intersection(chip["bounds"], aoi_bbox)
            _push(job_id, {
                "type":    "progress",
                "value":   ci / len(hits),
                "message": f"Chip {ci+1}/{len(hits)}: reading window …",
            })

            raw = _read_chip_aoi(chip["path"], inter)
            if raw is None or raw.size == 0:
                _push(job_id, {"type": "warning", "message": f"Chip {chip['id']}: empty window, skipping."})
                continue

            hls      = build_hls_bands(raw)
            H, W     = hls.shape[1], hls.shape[2]
            sar      = build_sar_proxy((H, W))
            indices  = compute_spectral_indices(hls)
            hls_norm = normalize_hls(hls)
            tiles    = tile_image(hls_norm, sar, indices)

            if len(tiles) > MAX_TILES_AOI:
                tiles = tiles[:MAX_TILES_AOI]
                _push(job_id, {"type": "warning", "message": f"AOI clipped to {MAX_TILES_AOI} tiles."})

            _push(job_id, {"type": "status", "message": f"Running model on {len(tiles)} tile(s) …"})

            tile_results = []
            import cv2
            for ti, tile in enumerate(tiles):
                hls_t = torch.from_numpy(tile["hls"]).float().unsqueeze(0).to(DEVICE)
                sar_t = torch.from_numpy(tile["sar"]).float().unsqueeze(0).to(DEVICE)
                idx_t = torch.from_numpy(tile["indices"]).float().unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    c, n, b, l = model(hls_t, sar_t, idx_t, ablation=False)
                tile_results.append({
                    **tile,
                    "chl_map":     c.squeeze().cpu().numpy(),
                    "nitro_map":   n.squeeze().cpu().numpy(),
                    "biomass_map": b.squeeze().cpu().numpy(),
                    "loss_map":    l.squeeze().cpu().numpy(),
                })
                step = max(1, len(tiles) // 8)
                if (ti + 1) % step == 0 or ti == len(tiles) - 1:
                    frac = (ci + (ti + 1) / len(tiles)) / len(hits)
                    _push(job_id, {
                        "type":    "progress",
                        "value":   round(frac, 3),
                        "message": f"Tile {ti+1}/{len(tiles)} …",
                    })

            maps   = stitch_maps(tile_results, hls.shape)
            pred_m = compute_metrics(maps, indices)
            gt     = compute_gt_proxies(raw)
            MH     = maps["chlorophyll"].shape[0]
            MW     = maps["chlorophyll"].shape[1]
            vm     = gt["veg_mask"][:MH, :MW]

            err_chl = compute_error_metrics(
                maps["chlorophyll"][:MH, :MW] * MAX_CHL_UG_CM2,
                gt["chl_gt_ug_cm2"][:MH, :MW], vm,
            )
            err_bio = compute_error_metrics(
                maps["biomass"][:MH, :MW] * MAX_BIOMASS_MGHA,
                gt["agb_gt_mgha"][:MH, :MW], vm,
            )

            _push(job_id, {
                "type":    "result",
                "chip_id": chip["id"],
                "bbox":    inter,
                "metrics": pred_m,
                "gt_proxies": {
                    "chl_gt_ug_cm2":  round(float(gt["chl_gt_ug_cm2"][gt["veg_mask"]].mean()), 3)
                                      if gt["veg_mask"].any() else 0.0,
                    "agb_gt_mgha":    round(float(gt["agb_gt_mgha"][gt["veg_mask"]].mean()), 2)
                                      if gt["veg_mask"].any() else 0.0,
                    "vegetation_pct": round(gt["veg_pct"], 2),
                },
                "error_chlorophyll": err_chl,
                "error_biomass":     err_bio,
                "severity_color":    _severity_color(pred_m["stress_severity"]),
            })

        _push(job_id, {"type": "progress", "value": 1.0, "message": "Done."})
        _push(job_id, {"type": "done"})

    except Exception as e:
        traceback.print_exc()
        _push(job_id, {"type": "error", "message": str(e)})
    finally:
        _jobs[job_id]["done"] = True


@app.route("/")
def index():
    return send_from_directory("index.html")

@app.route("/style.css")
def css():
    return send_from_directory("frontend", "style.css")

@app.route("/app.js")
def js():
    return send_from_directory("frontend", "app.js")


@app.route(f"/api/{API_VERSION}/health")
def health():
    return jsonify({
        "status":       "ok",
        "api_version":  API_VERSION,
        "device":       DEVICE,
        "model_ready":  _model is not None,
        "chips_loaded": len(_manifest),
    })


@app.route(f"/api/{API_VERSION}/model/info")
def model_info():
    _get_model()
    return jsonify({"status": "ok", "ablation_enabled": ABLATION, **_model_meta})


@app.route(f"/api/{API_VERSION}/chips")
def list_chips():
    if not _manifest:
        return jsonify({"status": "error", "message": "No chips. Run temp.py first."}), 404
    return jsonify({"status": "ok", "count": len(_manifest), "chips": _manifest})


@app.route(f"/api/{API_VERSION}/analyze/aoi", methods=["POST"])
def analyze_aoi():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"status": "error", "message": "JSON body with west/south/east/north required."}), 400

    for key in ("west", "south", "east", "north"):
        if key not in body:
            return jsonify({"status": "error", "message": f"Missing field: {key}"}), 400

    aoi  = {k: float(body[k]) for k in ("west", "south", "east", "north")}
    ablation = str(body.get("ablation", "false")).lower() == "true"

    if aoi["west"] >= aoi["east"] or aoi["south"] >= aoi["north"]:
        return jsonify({"status": "error", "message": "Degenerate bbox: west≥east or south≥north."}), 400

    if not _manifest:
        return jsonify({"status": "error", "message": "No chips loaded. Run temp.py first."}), 503

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"events": [], "done": False}
    threading.Thread(target=_process_aoi, args=(job_id, aoi, ablation), daemon=True).start()
    return jsonify({"status": "ok", "job_id": job_id})


@app.route(f"/api/{API_VERSION}/stream/<job_id>")
def stream_job(job_id: str):
    def generate():
        if job_id not in _jobs:
            yield f"data: {json.dumps({'type':'error','message':'Job not found'})}\n\n"
            return
        sent    = 0
        timeout = time.time() + 300
        while time.time() < timeout:
            job    = _jobs[job_id]
            events = job["events"]
            while sent < len(events):
                yield f"data: {json.dumps(events[sent])}\n\n"
                sent += 1
            if job["done"] and sent >= len(events):
                break
            time.sleep(0.08)
        _jobs.pop(job_id, None)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


@app.errorhandler(400)
def bad_request(e):
    return jsonify({"status": "error", "message": str(e)}), 400

@app.errorhandler(413)
def too_large(_e):
    return jsonify({"status": "error", "message": "File too large."}), 413

@app.errorhandler(500)
def internal_error(_e):
    return jsonify({"status": "error", "message": "Internal server error."}), 500


if __name__ == "__main__":
    print(f"\n  Crop Health Monitor  ·  API v{API_VERSION}")
    print(f"  Device   : {DEVICE}")
    print(f"  Samples  : {SAMPLE_DIR}")
    _load_manifest()
    _get_model()
    print(f"\n  http://localhost:{API_PORT}/\n")
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG, threaded=True)
