import os
import json
import tarfile
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

SAMPLE_DIR  = Path("sample_data")
N_CHIPS     = 6
DATASET_REPO = "ibm-nasa-geospatial/multi-temporal-crop-classification"
VAL_ARCHIVE  = "validation_chips.tgz"


def _bounds_to_wgs84(src):
    try:
        from rasterio.warp import transform_bounds
        return transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    except Exception:
        return None


def download_samples():
    import rasterio
    from huggingface_hub import hf_hub_download

    hf_token = os.getenv("HF_TOKEN") or None
    print(f"  HF_TOKEN: {'set' if hf_token else 'not set (public access)'}")

    SAMPLE_DIR.mkdir(exist_ok=True)

    print(f"\n  Downloading {VAL_ARCHIVE} from HuggingFace …")
    tgz_path = hf_hub_download(
        repo_id   = DATASET_REPO,
        filename  = VAL_ARCHIVE,
        repo_type = "dataset",
        token     = hf_token,
    )
    print(f"  Cached: {tgz_path}")

    extract_dir = SAMPLE_DIR / "extracted"
    if not extract_dir.exists():
        print(f"  Extracting → {extract_dir} …")
        extract_dir.mkdir(parents=True)
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        print("  Done.")

    merged_paths = sorted(
        p for p in extract_dir.rglob("*_merged.tif")
        if not p.name.startswith("._")
    )[:N_CHIPS]

    if not merged_paths:
        raise RuntimeError(f"No *_merged.tif found under {extract_dir}")

    manifest = []
    print(f"\n  Copying {len(merged_paths)} sample chips → {SAMPLE_DIR}/\n")

    for i, src_path in enumerate(merged_paths):
        dst_name = src_path.name
        dst_path = SAMPLE_DIR / dst_name
        mask_src = Path(str(src_path).replace("_merged.tif", "_mask.tif"))
        mask_dst = SAMPLE_DIR / mask_src.name if mask_src.exists() else None

        if not dst_path.exists():
            import shutil
            shutil.copy2(src_path, dst_path)
            if mask_src.exists() and not mask_src.name.startswith("._"):
                shutil.copy2(mask_src, SAMPLE_DIR / mask_src.name)

        with rasterio.open(str(dst_path)) as src:
            bounds_wgs84 = _bounds_to_wgs84(src)
            crs_str      = str(src.crs)
            width        = src.width
            height       = src.height
            n_bands      = src.count
            dtype        = str(src.dtypes[0])
            data_preview = src.read([1, 2, 3]).astype(np.float32)
            band_means   = [round(float(data_preview[b].mean()), 2) for b in range(3)]

        if bounds_wgs84:
            west, south, east, north = bounds_wgs84
        else:
            west  = -95.0 + i * 0.02
            south = 40.0  + i * 0.01
            east  = west  + 0.065
            north = south + 0.065

        chip_id = dst_path.stem.replace("_merged", "")
        entry   = {
            "id":         chip_id,
            "filename":   dst_name,
            "path":       str(dst_path.resolve()),
            "crs":        crs_str,
            "width_px":   width,
            "height_px":  height,
            "n_bands":    n_bands,
            "dtype":      dtype,
            "bounds": {
                "west":  round(west,  6),
                "south": round(south, 6),
                "east":  round(east,  6),
                "north": round(north, 6),
            },
            "center": {
                "lat": round((south + north) / 2, 6),
                "lon": round((west  + east)  / 2, 6),
            },
            "band_means_b02_b03_b04": band_means,
        }
        manifest.append(entry)
        print(f"  ✓  [{i+1}/{N_CHIPS}]  {dst_name}")
        print(f"       bounds  W={entry['bounds']['west']:.4f}  S={entry['bounds']['south']:.4f}"
              f"  E={entry['bounds']['east']:.4f}  N={entry['bounds']['north']:.4f}")
        print(f"       size    {width}×{height}  bands={n_bands}  crs={crs_str[:40]}")

    manifest_path = SAMPLE_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Manifest saved → {manifest_path.resolve()}")
    print(f"  Total chips    : {len(manifest)}")
    return manifest


if __name__ == "__main__":
    print("━" * 60)
    print("  SAMPLE DATA DOWNLOADER")
    print("  Dataset : ibm-nasa-geospatial/multi-temporal-crop-classification")
    print("  Output  : sample_data/")
    print("━" * 60)
    manifest = download_samples()
    print("\n  All done. Run main.py to start the API.\n")
