"""
Test suite for DualTaskCropHealthModel API.
- Downloads validation chips
- Creates test GeoTIFF files
- Makes API calls to /analyze endpoint
- Saves results and validation reports
"""

import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import rasterio
from rasterio.transform import Affine

from config import (
    OUTPUT_DIR, API_HOST, API_PORT, API_VERSION, NUM_TEST_CHIPS,
    BEST_CKPT_PATH, DATA_DIR,
)
from data_loading import download_hls_chips


# ============================================================================
# Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

API_BASE_URL = f"http://{API_HOST}:{API_PORT}"
TEST_GEOTIFF_DIR = Path(OUTPUT_DIR) / "test_geotiffs"
TEST_RESULTS_DIR = Path(OUTPUT_DIR) / "test_results"

TEST_GEOTIFF_DIR.mkdir(parents=True, exist_ok=True)
TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# GeoTIFF Creation
# ============================================================================

def create_geotiff_from_chip(
    chip_data: np.ndarray,
    output_path: Path,
    chip_name: str,
) -> bool:
    """
    Create a GeoTIFF file from chip data.
    
    Args:
        chip_data: numpy array of shape (bands, height, width)
        output_path: Path to save GeoTIFF
        chip_name: Name for logging
        
    Returns:
        True if successful, False otherwise
    """
    try:
        bands, height, width = chip_data.shape
        
        # Create a simple georeferencing (arbitrary but consistent)
        # Using Affine transformation for pixel-to-geographic mapping
        transform = Affine.translation(0, height) * Affine.scale(1, -1)
        
        with rasterio.open(
            str(output_path),
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=bands,
            dtype=chip_data.dtype,
            crs='EPSG:32633',  # UTM Zone 33N (arbitrary)
            transform=transform,
        ) as dst:
            for i in range(bands):
                dst.write(chip_data[i], i + 1)
        
        logger.info(f"✓ Created GeoTIFF: {output_path.name} (shape: {chip_data.shape})")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to create GeoTIFF {chip_name}: {e}")
        return False


def download_and_prepare_test_chips(n_chips: int = 1) -> list:
    """
    Download validation chips and create GeoTIFF files for testing.
    
    Args:
        n_chips: Number of chips to download and prepare
        
    Returns:
        List of (geotiff_path, chip_data) tuples
    """
    logger.info("=" * 70)
    logger.info("Downloading validation chips...")
    logger.info("=" * 70)
    
    try:
        chips, masks, names = download_hls_chips(
            n=n_chips,
            data_dir=DATA_DIR,
        )
        
        geotiff_paths = []
        for i, (chip_data, chip_name) in enumerate(zip(chips, names)):
            output_file = TEST_GEOTIFF_DIR / f"test_chip_{i:02d}_{chip_name}"
            if create_geotiff_from_chip(chip_data, output_file, chip_name):
                geotiff_paths.append((output_file, chip_data, chip_name))
        
        logger.info(f"✓ Prepared {len(geotiff_paths)} test GeoTIFF files")
        return geotiff_paths
        
    except Exception as e:
        logger.error(f"✗ Failed to download chips: {e}")
        return []


# ============================================================================
# API Interaction
# ============================================================================

def check_api_health(timeout: int = 5) -> bool:
    """
    Check if API is running and healthy.
    
    Args:
        timeout: Request timeout in seconds
        
    Returns:
        True if API is healthy, False otherwise
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=timeout,
        )
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"✓ API Health: {health_data['status']}")
            logger.info(f"  Device: {health_data['device']}")
            logger.info(f"  Model loaded: {health_data['model_loaded']}")
            logger.info(f"  Version: {health_data['version']}")
            return health_data['model_loaded']
        else:
            logger.error(f"✗ API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error(f"✗ Cannot connect to API at {API_BASE_URL}")
        return False
    except Exception as e:
        logger.error(f"✗ Health check failed: {e}")
        return False


def analyze_geotiff(geotiff_path: Path, timeout: int = 300) -> Optional[dict]:
    """
    Send GeoTIFF to API for analysis.
    
    Args:
        geotiff_path: Path to GeoTIFF file
        timeout: Request timeout in seconds
        
    Returns:
        Response JSON if successful, None otherwise
    """
    try:
        logger.info(f"\nAnalyzing: {geotiff_path.name}")
        
        with open(geotiff_path, 'rb') as f:
            files = {'file': (geotiff_path.name, f, 'image/tiff')}
            response = requests.post(
                f"{API_BASE_URL}/analyze",
                files=files,
                timeout=timeout,
            )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✓ Analysis successful")
            logger.info(f"  Processing time: {result['processing_time_ms']:.2f}ms")
            return result
        else:
            logger.error(f"✗ API returned status {response.status_code}")
            logger.error(f"  Response: {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"✗ Analysis failed: {e}")
        return None


# ============================================================================
# Results Processing
# ============================================================================

def save_analysis_results(
    result: dict,
    chip_index: int,
    chip_name: str,
) -> Path:
    """
    Save analysis results to JSON file.
    
    Args:
        result: API response JSON
        chip_index: Index of chip being analyzed
        chip_name: Original chip name
        
    Returns:
        Path to saved results file
    """
    output_file = TEST_RESULTS_DIR / f"result_{chip_index:02d}_{chip_name.replace('.tif', '')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"✓ Results saved to: {output_file.name}")
    return output_file


def print_metrics_summary(result: dict):
    """Pretty-print metrics summary."""
    if not result.get('success'):
        logger.warning(f"Analysis failed: {result.get('message')}")
        return
    
    metrics = result.get('metrics', {})
    
    logger.info("\n" + "=" * 70)
    logger.info("BIOPHYSICAL METRICS SUMMARY")
    logger.info("=" * 70)
    
    # Chlorophyll
    chl = metrics.get('chlorophyll', {})
    logger.info(f"\n[CHLOROPHYLL]")
    logger.info(f"  Concentration: {chl.get('chlorophyll_ug_cm2', 'N/A')} µg/cm²")
    logger.info(f"  Health: {chl.get('chlorophyll_pct_healthy', 'N/A')}%")
    logger.info(f"  Stress: {chl.get('chlorophyll_stress_pct', 'N/A')}%")
    
    # Nitrogen
    n = metrics.get('nitrogen', {})
    logger.info(f"\n[NITROGEN]")
    logger.info(f"  Concentration: {n.get('n_concentration_pct', 'N/A')}%")
    logger.info(f"  Normalized: {n.get('n_normalized_pct', 'N/A')}%")
    
    # Biomass
    bio = metrics.get('biomass', {})
    logger.info(f"\n[BIOMASS]")
    logger.info(f"  AGB: {bio.get('biomass_agb_mgha', 'N/A')} Mg/ha")
    logger.info(f"  % of Max: {bio.get('biomass_pct_of_max', 'N/A')}%")
    logger.info(f"  Loss: {bio.get('biomass_loss_mgha', 'N/A')} Mg/ha")
    logger.info(f"  Loss %: {bio.get('biomass_loss_pct', 'N/A')}%")
    logger.info(f"  Loss Area: {bio.get('biomass_loss_area_pct', 'N/A')}%")
    
    # Vegetation
    veg = metrics.get('vegetation', {})
    logger.info(f"\n[VEGETATION]")
    logger.info(f"  Coverage: {veg.get('vegetation_coverage_pct', 'N/A')}%")
    logger.info(f"  Stressed Area: {veg.get('stressed_area_pct', 'N/A')}%")
    logger.info(f"  Stress Severity: {veg.get('stress_severity', 'N/A')}")
    
    # Image Info
    img = metrics.get('image', {})
    logger.info(f"\n[IMAGE]")
    logger.info(f"  Size: {img.get('image_size_px', 'N/A')} pixels")
    logger.info(f"  Total pixels: {img.get('total_pixels', 'N/A'):,}")
    
    logger.info("\n" + "=" * 70)


def generate_test_report(
    test_results: list,
    geotiff_paths: list,
    duration_sec: float,
):
    """Generate comprehensive test report."""
    report_file = TEST_RESULTS_DIR / "test_report.json"
    
    successful = sum(1 for r in test_results if r is not None)
    
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "api_version": API_VERSION,
        "best_model_used": str(BEST_CKPT_PATH),
        "test_summary": {
            "total_chips_tested": len(geotiff_paths),
            "successful_analyses": successful,
            "failed_analyses": len(geotiff_paths) - successful,
            "success_rate_pct": round(100 * successful / len(geotiff_paths), 2) if geotiff_paths else 0,
            "total_duration_sec": round(duration_sec, 2),
        },
        "test_files": {
            "geotiff_dir": str(TEST_GEOTIFF_DIR),
            "results_dir": str(TEST_RESULTS_DIR),
        },
        "results": test_results,
    }
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n✓ Test report saved to: {report_file}")
    return report


# ============================================================================
# Main Test Flow
# ============================================================================

def start_api_server() -> Optional[subprocess.Popen]:
    """
    Start Flask API server in background.
    
    Returns:
        Process object if successful, None otherwise
    """
    try:
        logger.info("Starting API server...")
        proc = subprocess.Popen(
            [sys.executable, "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        # Wait for server to start
        for attempt in range(30):
            time.sleep(1)
            if check_api_health():
                logger.info(f"✓ API server started (PID: {proc.pid})")
                return proc
            logger.debug(f"  Waiting for API... ({attempt + 1}/30)")
        
        logger.error("✗ API server did not start within 30 seconds")
        return None
        
    except Exception as e:
        logger.error(f"✗ Failed to start API server: {e}")
        return None


def run_tests(
    n_chips: int = 1,
    skip_api_start: bool = False,
):
    """
    Run complete test suite.
    
    Args:
        n_chips: Number of chips to test
        skip_api_start: If True, assume API is already running
    """
    start_time = time.time()
    api_process = None
    
    try:
        # Step 1: Start API (optional)
        if not skip_api_start:
            api_process = start_api_server()
            if api_process is None:
                logger.error("Cannot proceed without API server")
                return
        else:
            logger.info("Skipping API startup (assuming it's running)")
        
        # Step 2: Check API health
        logger.info("\nChecking API health...")
        if not check_api_health():
            logger.error("API health check failed")
            return
        
        # Step 3: Download and prepare test chips
        logger.info("\nPreparing test data...")
        geotiff_paths = download_and_prepare_test_chips(n_chips=n_chips)
        
        if not geotiff_paths:
            logger.error("No test data prepared")
            return
        
        # Step 4: Verify model exists
        if not Path(BEST_CKPT_PATH).exists():
            logger.warning(f"⚠ Best model not found at {BEST_CKPT_PATH}")
            logger.info("  This is expected if training hasn't been run yet")
        else:
            logger.info(f"✓ Best model found at {BEST_CKPT_PATH}")
        
        # Step 5: Run analysis on each chip
        logger.info("\nRunning analysis on test chips...")
        logger.info("=" * 70)
        
        test_results = []
        for chip_idx, (geotiff_path, chip_data, chip_name) in enumerate(geotiff_paths):
            result = analyze_geotiff(geotiff_path)
            
            if result:
                test_results.append(result)
                save_analysis_results(result, chip_idx, chip_name)
                print_metrics_summary(result)
            else:
                test_results.append(None)
                logger.error(f"✗ Analysis failed for chip {chip_idx}")
        
        # Step 6: Generate report
        duration = time.time() - start_time
        logger.info("\nGenerating test report...")
        report = generate_test_report(test_results, geotiff_paths, duration)
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("TEST SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Chips tested: {report['test_summary']['total_chips_tested']}")
        logger.info(f"Successful: {report['test_summary']['successful_analyses']}")
        logger.info(f"Failed: {report['test_summary']['failed_analyses']}")
        logger.info(f"Success rate: {report['test_summary']['success_rate_pct']}%")
        logger.info(f"Total duration: {report['test_summary']['total_duration_sec']}s")
        logger.info("\nTest files saved to:")
        logger.info(f"  GeoTIFFs: {TEST_GEOTIFF_DIR}")
        logger.info(f"  Results: {TEST_RESULTS_DIR}")
        logger.info("=" * 70)
        
    except KeyboardInterrupt:
        logger.info("\n⚠ Test interrupted by user")
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
    finally:
        # Step 7: Cleanup
        if api_process is not None:
            logger.info("\nShutting down API server...")
            api_process.terminate()
            try:
                api_process.wait(timeout=5)
                logger.info("✓ API server stopped")
            except subprocess.TimeoutExpired:
                logger.warning("API server did not stop gracefully, killing...")
                api_process.kill()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test DualTaskCropHealthModel API with validation chips"
    )
    parser.add_argument(
        "--chips",
        type=int,
        default=1,
        help=f"Number of validation chips to test (default: 1, max: {NUM_TEST_CHIPS})",
    )
    parser.add_argument(
        "--skip-api-start",
        action="store_true",
        help="Skip API startup (assumes it's already running)",
    )
    
    args = parser.parse_args()
    n_chips = min(args.chips, NUM_TEST_CHIPS)
    
    logger.info("=" * 70)
    logger.info("DualTaskCropHealthModel API Test Suite")
    logger.info("=" * 70)
    logger.info(f"Test configuration:")
    logger.info(f"  Chips to test: {n_chips}")
    logger.info(f"  Best model: {BEST_CKPT_PATH}")
    logger.info(f"  API endpoint: {API_BASE_URL}")
    logger.info("=" * 70)
    
    run_tests(n_chips=n_chips, skip_api_start=args.skip_api_start)
