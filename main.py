"""
Flask API for biophysical metrics extraction from satellite imagery.
Accepts GeoTIFF images with multi-spectral bands and returns analyzed metrics.
"""

import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import rasterio
import torch
from flask import Flask, request, jsonify
from pydantic import BaseModel, Field, ValidationError

from config import (
    API_HOST, API_PORT, API_DEBUG, API_VERSION, MAX_UPLOAD_MB,
    DEVICE, OUTPUT_DIR, BEST_CKPT_PATH,
)
from inference import run_chip_analysis
from model import DualTaskCropHealthModel, build_model


# ============================================================================
# Pydantic v2 Models for Request/Response Validation
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(description="Service status")
    timestamp: str = Field(description="Current timestamp")
    version: str = Field(description="API version")
    device: str = Field(description="PyTorch device being used")
    model_loaded: bool = Field(description="Whether model is loaded")


class ChlorophyllMetrics(BaseModel):
    """Chlorophyll-related metrics."""
    chlorophyll_ug_cm2: float = Field(
        description="Mean chlorophyll concentration in µg/cm²"
    )
    chlorophyll_pct_healthy: float = Field(
        description="Chlorophyll health percentage (0-100)"
    )
    chlorophyll_stress_pct: float = Field(
        description="Chlorophyll stress percentage"
    )


class NitrogenMetrics(BaseModel):
    """Nitrogen-related metrics."""
    n_concentration_pct: float = Field(
        description="Mean nitrogen concentration percentage"
    )
    n_normalized_pct: float = Field(
        description="Normalized nitrogen percentage relative to maximum"
    )


class BiomassMetrics(BaseModel):
    """Biomass-related metrics."""
    biomass_agb_mgha: float = Field(
        description="Above-ground biomass in Mg/ha"
    )
    biomass_pct_of_max: float = Field(
        description="Biomass as percentage of maximum theoretical"
    )
    biomass_loss_mgha: float = Field(
        description="Estimated biomass loss in Mg/ha"
    )
    biomass_loss_pct: float = Field(
        description="Biomass loss as percentage"
    )
    biomass_loss_area_pct: float = Field(
        description="Area affected by biomass loss percentage"
    )


class VegetationMetrics(BaseModel):
    """Vegetation coverage and stress metrics."""
    vegetation_coverage_pct: float = Field(
        description="Vegetation coverage percentage (0-100)"
    )
    stressed_area_pct: float = Field(
        description="Area under stress percentage"
    )
    stress_severity: str = Field(
        description="Stress severity level (MILD, MODERATE, SEVERE)"
    )


class ImageMetadata(BaseModel):
    """Image metadata."""
    image_size_px: list = Field(description="Image dimensions [height, width]")
    total_pixels: int = Field(description="Total pixel count")


class BiophysicalMetrics(BaseModel):
    """Complete biophysical analysis results."""
    chlorophyll: ChlorophyllMetrics = Field(description="Chlorophyll metrics")
    nitrogen: NitrogenMetrics = Field(description="Nitrogen metrics")
    biomass: BiomassMetrics = Field(description="Biomass metrics")
    vegetation: VegetationMetrics = Field(description="Vegetation metrics")
    image: ImageMetadata = Field(description="Image metadata")


class AnalysisResponse(BaseModel):
    """Complete API response for analysis endpoint."""
    success: bool = Field(description="Whether analysis succeeded")
    timestamp: str = Field(description="Analysis timestamp")
    file_name: str = Field(description="Input file name")
    message: Optional[str] = Field(
        default=None,
        description="Status message or error details"
    )
    metrics: Optional[BiophysicalMetrics] = Field(
        default=None,
        description="Biophysical metrics if successful"
    )
    processing_time_ms: float = Field(
        description="Processing time in milliseconds"
    )


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = Field(default=False)
    error: str = Field(description="Error message")
    timestamp: str = Field(description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )


# ============================================================================
# Flask Application Setup
# ============================================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global model instance
model = None


def load_model():
    """Load the trained model into memory."""
    global model
    try:
        logger.info("Loading DualTaskCropHealthModel model...")
        model = build_model(device=DEVICE)
        
        if Path(BEST_CKPT_PATH).exists():
            logger.info(f"Loading checkpoint from {BEST_CKPT_PATH}...")
            checkpoint = torch.load(BEST_CKPT_PATH, map_location=DEVICE, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"✓ Model checkpoint loaded")
        else:
            logger.warning(f"Checkpoint not found at {BEST_CKPT_PATH}")
            logger.info("  Model will run with randomly initialized weights")
        
        model.eval()
        logger.info(f"✓ Model ready on device: {DEVICE}")
        logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return False


def load_geotiff_from_file(file) -> Optional[np.ndarray]:
    """
    Load GeoTIFF file and extract multi-spectral bands.
    
    Args:
        file: FileStorage object from Flask request
        
    Returns:
        numpy array of shape (bands, height, width) or None if error
    """
    try:
        file_bytes = file.read()
        with rasterio.open(io.BytesIO(file_bytes)) as src:
            # Read all bands
            data = src.read().astype(np.float32)
            logger.info(f"Loaded GeoTIFF: shape={data.shape}, dtype={data.dtype}")
            return data
    except Exception as e:
        logger.error(f"Error loading GeoTIFF: {e}")
        return None


def parse_metrics_from_result(result: Dict) -> BiophysicalMetrics:
    """
    Parse inference result into structured BiophysicalMetrics.
    
    Args:
        result: Dictionary from run_chip_analysis()
        
    Returns:
        BiophysicalMetrics Pydantic model
    """
    metrics = result["metrics"]

    return BiophysicalMetrics(
        chlorophyll=ChlorophyllMetrics(
            chlorophyll_ug_cm2=metrics["chlorophyll_ug_cm2"],
            chlorophyll_pct_healthy=metrics["chlorophyll_pct_healthy"],
            chlorophyll_stress_pct=metrics["chlorophyll_stress_pct"],
        ),
        nitrogen=NitrogenMetrics(
            n_concentration_pct=metrics["n_concentration_pct"],
            n_normalized_pct=metrics["n_normalized_pct"],
        ),
        biomass=BiomassMetrics(
            biomass_agb_mgha=metrics["biomass_agb_mgha"],
            biomass_pct_of_max=metrics["biomass_pct_of_max"],
            biomass_loss_mgha=metrics["biomass_loss_mgha"],
            biomass_loss_pct=metrics["biomass_loss_pct"],
            biomass_loss_area_pct=metrics["biomass_loss_area_pct"],
        ),
        vegetation=VegetationMetrics(
            vegetation_coverage_pct=metrics["vegetation_coverage_pct"],
            stressed_area_pct=metrics["stressed_area_pct"],
            stress_severity=metrics["stress_severity"],
        ),
        image=ImageMetadata(
            image_size_px=metrics["image_size_px"],
            total_pixels=metrics["total_pixels"],
        ),
    )


# ============================================================================
# API Endpoints
# ============================================================================

@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON response with service status and model readiness
    """
    try:
        response = HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            version=API_VERSION,
            device=DEVICE,
            model_loaded=(model is not None),
        )
        return jsonify(response.model_dump()), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        error_resp = ErrorResponse(
            error="Health check failed",
            timestamp=datetime.utcnow().isoformat(),
            details={"exception": str(e)},
        )
        return jsonify(error_resp.model_dump()), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Main analysis endpoint. Accepts GeoTIFF file upload and returns biophysical metrics.
    
    Request: multipart/form-data
        - file: GeoTIFF file (required, max 512MB)
    
    Response: JSON with analyzed biophysical metrics
    
    Returns:
        AnalysisResponse JSON with metrics or error details
    """
    start_time = datetime.utcnow()

    try:
        # Validate model is loaded
        if model is None:
            error_resp = ErrorResponse(
                error="Model not loaded",
                timestamp=datetime.utcnow().isoformat(),
                details={"suggestion": "Check server logs during startup"},
            )
            return jsonify(error_resp.model_dump()), 503

        # Check for file in request
        if "file" not in request.files:
            error_resp = ErrorResponse(
                error="No file provided",
                timestamp=datetime.utcnow().isoformat(),
                details={
                    "expected_field": "file",
                    "content_type": "multipart/form-data",
                },
            )
            return jsonify(error_resp.model_dump()), 400

        file = request.files["file"]

        if file.filename == "":
            error_resp = ErrorResponse(
                error="Empty filename",
                timestamp=datetime.utcnow().isoformat(),
            )
            return jsonify(error_resp.model_dump()), 400

        # Validate file extension
        if not file.filename.lower().endswith((".tif", ".tiff")):
            error_resp = ErrorResponse(
                error="Invalid file format",
                timestamp=datetime.utcnow().isoformat(),
                details={
                    "accepted_formats": [".tif", ".tiff"],
                    "received": Path(file.filename).suffix,
                },
            )
            return jsonify(error_resp.model_dump()), 400

        logger.info(f"Processing file: {file.filename}")

        # Load GeoTIFF
        chip_data = load_geotiff_from_file(file)
        if chip_data is None:
            error_resp = ErrorResponse(
                error="Failed to load GeoTIFF file",
                timestamp=datetime.utcnow().isoformat(),
                details={"file_name": file.filename},
            )
            return jsonify(error_resp.model_dump()), 400

        # Validate data shape (expect at least 4 bands for HLS, optional SAR)
        if len(chip_data.shape) != 3 or chip_data.shape[0] < 4:
            error_resp = ErrorResponse(
                error="Invalid GeoTIFF structure",
                timestamp=datetime.utcnow().isoformat(),
                details={
                    "expected": "3D array with at least 4 bands",
                    "received_shape": list(chip_data.shape),
                },
            )
            return jsonify(error_resp.model_dump()), 400

        # Run inference
        logger.info("Starting chip analysis...")
        with torch.no_grad():
            result = run_chip_analysis(chip_data, model, device=DEVICE)

        # Parse and validate metrics
        metrics = parse_metrics_from_result(result)

        # Create response
        elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        response = AnalysisResponse(
            success=True,
            timestamp=datetime.utcnow().isoformat(),
            file_name=file.filename,
            message="Analysis completed successfully",
            metrics=metrics,
            processing_time_ms=round(elapsed_ms, 2),
        )

        logger.info(f"✓ Analysis complete in {elapsed_ms:.0f}ms")
        return jsonify(response.model_dump()), 200

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        error_resp = ErrorResponse(
            error="Response validation failed",
            timestamp=datetime.utcnow().isoformat(),
            details={"validation_errors": e.errors()},
        )
        return jsonify(error_resp.model_dump()), 500

    except Exception as e:
        logger.error(f"Unexpected error in /analyze: {e}", exc_info=True)
        error_resp = ErrorResponse(
            error="Internal server error",
            timestamp=datetime.utcnow().isoformat(),
            details={"exception": str(e)},
        )
        return jsonify(error_resp.model_dump()), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors."""
    error_resp = ErrorResponse(
        error="File too large",
        timestamp=datetime.utcnow().isoformat(),
        details={
            "max_size_mb": MAX_UPLOAD_MB,
            "message": f"Maximum upload size is {MAX_UPLOAD_MB}MB",
        },
    )
    return jsonify(error_resp.model_dump()), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    error_resp = ErrorResponse(
        error="Endpoint not found",
        timestamp=datetime.utcnow().isoformat(),
        details={
            "available_endpoints": [
                "GET /health",
                "POST /analyze",
            ],
        },
    )
    return jsonify(error_resp.model_dump()), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    error_resp = ErrorResponse(
        error="Method not allowed",
        timestamp=datetime.utcnow().isoformat(),
    )
    return jsonify(error_resp.model_dump()), 405


# ============================================================================
# Application Startup/Shutdown
# ============================================================================

@app.before_request
def before_request():
    """Pre-request logging."""
    logger.debug(f"{request.method} {request.path}")


@app.after_request
def after_request(response):
    """Post-request logging."""
    if request.method != "OPTIONS":
        logger.debug(f"{request.method} {request.path} → {response.status_code}")
    return response


if __name__ == "__main__":
    import sys

    # Output directory setup
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load model on startup
    logger.info("=" * 70)
    logger.info("DualTaskCropHealthModel Biophysical Metrics API")
    logger.info("=" * 70)
    
    if not load_model():
        logger.error("Failed to load model. Check logs above.")
        sys.exit(1)

    logger.info(f"Starting API server on {API_HOST}:{API_PORT}")
    logger.info(f"API Version: {API_VERSION}")
    logger.info(f"Max upload size: {MAX_UPLOAD_MB}MB")
    logger.info("=" * 70)

    # Start Flask app
    app.run(
        host=API_HOST,
        port=API_PORT,
        debug=API_DEBUG,
        use_reloader=False,  # Disable reloader to avoid model being loaded twice
    )
