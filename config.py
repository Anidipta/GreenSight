import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PRITHVI_REPO    = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M"
PRITHVI_WEIGHTS = "Prithvi_EO_V1_100M.pt"
PRITHVI_CFG     = "config.json"
PRITHVI_SRC     = "prithvi_mae.py"

EMBED_DIM        = 768
PATCH_SIZE       = 16
TILE_SIZE        = 224
NUM_PATCHES      = (TILE_SIZE // PATCH_SIZE) ** 2
HLS_CHANNELS     = 6
SAR_CHANNELS     = 4
SPECTRAL_QUERIES = 8
DECODER_DIM      = 256
MAMBA_D_STATE    = 16
MAMBA_D_CONV     = 4
SCATTER_COMPS    = 3
NUM_FRAMES       = 1

DATASET_REPO       = "ibm-nasa-geospatial/multi-temporal-crop-classification"
TRAIN_ARCHIVE      = "training_chips.tgz"
VAL_ARCHIVE        = "validation_chips.tgz"
NUM_TEST_CHIPS     = 6

HLS_MEAN = np.array([
    775.2290211032589,
    1080.992780391705,
    1228.5855250417867,
    2497.2022620507532,
    2204.2139147975554,
    1610.8324823273745,
], dtype=np.float32)

HLS_STD = np.array([
    1281.526139861424,
    1270.0297974547493,
    1399.4802505642526,
    1368.3446143747644,
    1291.6764008585435,
    1154.505683480695,
], dtype=np.float32)

TILE_STRIDE = 196
BATCH_SIZE  = 4

MAX_CHL_UG_CM2      = 80.0
MAX_N_PERCENT       = 4.5
MAX_BIOMASS_MGHA    = 250.0
HEALTHY_BIOMASS_REF = 80.0

VEG_NDVI_THRESH     = 0.20
STRESS_THRESHOLD    = 0.35
BIOMASS_LOSS_THRESH = 0.25

DATA_DIR   = "./data"
OUTPUT_DIR = "./output"
CKPT_PATH      = f"{OUTPUT_DIR}/trained_model.pt"
BEST_CKPT_PATH = f"{OUTPUT_DIR}/best_model.pt"

EPOCHS         = 30
TRAIN_BATCH    = 8
LR_DECODERS    = 3e-4
LR_BACKBONE    = 3e-5
UNFREEZE_EPOCH = 5
WEIGHT_DECAY   = 1e-4
NUM_WORKERS    = 2
VAL_EVERY      = 1
GRAD_CLIP      = 1.0
W_CHL          = 1.0
W_NITRO        = 1.0
W_BIO          = 1.0
W_LOSS         = 0.8

ABLATION = True

_PRITHVI_CFG_DROP = frozenset({"mask_ratio", "bands", "mean", "std", "origin_url", "paper_ids"})

CDL_CLASSES = {
    0: "Background",   1: "Corn",        2: "Cotton",
    3: "Rice",         4: "Sorghum",     5: "Soybeans",
    6: "Sunflower",    7: "Peanuts",     8: "Tobacco",
    9: "Sweet Corn",  10: "Pop. Corn",  11: "Mint",
   12: "Winter Wheat",
}

API_HOST    = "0.0.0.0"
API_PORT    = 5000
API_DEBUG   = False
API_VERSION = "v1"
MAX_UPLOAD_MB = 512
