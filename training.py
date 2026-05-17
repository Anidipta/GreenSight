import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    DEVICE, OUTPUT_DIR, CKPT_PATH, BEST_CKPT_PATH,
    EPOCHS, TRAIN_BATCH, LR_DECODERS, LR_BACKBONE,
    UNFREEZE_EPOCH, WEIGHT_DECAY, NUM_WORKERS, VAL_EVERY,
    GRAD_CLIP, W_CHL, W_NITRO, W_BIO, W_LOSS,
    DATA_DIR, VEG_NDVI_THRESH,
)
from model import DualTaskCropHealthModel, build_model
from data_loading import build_dataloaders
from utils import _section, _fmt


def biophysical_loss(
    chl_pred:   torch.Tensor,
    nitro_pred: torch.Tensor,
    bio_pred:   torch.Tensor,
    loss_pred:  torch.Tensor,
    gt:         torch.Tensor,
    veg_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    def _mse(p, g):
        diff = (p.squeeze(1) - g) ** 2
        if veg_weight is not None:
            diff = diff * veg_weight
        return diff.mean()
    chl_gt, nit_gt, bio_gt, los_gt = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
    return (
        W_CHL   * _mse(chl_pred,   chl_gt) +
        W_NITRO * _mse(nitro_pred, nit_gt) +
        W_BIO   * _mse(bio_pred,   bio_gt) +
        W_LOSS  * _mse(loss_pred,  los_gt)
    )


def _resize_preds(chl, nit, bio, los, target_h: int, target_w: int):
    def _r(t):
        if t.shape[-2] == target_h and t.shape[-1] == target_w:
            return t
        return F.interpolate(t, (target_h, target_w), mode="bilinear", align_corners=False)
    return _r(chl), _r(nit), _r(bio), _r(los)


def build_optimizer(model: DualTaskCropHealthModel, epoch: int) -> torch.optim.Optimizer:
    backbone_params = list(model.backbone.parameters())
    decoder_params  = list(model.scad.parameters()) + list(model.pmfd.parameters())
    if epoch < UNFREEZE_EPOCH:
        for p in backbone_params:
            p.requires_grad = False
        groups = [{"params": decoder_params, "lr": LR_DECODERS}]
    else:
        for p in backbone_params:
            p.requires_grad = True
        groups = [
            {"params": decoder_params,  "lr": LR_DECODERS},
            {"params": backbone_params, "lr": LR_BACKBONE},
        ]
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def run_epoch(
    model:     DualTaskCropHealthModel,
    loader:    DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device:    str,
    train:     bool,
) -> dict:
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for hls, sar, idx, gt in loader:
            hls, sar, idx, gt = (
                hls.to(device), sar.to(device), idx.to(device), gt.to(device)
            )
            chl, nit, bio, los = model(hls, sar, idx, ablation=False)
            H, W               = gt.shape[-2], gt.shape[-1]
            chl, nit, bio, los = _resize_preds(chl, nit, bio, los, H, W)
            ndvi               = idx[:, 0]
            veg_w              = (ndvi > VEG_NDVI_THRESH).float() * 1.5 + 0.5
            loss_val           = biophysical_loss(chl, nit, bio, los, gt, veg_w)
            if train and optimizer is not None:
                optimizer.zero_grad()
                loss_val.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            total_loss += loss_val.item()
            n_batches  += 1
    return {"loss": total_loss / max(n_batches, 1)}


def train(
    data_dir:    str  = DATA_DIR,
    output_dir:  str  = OUTPUT_DIR,
    epochs:      int  = EPOCHS,
    batch_size:  int  = TRAIN_BATCH,
    num_workers: int  = NUM_WORKERS,
    device:      str  = DEVICE,
    resume_ckpt: Optional[str] = None,
):
    _section(f"CROP HEALTH TRAINING  ·  device={device}")
    print(_fmt("Backbone  :", "Prithvi-EO-1.0-100M  (frozen → fine-tuned)"))
    print(_fmt("Decoders  :", "SCAD + PMFD"))
    print(_fmt("GT source :", "CIre→CHL  NDRE→N  NDVI→AGB  pseudo-labels"))
    print(_fmt("Epochs    :", epochs))
    print(_fmt("Batch     :", batch_size))
    print(_fmt("LR dec    :", LR_DECODERS))
    print(_fmt("LR bb     :", f"{LR_BACKBONE}  (active after epoch {UNFREEZE_EPOCH})"))
    print(_fmt("Output    :", output_dir))

    _section("BUILDING DATALOADERS")
    train_dl, val_dl = build_dataloaders(data_dir, batch_size, num_workers)

    _section("MODEL INIT")
    model = build_model(device)
    if resume_ckpt and os.path.exists(resume_ckpt):
        model.load_state_dict(torch.load(resume_ckpt, map_location=device))
        print(f"  Resumed from: {resume_ckpt}")
    model.freeze_backbone()

    os.makedirs(output_dir, exist_ok=True)
    ckpt_path      = os.path.join(output_dir, "trained_model.pt")
    best_ckpt_path = os.path.join(output_dir, "best_model.pt")
    best_val_loss  = float("inf")
    history        = []

    _section(f"TRAINING  ·  {epochs} epochs")
    print(f"  {'Ep':>4}  {'Phase':<5}  {'Loss':>10}  {'Status':<24}  {'Time':>6}")
    print(f"  {'─'*58}")

    for epoch in range(1, epochs + 1):
        frozen = epoch <= UNFREEZE_EPOCH
        if epoch == UNFREEZE_EPOCH + 1:
            print(f"\n  ── Epoch {epoch}: backbone unfrozen  lr={LR_BACKBONE}\n")
            model.unfreeze_backbone()

        optimizer = build_optimizer(model, epoch)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        t0        = time.time()
        tr_met    = run_epoch(model, train_dl, optimizer, device, train=True)
        t_train   = time.time() - t0
        status    = "backbone=frozen" if frozen else "backbone=active"
        print(f"  {epoch:>4}  train  loss={tr_met['loss']:.5f}  {status:<24}  {t_train:.1f}s")

        if epoch % VAL_EVERY == 0:
            t0      = time.time()
            va_met  = run_epoch(model, val_dl, None, device, train=False)
            t_val   = time.time() - t0
            print(f"  {epoch:>4}  val    loss={va_met['loss']:.5f}  {status:<24}  {t_val:.1f}s")
            if va_met["loss"] < best_val_loss:
                best_val_loss = va_met["loss"]
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"  ✓  Best model saved  (val_loss={best_val_loss:.5f})")
            history.append({
                "epoch":      epoch,
                "train_loss": tr_met["loss"],
                "val_loss":   va_met["loss"],
            })

        scheduler.step()

    torch.save(model.state_dict(), ckpt_path)

    _section("TRAINING COMPLETE")
    print(_fmt("Final checkpoint  :", ckpt_path))
    print(_fmt("Best checkpoint   :", best_ckpt_path))
    print(_fmt("Best val loss     :", f"{best_val_loss:.5f}"))
    if history:
        best_ep = min(history, key=lambda r: r["val_loss"])
        print(_fmt("Best epoch        :", best_ep["epoch"]))
    print(f"\n  Load for inference:")
    print(f"    from model import load_model")
    print(f"    model = load_model('{best_ckpt_path}')")
    print("━" * 72)


if __name__ == "__main__":
    train()
