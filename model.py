import importlib.util
import json
import math
import os
import shutil
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    PRITHVI_REPO, PRITHVI_WEIGHTS, PRITHVI_CFG, PRITHVI_SRC,
    EMBED_DIM, DECODER_DIM, SAR_CHANNELS, SPECTRAL_QUERIES,
    SCATTER_COMPS, MAMBA_D_STATE, MAMBA_D_CONV,
    NUM_PATCHES, NUM_FRAMES, _PRITHVI_CFG_DROP, DEVICE, ABLATION,
)
from ablation import (
    AblationConfig, AblationOutput, ModelOutput, ABLATION_VARIANTS,
)


def _cbg(ic, oc, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(ic, oc, k, s, p, bias=False), nn.BatchNorm2d(oc), nn.GELU())

def _dcbg(ic, oc, k=4, s=2, p=1):
    return nn.Sequential(nn.ConvTranspose2d(ic, oc, k, s, p, bias=False), nn.BatchNorm2d(oc), nn.GELU())


class PrithviBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self._inner         = None
        self._strategy      = "none"
        self._err_automodel = None
        self._err_pt        = None
        self._try_automodel()
        if self._inner is None:
            self._try_pt_download()
        if self._inner is None:
            raise RuntimeError(
                "\n\nPrithvi-EO-1.0-100M failed to load via both strategies."
                f"\n  Strategy 1 — AutoModel :  {self._err_automodel}"
                f"\n  Strategy 2 — .pt direct:  {self._err_pt}"
                "\n\nCommon fixes:"
                "\n  1. pip install -U transformers huggingface_hub"
                "\n  2. Set HF_TOKEN in .env if the repo requires authentication"
                "\n  3. Check internet / proxy access to huggingface.co"
                "\n  4. Pre-download weights and set HF_HOME to local cache path"
                "\n  5. NumPy>=2.0 breaks Prithvi — run: pip install \"numpy<2\""
            )

    def _try_automodel(self):
        try:
            from transformers import AutoModel
            token = os.getenv("HF_TOKEN") or None
            print("  [Backbone] Trying AutoModel.from_pretrained …")
            m = AutoModel.from_pretrained(PRITHVI_REPO, trust_remote_code=True, token=token)
            self._inner    = m
            self._strategy = "automodel"
            print(f"  [Backbone] ✓ AutoModel  ({sum(p.numel() for p in m.parameters()):,} params)")
        except Exception as e:
            self._err_automodel = f"{type(e).__name__}: {e}"
            print(f"  [Backbone] ✗ AutoModel: {self._err_automodel}")

    def _try_pt_download(self):
        try:
            from huggingface_hub import hf_hub_download
            token = os.getenv("HF_TOKEN") or None
            print("  [Backbone] Trying hf_hub_download …")
            ckpt_path = hf_hub_download(PRITHVI_REPO, PRITHVI_WEIGHTS, token=token)
            cfg_path  = hf_hub_download(PRITHVI_REPO, PRITHVI_CFG,     token=token)
            src_path  = hf_hub_download(PRITHVI_REPO, PRITHVI_SRC,     token=token)
            try:
                local_src = Path(__file__).parent / "prithvi_mae.py"
            except NameError:
                local_src = Path(os.getcwd()) / "prithvi_mae.py"
            if not local_src.exists():
                shutil.copy(src_path, local_src)
            import numpy as _np
            _nv = tuple(int(x) for x in _np.__version__.split(".")[:2])
            if _nv >= (2, 0):
                print(
                    f"  [Backbone] ⚠  NumPy {_np.__version__} detected. "
                    "Prithvi requires NumPy <2.0.  "
                    "Run: pip install \"numpy<2\"  then restart."
                )

            spec = importlib.util.spec_from_file_location("prithvi_mae", str(local_src))
            mod  = importlib.util.module_from_spec(spec)
            mod.np = _np
            for _attr in ("bool", "int", "float", "complex", "object", "str"):
                if not hasattr(_np, _attr):
                    setattr(_np, _attr, getattr(__builtins__, _attr, None))
            spec.loader.exec_module(mod)
            with open(cfg_path) as f:
                raw_cfg = json.load(f)
            pcfg    = raw_cfg.get("pretrained_cfg", raw_cfg)
            init_kw = {k: v for k, v in pcfg.items() if k not in _PRITHVI_CFG_DROP}
            init_kw["num_frames"] = NUM_FRAMES
            model, cls_errors = None, []
            for cls_name in ("PrithviViT", "PrithviMAE"):
                if hasattr(mod, cls_name):
                    try:
                        model = getattr(mod, cls_name)(**init_kw)
                        print(f"  [Backbone] Instantiated {cls_name}")
                        break
                    except Exception as ce:
                        cls_errors.append(f"{cls_name}: {ce}")
                        print(f"  [Backbone] {cls_name} failed: {ce}")
            if model is None:
                available = [x for x in dir(mod) if not x.startswith("_")]
                raise RuntimeError(
                    f"Neither PrithviViT nor PrithviMAE could be instantiated. "
                    f"Errors: {cls_errors}. "
                    f"Available in prithvi_mae.py: {available}"
                )
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            state       = {k: v for k, v in state.items() if "pos_embed" not in k}
            miss, unexp = model.load_state_dict(state, strict=False)
            model.eval()
            self._inner    = model
            self._strategy = "pt_direct"
            n = sum(p.numel() for p in model.parameters())
            print(f"  [Backbone] ✓ {type(model).__name__}  ({n:,} params)  "
                  f"missing={len(miss)}  unexpected={len(unexp)}")
        except Exception as e:
            self._err_pt = f"{type(e).__name__}: {e}"
            print(f"  [Backbone] ✗ .pt load: {self._err_pt}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self._strategy == "automodel":
            with torch.no_grad():
                out = self._inner(pixel_values=x)
            hs = getattr(out, "last_hidden_state", None) or out[0]
            return hs[:, 1:, :] if hs.shape[1] == NUM_PATCHES + 1 else hs
        if self._strategy == "pt_direct":
            x5 = x.unsqueeze(2)
            with torch.no_grad():
                if hasattr(self._inner, "forward_encoder"):
                    latent, _, _ = self._inner.forward_encoder(x5, mask_ratio=0.0)
                else:
                    out    = self._inner(x5)
                    latent = out[0] if isinstance(out, (tuple, list)) else out
            return latent[:, 1:, :]
        raise RuntimeError(f"Unknown backbone strategy: {self._strategy}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class BandRatioQueryGenerator(nn.Module):
    def __init__(self, ni=8, nq=SPECTRAL_QUERIES, d=EMBED_DIM):
        super().__init__()
        self.proj = nn.Linear(ni, d)
        self.gen  = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, nq * d))
        self.nq, self.d = nq, d

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        pooled = idx.mean(dim=[2, 3])
        return self.gen(self.proj(pooled)).view(pooled.shape[0], self.nq, self.d)


class SpectralCrossAttentionDecoder(nn.Module):
    def __init__(self, d=EMBED_DIM, dd=DECODER_DIM, nq=SPECTRAL_QUERIES):
        super().__init__()
        self.qgen    = BandRatioQueryGenerator(8, nq, d)
        self.xattn   = nn.MultiheadAttention(d, 8, batch_first=True)
        self.qnorm   = nn.LayerNorm(d)
        self.tnorm   = nn.LayerNorm(d)
        self.to_feat = nn.Linear(d, dd)
        self.up1 = _dcbg(dd,    dd);     self.r1 = _cbg(dd,    dd)
        self.up2 = _dcbg(dd,    dd//2);  self.r2 = _cbg(dd//2, dd//2)
        self.up3 = _dcbg(dd//2, dd//4);  self.r3 = _cbg(dd//4, dd//4)
        self.up4 = _dcbg(dd//4, dd//8);  self.r4 = _cbg(dd//8, dd//8)
        fc = dd // 8
        self.chl_head   = nn.Sequential(_cbg(fc, 32), nn.Conv2d(32, 1, 1), nn.Sigmoid())
        self.nitro_head = nn.Sequential(_cbg(fc, 32), nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def forward(
        self,
        tokens:             torch.Tensor,
        indices:            torch.Tensor,
        disable_cross_attn: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = tokens.shape
        side    = int(math.sqrt(N))
        tn      = self.tnorm(tokens)
        if not disable_cross_attn:
            q     = self.qnorm(self.qgen(indices))
            tn, _ = self.xattn(q, tn, tn)
            tn    = self.tnorm(tokens + F.linear(
                tn.mean(1, keepdim=True).expand_as(tokens),
                torch.eye(D, device=tokens.device),
            ))
        feat = self.to_feat(tn).transpose(1, 2).reshape(B, -1, side, side)
        feat = self.r1(self.up1(feat))
        feat = self.r2(self.up2(feat))
        feat = self.r3(self.up3(feat))
        feat = self.r4(self.up4(feat))
        return self.chl_head(feat), self.nitro_head(feat)


class ScatteringDecomposition(nn.Module):
    def __init__(self, ic=SAR_CHANNELS, oc=SCATTER_COMPS):
        super().__init__()
        self.net = nn.Sequential(_cbg(ic, 64), _cbg(64, 64),
                                 nn.Conv2d(64, oc, 1), nn.Softmax(dim=1))
        self.oc  = oc

    def forward(self, sar: torch.Tensor, disable_scatter: bool = False) -> torch.Tensor:
        if disable_scatter:
            B, _, H, W = sar.shape
            return torch.full((B, self.oc, H, W), 1.0 / self.oc,
                              device=sar.device, dtype=sar.dtype)
        return self.net(sar)


class SelectiveStateScan(nn.Module):
    def __init__(self, d=EMBED_DIM, S=MAMBA_D_STATE, dc=MAMBA_D_CONV):
        super().__init__()
        self.d, self.S = d, S
        self.in_proj   = nn.Linear(d, d * 2)
        self.conv1d    = nn.Conv1d(d, d, dc, padding=dc - 1, groups=d)
        self.x_proj    = nn.Linear(d, S + S + 1)
        self.dt_proj   = nn.Linear(1, S)
        self.A_log     = nn.Parameter(torch.log(torch.arange(1, S + 1).float()))
        self.D         = nn.Parameter(torch.ones(d))
        self.out_proj  = nn.Linear(d, d)
        self.ratio_gate= nn.Sequential(nn.Linear(1, d), nn.Sigmoid())
        self.bypass    = nn.Linear(d, d)

    def forward(
        self,
        tokens:      torch.Tensor,
        sar_ratio:   Optional[torch.Tensor] = None,
        disable_ssm: bool = False,
    ) -> torch.Tensor:
        if disable_ssm:
            return self.bypass(tokens)
        B, N, D = tokens.shape
        S       = self.S
        xz      = self.in_proj(tokens)
        x, z    = xz.chunk(2, dim=-1)
        xc      = self.conv1d(x.transpose(1, 2))[:, :, :N].transpose(1, 2)
        xc      = F.silu(xc)
        dBC     = self.x_proj(xc)
        dtr, Bm, Cm = dBC.split([1, S, S], dim=-1)
        dt      = F.softplus(self.dt_proj(dtr))
        A       = -torch.exp(self.A_log.float())
        h       = torch.zeros(B, S, D, device=tokens.device, dtype=tokens.dtype)
        ys      = []
        for i in range(N):
            dA  = torch.exp(dt[:, i, :] * A.unsqueeze(0))
            dBu = (dt[:, i, :].unsqueeze(-1) *
                   Bm[:, i, :].unsqueeze(-1) *
                   xc[:, i, :].unsqueeze(1))
            h   = h * dA.unsqueeze(-1) + dBu
            ys.append((h * Cm[:, i, :].unsqueeze(-1)).sum(1))
        y = torch.stack(ys, dim=1) + xc * self.D.unsqueeze(0).unsqueeze(0)
        if sar_ratio is not None:
            y = y * self.ratio_gate(sar_ratio.unsqueeze(-1))
        return self.out_proj(y * F.silu(z))


class PolarimetricFusionDecoder(nn.Module):
    def __init__(self, d=EMBED_DIM, dd=DECODER_DIM, sc=SAR_CHANNELS):
        super().__init__()
        self.sar_enc = nn.Sequential(_cbg(sc, 64), _cbg(64, 128), _cbg(128, dd))
        self.scatter = ScatteringDecomposition(sc, SCATTER_COMPS)
        self.sc_fuse = _cbg(SCATTER_COMPS, dd // 4)
        self.ssm     = SelectiveStateScan(d, MAMBA_D_STATE, MAMBA_D_CONV)
        self.to_feat = nn.Linear(d, dd)
        self.xattn   = nn.MultiheadAttention(dd, 8, batch_first=True)
        self.xnorm   = nn.LayerNorm(dd)
        self.up1 = _dcbg(dd + dd // 4, dd);  self.r1 = _cbg(dd,    dd)
        self.up2 = _dcbg(dd,           dd // 2); self.r2 = _cbg(dd//2, dd//2)
        self.up3 = _dcbg(dd // 2,      dd // 4); self.r3 = _cbg(dd//4, dd//4)
        self.up4 = _dcbg(dd // 4,      dd // 8); self.r4 = _cbg(dd//8, dd//8)
        fc = dd // 8
        self.bio_head  = nn.Sequential(_cbg(fc, 32), nn.Conv2d(32, 1, 1), nn.Sigmoid())
        self.loss_head = nn.Sequential(_cbg(fc, 32), nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def _sar_ratio_per_patch(self, sar_maps: torch.Tensor, side: int) -> torch.Tensor:
        p = F.adaptive_avg_pool2d(sar_maps, (side, side)).flatten(2).transpose(1, 2)
        return p[:, :, 0] / (p[:, :, 1].abs() + 1e-8)

    def forward(
        self,
        tokens:          torch.Tensor,
        sar:             torch.Tensor,
        disable_ssm:     bool = False,
        disable_scatter: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B    = tokens.shape[0]
        side = int(math.sqrt(tokens.shape[1]))
        sm   = self.sar_enc(sar)
        sc   = self.scatter(sar, disable_scatter=disable_scatter)
        scf  = self.sc_fuse(sc)
        rat  = self._sar_ratio_per_patch(sm, side)
        sca  = self.ssm(tokens, rat, disable_ssm=disable_ssm)
        tf   = self.to_feat(sca)
        ss   = F.adaptive_avg_pool2d(sm, (side, side)).flatten(2).transpose(1, 2)
        fn   = self.xnorm(tf)
        fus, _ = self.xattn(fn, self.xnorm(ss), self.xnorm(ss))
        feat   = fus.transpose(1, 2).reshape(B, -1, side, side)
        scd    = F.interpolate(scf, (side, side), mode="bilinear", align_corners=False)
        feat   = torch.cat([feat, scd], dim=1)
        feat   = self.r1(self.up1(feat))
        feat   = self.r2(self.up2(feat))
        feat   = self.r3(self.up3(feat))
        feat   = self.r4(self.up4(feat))
        return self.bio_head(feat), self.loss_head(feat)


class DualTaskCropHealthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = PrithviBackbone()
        self.scad     = SpectralCrossAttentionDecoder(EMBED_DIM, DECODER_DIM, SPECTRAL_QUERIES)
        self.pmfd     = PolarimetricFusionDecoder(EMBED_DIM, DECODER_DIM, SAR_CHANNELS)

    def _ablate_inputs(self, hls, sar, idx, cfg: AblationConfig):
        h, s, x = hls.clone(), sar.clone(), idx.clone()
        if cfg.zero_sar:       s.zero_()
        if cfg.zero_indices:   x.zero_()
        if cfg.zero_nir:       h[:, 4].zero_(); h[:, 5].zero_()
        if cfg.zero_swir:      h[:, 3].zero_(); h[:, 4].zero_()
        if cfg.zero_red_edge:  h[:, 2].zero_()
        return h, s, x

    def _run_variant(self, hls, sar, idx, cfg: AblationConfig) -> ModelOutput:
        B       = hls.shape[0]
        h, s, x = self._ablate_inputs(hls, sar, idx, cfg)
        tokens  = self.backbone(h)
        S_out   = int(math.sqrt(tokens.shape[1])) * (2 ** 4)
        if cfg.disable_scad:
            chl = nitro = torch.zeros(B, 1, S_out, S_out, device=h.device, dtype=h.dtype)
        else:
            chl, nitro = self.scad(tokens, x, disable_cross_attn=cfg.disable_cross_attn)
        if cfg.disable_pmfd:
            bio = loss = torch.zeros(B, 1, S_out, S_out, device=h.device, dtype=h.dtype)
        else:
            bio, loss = self.pmfd(tokens, s,
                                  disable_ssm=cfg.disable_ssm,
                                  disable_scatter=cfg.disable_scatter)
        return ModelOutput(chl_map=chl, nitro_map=nitro,
                           biomass_map=bio, loss_map=loss,
                           variant_name=cfg.name)

    def forward(self, hls, sar, indices, ablation: bool = ABLATION):
        if not ablation:
            out = self._run_variant(hls, sar, indices, AblationConfig("Full model (baseline)"))
            return out.chl_map, out.nitro_map, out.biomass_map, out.loss_map
        result = AblationOutput()
        for vcfg in ABLATION_VARIANTS:
            with torch.no_grad():
                out = self._run_variant(hls, sar, indices, vcfg)
            result.variants[vcfg.name] = out
        return result

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    @property
    def param_counts(self):
        return {
            "total":    sum(p.numel() for p in self.parameters()),
            "backbone": sum(p.numel() for p in self.backbone.parameters()),
            "scad":     sum(p.numel() for p in self.scad.parameters()),
            "pmfd":     sum(p.numel() for p in self.pmfd.parameters()),
        }


def build_model(device: str = DEVICE) -> DualTaskCropHealthModel:
    return DualTaskCropHealthModel().to(device).eval()


def load_model(ckpt_path: str, device: str = DEVICE) -> DualTaskCropHealthModel:
    model = build_model(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model