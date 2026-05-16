from dataclasses import dataclass, field
from typing import Dict, List
import math
import torch

from config import (
    MAX_CHL_UG_CM2, MAX_N_PERCENT, MAX_BIOMASS_MGHA, HEALTHY_BIOMASS_REF,
)


@dataclass
class AblationConfig:
    name:               str
    zero_sar:           bool = False
    zero_indices:       bool = False
    zero_nir:           bool = False
    zero_swir:          bool = False
    zero_red_edge:      bool = False
    disable_scad:       bool = False
    disable_pmfd:       bool = False
    disable_ssm:        bool = False
    disable_scatter:    bool = False
    disable_cross_attn: bool = False


@dataclass
class ModelOutput:
    chl_map:      torch.Tensor
    nitro_map:    torch.Tensor
    biomass_map:  torch.Tensor
    loss_map:     torch.Tensor
    variant_name: str = "Full model (baseline)"


@dataclass
class AblationOutput:
    variants: Dict[str, ModelOutput] = field(default_factory=dict)

    def delta_vs_baseline(self, key: str) -> Dict[str, float]:
        b = self.variants["Full model (baseline)"]
        v = self.variants[key]
        def _d(a, bb):
            return float((a - bb).abs().mean().item())
        return {
            "d_chl":     _d(v.chl_map,     b.chl_map),
            "d_nitro":   _d(v.nitro_map,   b.nitro_map),
            "d_biomass": _d(v.biomass_map,  b.biomass_map),
            "d_loss":    _d(v.loss_map,     b.loss_map),
        }

    def summary_table(self) -> str:
        BASE  = "Full model (baseline)"
        W     = 78
        lines = [
            f"\n{'━'*W}",
            f"  {'ABLATION STUDY  —  Mean Absolute Delta vs Baseline':^{W-4}}",
            f"{'━'*W}",
            f"  {'Variant':<32} {'Δ CHL':>8} {'Δ Nitro':>8} {'Δ Biomass':>10} {'Δ Loss':>8}",
            f"{'─'*W}",
        ]
        for name, out in self.variants.items():
            if name == BASE:
                lines.append(f"  {name:<32}  ← baseline")
                continue
            d = self.delta_vs_baseline(name)
            lines.append(
                f"  {name:<32} {d['d_chl']:>8.4f} {d['d_nitro']:>8.4f}"
                f" {d['d_biomass']:>10.4f} {d['d_loss']:>8.4f}"
            )
        lines += [
            f"{'━'*W}",
            f"  Δ = mean |pred_ablated − pred_baseline|  (0–1 scale)  higher = more important",
            f"{'━'*W}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        BASE   = "Full model (baseline)"
        result = {}
        for name in self.variants:
            if name == BASE:
                result[name] = {"delta": None, "note": "baseline"}
            else:
                result[name] = self.delta_vs_baseline(name)
        return result


ABLATION_VARIANTS: List[AblationConfig] = [
    AblationConfig("Full model (baseline)"),
    AblationConfig("-SAR input",           zero_sar=True),
    AblationConfig("-Spectral indices",    zero_indices=True),
    AblationConfig("-NIR proxy (B07)",     zero_nir=True),
    AblationConfig("-Red-edge (B05-B07)",  zero_swir=True),
    AblationConfig("-Red band (B04)",      zero_red_edge=True),
    AblationConfig("-SCAD decoder",        disable_scad=True),
    AblationConfig("-PMFD decoder",        disable_pmfd=True),
    AblationConfig("-SSM (Mamba gate)",    disable_ssm=True),
    AblationConfig("-Scatter decomp",      disable_scatter=True),
    AblationConfig("-Cross-attention",     disable_cross_attn=True),
]
