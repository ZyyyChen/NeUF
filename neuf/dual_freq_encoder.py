from __future__ import annotations

import torch
import torch.nn as nn

from neuf.hash_encoder import HashEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GateNet(nn.Module):
    """Spatial gate that predicts the high-frequency mixing weight."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat_low: torch.Tensor) -> torch.Tensor:
        return self.net(feat_low)


class DualFreqEncoder(nn.Module):
    """Retained low/high HashGrid encoder for the current Phase 1 model."""

    def __init__(
        self,
        bounding_box=None,
        n_levels_low: int = 8,
        n_levels_high: int = 8,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution_low: int = 16,
        finest_resolution_low: int = 64,
        base_resolution_high: int = 64,
        finest_resolution_high: int = 512,
        use_gate: bool = True,
        gate_hidden: int = 64,
        hf_activate_ratio: float = 0.2,
        hf_max_weight: float = 1.0,
    ):
        super().__init__()

        self.use_gate = bool(use_gate)
        self.hf_activate_ratio = float(hf_activate_ratio)
        self.hf_max_weight = float(hf_max_weight)

        if bounding_box is None:
            raise ValueError("DUAL_HASH requires a bounding_box")
        self.enc_low = HashEncoder(
            bounding_box=bounding_box,
            n_levels=n_levels_low,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution_low,
            finest_resolution=finest_resolution_low,
        )
        self.enc_high = HashEncoder(
            bounding_box=bounding_box,
            n_levels=n_levels_high,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution_high,
            finest_resolution=finest_resolution_high,
        )

        self.out_dim_low = self.enc_low.out_dim
        self.out_dim_high = self.enc_high.out_dim
        self.out_dim = self.out_dim_low + self.out_dim_high

        self.gate_net = GateNet(self.out_dim_low, hidden=gate_hidden) if self.use_gate else None
        self.to(DEVICE)

    def global_hf_weight(
        self,
        progress: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        k = 10.0
        raw = torch.as_tensor(
            k * (float(progress) - self.hf_activate_ratio),
            device=device,
            dtype=dtype,
        )
        return self.hf_max_weight * torch.sigmoid(raw)

    def forward_decomposed(self, x: torch.Tensor, progress: float = 1.0) -> dict[str, torch.Tensor]:
        feat_low = self.enc_low(x)
        feat_high = self.enc_high(x)

        if self.gate_net is not None:
            gate = self.gate_net(feat_low)
        else:
            gate = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)

        hf_weight = self.global_hf_weight(progress, device=x.device, dtype=x.dtype)
        weighted_high = hf_weight * gate * feat_high
        combined_raw = torch.cat([feat_low, feat_high], dim=-1)
        combined = torch.cat([feat_low, weighted_high], dim=-1)

        return {
            "feat_low": feat_low,
            "feat_high": feat_high,
            "gate": gate,
            "hf_weight": hf_weight,
            "weighted_high": weighted_high,
            "combined_raw": combined_raw,
            "combined": combined,
        }

    def forward(self, x: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        return self.forward_decomposed(x, progress)["combined"]

    def parameters_low(self):
        return list(self.enc_low.parameters())

    def parameters_high(self):
        params = list(self.enc_high.parameters())
        if self.gate_net is not None:
            params += list(self.gate_net.parameters())
        return params
