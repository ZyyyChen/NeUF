from __future__ import annotations

import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRIMES = [1, 2654435761]


def hash2d(coords_int: torch.Tensor, log2_size: int) -> torch.Tensor:
    """Hash integer 2D grid coordinates into a power-of-two table."""
    return (
        coords_int[..., 0] * PRIMES[0] ^ coords_int[..., 1] * PRIMES[1]
    ) % (2**log2_size)


class HashEncoder2D(nn.Module):
    """2D hash grid encoder with bilinear interpolation over four corners."""

    def __init__(
        self,
        bbox_2d,
        n_levels: int,
        n_features_per_level: int,
        log2_hashmap_size: int,
        base_resolution: int,
        finest_resolution: int,
    ):
        super().__init__()

        if n_levels < 1:
            raise ValueError(f"n_levels must be >= 1, got {n_levels}")

        self.bbox_2d = bbox_2d
        self.n_levels = int(n_levels)
        self.n_features_per_level = int(n_features_per_level)
        self.n_feat = self.n_features_per_level
        self.log2_hashmap_size = int(log2_hashmap_size)
        self.log2_size = self.log2_hashmap_size
        self.out_dim = self.n_levels * self.n_features_per_level

        self.register_buffer("bbox_min", torch.tensor(bbox_2d[0], dtype=torch.float32))
        self.register_buffer("bbox_max", torch.tensor(bbox_2d[1], dtype=torch.float32))
        self.register_buffer("base_resolution", torch.tensor(float(base_resolution)))
        self.register_buffer("finest_resolution", torch.tensor(float(finest_resolution)))
        self.register_buffer(
            "offsets",
            torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.long),
        )

        if self.n_levels == 1:
            b = torch.tensor(1.0)
        else:
            b = torch.exp(
                (torch.log(self.finest_resolution) - torch.log(self.base_resolution))
                / (self.n_levels - 1)
            )
        self.register_buffer("b", b)
        self.base = self.base_resolution

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(2**self.log2_hashmap_size, self.n_features_per_level)
                for _ in range(self.n_levels)
            ]
        )
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, a=-0.0001, b=0.0001)

        self.to(DEVICE)

    def bilinear_interp(
        self,
        x_norm: torch.Tensor,
        corner_embedds: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate four corner features using local [0, 1)^2 coordinates."""
        wu = x_norm[:, 0:1]
        wv = x_norm[:, 1:2]

        return (
            (1 - wu) * (1 - wv) * corner_embedds[:, 0]
            + wu * (1 - wv) * corner_embedds[:, 1]
            + (1 - wu) * wv * corner_embedds[:, 2]
            + wu * wv * corner_embedds[:, 3]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        bbox_min = self.bbox_min.to(device=x.device, dtype=x.dtype)
        bbox_max = self.bbox_max.to(device=x.device, dtype=x.dtype)
        base_resolution = self.base_resolution.to(device=x.device, dtype=x.dtype)
        b = self.b.to(device=x.device, dtype=x.dtype)

        for i in range(self.n_levels):
            resolution = torch.floor(base_resolution * b**i)
            x_scaled = (x - bbox_min) / (bbox_max - bbox_min) * resolution
            x_floor = torch.floor(x_scaled).long()
            x_frac = x_scaled - x_floor.to(dtype=x.dtype)

            corners = x_floor.unsqueeze(1) + self.offsets.to(x.device).unsqueeze(0)
            indices = hash2d(corners, self.log2_hashmap_size)
            corner_embedds = self.embeddings[i](indices)
            feats.append(self.bilinear_interp(x_frac, corner_embedds))

        return torch.cat(feats, dim=-1)


class KroneckerHashPE(nn.Module):
    """Tri-plane hash-grid positional encoding for anisotropic ultrasound volumes."""

    def __init__(
        self,
        bounding_box,
        n_levels_lateral: int = 8,
        n_levels_axial: int = 8,
        finest_lateral: int = 128,
        finest_axial: int = 512,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        combine: str = "cat",
    ):
        super().__init__()

        combine = combine.lower()
        if combine not in {"cat", "sum", "product"}:
            raise ValueError(f"Unknown combine mode: {combine}")

        self.bounding_box = bounding_box
        self.combine = combine

        xmin, ymin, zmin = bounding_box[0]
        xmax, ymax, zmax = bounding_box[1]

        common = dict(
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
        )

        self.enc_xy = HashEncoder2D(
            bbox_2d=[[xmin, ymin], [xmax, ymax]],
            n_levels=n_levels_lateral,
            finest_resolution=finest_lateral,
            **common,
        )
        self.enc_xz = HashEncoder2D(
            bbox_2d=[[xmin, zmin], [xmax, zmax]],
            n_levels=n_levels_axial,
            finest_resolution=finest_axial,
            **common,
        )
        self.enc_yz = HashEncoder2D(
            bbox_2d=[[ymin, zmin], [ymax, zmax]],
            n_levels=n_levels_axial,
            finest_resolution=finest_axial,
            **common,
        )

        d_xy = self.enc_xy.out_dim
        d_xz = self.enc_xz.out_dim
        d_yz = self.enc_yz.out_dim

        if combine == "cat":
            self.out_dim = d_xy + d_xz + d_yz
        else:
            if d_xy != d_xz or d_xy != d_yz:
                raise ValueError(f"{combine} requires all three branch dimensions to match")
            self.out_dim = d_xy

        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_xy = self.enc_xy(x[:, [0, 1]])
        feat_xz = self.enc_xz(x[:, [0, 2]])
        feat_yz = self.enc_yz(x[:, [1, 2]])

        if self.combine == "cat":
            return torch.cat([feat_xy, feat_xz, feat_yz], dim=-1)
        if self.combine == "sum":
            return feat_xy + feat_xz + feat_yz
        if self.combine == "product":
            return feat_xy * feat_xz * feat_yz

        raise RuntimeError(f"Unsupported combine mode: {self.combine}")
