"""Strict Ultra-NeRF ultrasound physics renderer.

The tensor layout in this module intentionally follows the official
TensorFlow implementation: ``[W, H]``, where every row is one lateral
A-line and ``H`` is ordered from shallow to deep.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


REFERENCE_ULTRA_NERF_COMMIT = "4271e03539b4537be6db36b00e04654ceb781217"


def exclusive_cumprod(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Cumulative product excluding the element at the current index."""
    if x.shape[dim] < 1:
        raise ValueError("exclusive_cumprod requires a non-empty dimension")
    inclusive = torch.cumprod(x, dim=dim)
    head_shape = list(x.shape)
    head_shape[dim] = 1
    head = torch.ones(head_shape, dtype=x.dtype, device=x.device)
    return torch.cat(
        [head, inclusive.narrow(dim, 0, x.shape[dim] - 1)],
        dim=dim,
    )


def gaussian_psf_kernel(
    half_size: int = 3,
    lateral_std: float = 2.0,
    axial_std: float = 1.0,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return a normalized ``[1, 1, lateral, axial]`` Gaussian PSF."""
    if half_size < 0:
        raise ValueError(f"half_size must be >= 0, got {half_size}")
    if lateral_std <= 0 or axial_std <= 0:
        raise ValueError(
            "PSF standard deviations must be positive, got "
            f"lateral={lateral_std}, axial={axial_std}"
        )

    coordinates = torch.arange(
        -half_size,
        half_size + 1,
        dtype=dtype,
        device=device,
    )
    lateral = torch.exp(-0.5 * (coordinates / lateral_std) ** 2)
    axial = torch.exp(-0.5 * (coordinates / axial_std) ** 2)
    kernel = lateral[:, None] * axial[None, :]
    kernel = kernel / kernel.sum()
    return kernel[None, None]


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device.type if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    return generator


class UltraNeRFRenderer(nn.Module):
    """Official-code-aligned stochastic ultrasound renderer.

    ``raw_wh5`` is the five-parameter field output and ``z_vals_wh`` is the
    physical depth coordinate. The renderer converts millimetres to metres
    before applying the attenuation formula when requested.
    """

    # Public debug metadata used by read-only checkpoint diagnostics.  Keeping
    # the order beside the renderer prevents a diagnostic from silently
    # applying a stale interpretation to the five learned output channels.
    RAW_CHANNEL_NAMES = (
        "alpha_raw",
        "beta_raw",
        "rho_b_raw",
        "rho_s_raw",
        "phi_raw",
    )

    PARAMETER_MAP_NAMES = (
        "attenuation_coeff",
        "reflection_coeff",
        "border_probability",
        "border_indicator",
        "attenuation_transmission",
        "reflection_transmission",
        "scatterers_density_coeff",
        "scatterers_density",
        "scatter_amplitude",
        "psf_border",
        "psf_scatter",
        "transmission",
        "r",
        "b",
        "intensity_map",
    )

    def __init__(
        self,
        *,
        psf_half_size: int = 3,
        psf_lateral_std: float = 2.0,
        psf_axial_std: float = 1.0,
        distance_unit: str = "m",
    ) -> None:
        super().__init__()
        distance_unit = str(distance_unit).lower()
        if distance_unit not in {"m", "mm"}:
            raise ValueError(f"distance_unit must be 'm' or 'mm', got {distance_unit}")
        self.psf_half_size = int(psf_half_size)
        self.psf_lateral_std = float(psf_lateral_std)
        self.psf_axial_std = float(psf_axial_std)
        self.distance_unit = distance_unit
        self.register_buffer(
            "psf_kernel",
            gaussian_psf_kernel(
                self.psf_half_size,
                self.psf_lateral_std,
                self.psf_axial_std,
            ),
        )

    @staticmethod
    def _validate_indicator(
        indicator: torch.Tensor,
        reference: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if indicator.shape != reference.shape:
            raise ValueError(
                f"{name} must have shape {tuple(reference.shape)}, "
                f"got {tuple(indicator.shape)}"
            )
        indicator = indicator.to(dtype=reference.dtype, device=reference.device)
        if not torch.all((indicator == 0) | (indicator == 1)):
            raise ValueError(f"{name} must contain only zeros and ones")
        return indicator.detach()

    def forward(
        self,
        raw_wh5: torch.Tensor,
        z_vals_wh: torch.Tensor,
        *,
        border_generator: Optional[torch.Generator] = None,
        scatter_generator: Optional[torch.Generator] = None,
        border_indicator: Optional[torch.Tensor] = None,
        scatterers_density: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if raw_wh5.ndim != 3 or raw_wh5.shape[-1] != 5:
            raise ValueError(
                f"raw_wh5 must have shape [W, H, 5], got {tuple(raw_wh5.shape)}"
            )
        if z_vals_wh.shape != raw_wh5.shape[:2]:
            raise ValueError(
                "z_vals_wh must match raw_wh5[:2], got "
                f"{tuple(z_vals_wh.shape)} and {tuple(raw_wh5.shape[:2])}"
            )
        if raw_wh5.shape[1] < 2:
            raise ValueError("Ultra-NeRF rendering requires at least two axial samples")
        if not torch.is_floating_point(raw_wh5):
            raise TypeError("raw_wh5 must be a floating-point tensor")

        z_vals_wh = z_vals_wh.to(dtype=raw_wh5.dtype, device=raw_wh5.device)
        if not torch.all(torch.isfinite(raw_wh5)) or not torch.all(
            torch.isfinite(z_vals_wh)
        ):
            raise ValueError("Ultra-NeRF renderer inputs must be finite")

        alpha = torch.abs(raw_wh5[..., 0])
        beta = torch.sigmoid(raw_wh5[..., 1])
        rho_b = torch.sigmoid(raw_wh5[..., 2])
        rho_s = torch.sigmoid(raw_wh5[..., 3])
        phi = torch.sigmoid(raw_wh5[..., 4])
        if valid_mask is None:
            valid_weights = torch.ones_like(alpha)
        else:
            valid_weights = self._validate_indicator(
                valid_mask,
                alpha,
                "valid_mask",
            )
            alpha = alpha * valid_weights
            beta = beta * valid_weights
            rho_b = rho_b * valid_weights
            rho_s = rho_s * valid_weights
            phi = phi * valid_weights

        dists = torch.abs(z_vals_wh[:, 1:] - z_vals_wh[:, :-1])
        dists = torch.cat([dists, dists[:, -1:]], dim=1)
        if self.distance_unit == "mm":
            dists = dists * 1e-3

        attenuation_step = torch.exp(-alpha * dists)
        attenuation_transmission = exclusive_cumprod(attenuation_step, dim=1)

        if border_indicator is None:
            border_indicator = torch.bernoulli(
                rho_b,
                generator=border_generator,
            ).detach()
        else:
            border_indicator = self._validate_indicator(
                border_indicator,
                rho_b,
                "border_indicator",
            )

        reflection_step = 1.0 - beta * border_indicator
        reflection_transmission = exclusive_cumprod(reflection_step, dim=1)

        if scatterers_density is None:
            scatterers_density = torch.bernoulli(
                rho_s,
                generator=scatter_generator,
            ).detach()
        else:
            scatterers_density = self._validate_indicator(
                scatterers_density,
                rho_s,
                "scatterers_density",
            )

        kernel = self.psf_kernel.to(dtype=raw_wh5.dtype, device=raw_wh5.device)
        psf_border = F.conv2d(
            border_indicator[None, None],
            kernel,
            padding=self.psf_half_size,
        )[0, 0]
        scatterers_map = scatterers_density * phi
        psf_scatter = F.conv2d(
            scatterers_map[None, None],
            kernel,
            padding=self.psf_half_size,
        )[0, 0]

        transmission = attenuation_transmission * reflection_transmission
        b = transmission * psf_scatter
        r = transmission * beta * psf_border
        intensity_map = b + r
        maps = {
            "intensity_map": intensity_map,
            "attenuation_coeff": alpha,
            "reflection_coeff": beta,
            "border_probability": rho_b,
            "border_indicator": border_indicator,
            "attenuation_transmission": attenuation_transmission,
            "reflection_transmission": reflection_transmission,
            "scatterers_density_coeff": rho_s,
            "scatterers_density": scatterers_density,
            "scatter_amplitude": phi,
            "psf_border": psf_border,
            "psf_scatter": psf_scatter,
            "b": b,
            "r": r,
            "transmission": transmission,
        }
        if valid_mask is not None:
            maps = {name: value * valid_weights for name, value in maps.items()}
        return maps
