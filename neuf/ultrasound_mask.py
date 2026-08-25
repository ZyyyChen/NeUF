"""Mandatory valid-sector masking for ultrasound screen captures.

The anatomical fan is the largest temporally supported foreground component.
Text, dates, device settings, orientation labels, and the black canvas are
therefore excluded before they can become NeUF observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


SECTOR_MASK_VERSION = 1


@dataclass(frozen=True)
class SectorMaskDetection:
    mask: np.ndarray
    sampled_frames: int
    foreground_threshold: float
    valid_fraction: float
    source: str = "temporal-largest-component"


def _as_frame_stack(images: np.ndarray) -> np.ndarray:
    frames = np.asarray(images)
    if frames.ndim == 2:
        frames = frames[None, ...]
    if frames.ndim != 3:
        raise ValueError(
            f"Ultrasound mask input must have shape [N, H, W], got {frames.shape}"
        )
    if frames.shape[0] < 1 or frames.shape[1] < 2 or frames.shape[2] < 2:
        raise ValueError(f"Ultrasound mask input is too small: {frames.shape}")
    if not np.all(np.isfinite(frames)):
        raise ValueError("Ultrasound mask input contains non-finite values")
    return frames


def detect_ultrasound_sector_mask(
    images: np.ndarray,
    *,
    background_threshold: float | None = None,
    minimum_frame_frequency: float = 0.10,
    safety_margin_px: int = 2,
) -> SectorMaskDetection:
    """Detect one conservative, shared fan mask from representative frames.

    A shared mask prevents frame-varying anatomy or speckle from changing the
    spatial support.  The largest connected component rejects detached screen
    annotations; hole filling retains genuinely dark anatomy inside the fan.
    """
    frames = _as_frame_stack(images).astype(np.float32, copy=False)
    if not 0.0 < minimum_frame_frequency <= 1.0:
        raise ValueError("minimum_frame_frequency must be in (0, 1]")
    if safety_margin_px < 0:
        raise ValueError("safety_margin_px must be >= 0")

    value_max = float(frames.max())
    if background_threshold is None:
        background_threshold = 4.0 / 255.0 if value_max <= 1.0 else 4.0
    threshold = float(background_threshold)

    foreground_frequency = np.mean(frames > threshold, axis=0)
    candidate = foreground_frequency >= float(minimum_frame_frequency)
    if float(candidate.mean()) >= 0.95:
        mask = np.ones_like(candidate, dtype=bool)
        return SectorMaskDetection(
            mask=mask,
            sampled_frames=int(frames.shape[0]),
            foreground_threshold=threshold,
            valid_fraction=1.0,
            source="full-frame-acquisition",
        )
    candidate = ndimage.binary_opening(candidate, structure=np.ones((3, 3), dtype=bool))

    labels, component_count = ndimage.label(candidate)
    if component_count == 0:
        raise ValueError(
            "Could not detect an ultrasound acquisition sector: all sampled pixels "
            f"were below the background threshold {threshold:g}"
        )
    component_sizes = np.bincount(labels.reshape(-1))[1:]
    largest_label = int(np.argmax(component_sizes)) + 1
    mask = labels == largest_label
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5), dtype=bool))
    mask = ndimage.binary_fill_holes(mask)

    # A dense rectangular acquisition (for example a simulation) is already
    # entirely valid and should not lose its border to the conservative margin.
    if float(mask.mean()) >= 0.95:
        mask = np.ones_like(mask, dtype=bool)
        source = "full-frame-acquisition"
    else:
        if safety_margin_px:
            mask = ndimage.binary_erosion(mask, iterations=int(safety_margin_px))
        source = "temporal-largest-component"

    valid_fraction = float(mask.mean())
    if valid_fraction < 0.05:
        raise ValueError(
            "Detected ultrasound sector is implausibly small: "
            f"{valid_fraction:.2%} of the image"
        )
    return SectorMaskDetection(
        mask=np.asarray(mask, dtype=bool),
        sampled_frames=int(frames.shape[0]),
        foreground_threshold=threshold,
        valid_fraction=valid_fraction,
        source=source,
    )


def evenly_spaced_sample_indices(length: int, maximum_samples: int = 32) -> np.ndarray:
    if length < 1:
        raise ValueError("length must be >= 1")
    sample_count = min(int(maximum_samples), int(length))
    return np.unique(np.linspace(0, length - 1, sample_count, dtype=np.int64))
