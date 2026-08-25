from __future__ import annotations

import numpy as np
import torch

from neuf.nerf_network import NeRF
from neuf.slice_renderer_base import DEVICE
from neuf.slice_renderer_ultra_nerf import UltraNeRFSliceRenderer


class _DatasetStub:
    def __init__(self, height: int, width: int) -> None:
        self.px_height = height
        self.px_width = width
        self.height = float(height)
        self.width = float(width)
        self.point_min = np.array([0.0, -4.0, -1.0], dtype=np.float32)
        self.point_max = np.array([20.0, 4.0, 1.0], dtype=np.float32)
        self.point_min_dev = torch.as_tensor(
            self.point_min,
            dtype=torch.float32,
            device=DEVICE,
        )


def _geometry(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    depth = torch.arange(height, dtype=torch.float32, device=DEVICE)
    lateral = torch.arange(width, dtype=torch.float32, device=DEVICE) - width / 2
    depth_grid, lateral_grid = torch.meshgrid(depth, lateral, indexing="ij")
    points = torch.stack(
        [depth_grid, lateral_grid, 0.1 * lateral_grid],
        dim=-1,
    )
    viewdirs = torch.zeros_like(points)
    viewdirs[..., 0] = -1.0
    return points, viewdirs


def _model() -> NeRF:
    torch.manual_seed(11)
    model = NeRF(output_mode="ultra_nerf")
    model.init_base_encoding(use_directions=False, use_encoding=False)
    model.init_model(D=2, W=16)
    return model.to(DEVICE)


def test_network_has_independent_five_parameter_head_and_round_trips() -> None:
    model = _model()
    points = torch.randn(7, 3, device=DEVICE)
    viewdirs = torch.zeros_like(points)
    raw = model.query(points, viewdirs, netchunk=3)
    assert raw.shape == (7, 5)
    assert model.sigma_linear is None

    checkpoint = model.get_save_dict()
    restored = NeRF(checkpoint)
    restored_raw = restored.query(points, viewdirs, netchunk=2)
    torch.testing.assert_close(raw, restored_raw)


def test_default_ultra_head_initialization_survives_a_704_sample_aline() -> None:
    model = _model()
    raw_bias = model.output_linear.bias.detach().cpu()
    torch.testing.assert_close(raw_bias[0], torch.tensor(1.0))
    torch.testing.assert_close(torch.sigmoid(raw_bias[1]), torch.tensor(0.02))
    torch.testing.assert_close(torch.sigmoid(raw_bias[2]), torch.tensor(0.005))
    torch.testing.assert_close(torch.sigmoid(raw_bias[3]), torch.tensor(0.2))
    torch.testing.assert_close(torch.sigmoid(raw_bias[4]), torch.tensor(0.5))

    height, width = 704, 8
    dataset = _DatasetStub(height, width)
    points, viewdirs = _geometry(height, width)
    renderer = UltraNeRFSliceRenderer(dataset, bernoulli_seed=41, query_chunk=2048)
    maps = renderer.render_points(
        model,
        points,
        viewdirs,
        height=height,
        width=width,
        eval_seed=41,
    )
    assert maps["reflection_transmission"][-1].mean() > 0.8
    assert maps["intensity_map"].mean() > 0.05
    assert maps["intensity_map"][-1].mean() > 0.01


def test_full_frame_layout_and_query_chunk_do_not_change_result() -> None:
    height, width = 9, 5
    dataset = _DatasetStub(height, width)
    points, viewdirs = _geometry(height, width)
    model = _model()
    small_chunks = UltraNeRFSliceRenderer(
        dataset,
        bernoulli_seed=31,
        query_chunk=7,
    )
    large_chunks = UltraNeRFSliceRenderer(
        dataset,
        bernoulli_seed=31,
        query_chunk=10_000,
    )

    small = small_chunks.render_points(
        model,
        points,
        viewdirs,
        height=height,
        width=width,
        eval_seed=101,
    )
    large = large_chunks.render_points(
        model,
        points,
        viewdirs,
        height=height,
        width=width,
        eval_seed=101,
    )
    assert small["intensity_map"].shape == (height, width)
    for name in small:
        assert small[name].shape == (height, width)
        if name in {"border_indicator", "scatterers_density"}:
            torch.testing.assert_close(small[name], large[name], rtol=0, atol=0)
        else:
            torch.testing.assert_close(small[name], large[name], rtol=1e-6, atol=1e-7)


def test_coordinate_layout_is_shallow_to_deep_and_pose_gradient_is_finite() -> None:
    height, width = 8, 4
    dataset = _DatasetStub(height, width)
    base_points, viewdirs = _geometry(height, width)
    UltraNeRFSliceRenderer.validate_axial_layout(base_points)
    axial_coordinate = base_points[:, 0, 0]
    assert torch.all(axial_coordinate[1:] > axial_coordinate[:-1])

    model = _model()
    renderer = UltraNeRFSliceRenderer(dataset, query_chunk=13)
    translation = torch.zeros(3, dtype=torch.float32, device=DEVICE, requires_grad=True)
    points = base_points + translation
    maps = renderer.render_points(
        model,
        points,
        viewdirs,
        height=height,
        width=width,
        eval_seed=9,
    )
    maps["intensity_map"].mean().backward()
    assert translation.grad is not None
    assert torch.all(torch.isfinite(translation.grad))
    assert translation.grad.abs().sum() > 0


def test_ultra_renderer_is_exactly_zero_outside_valid_sector() -> None:
    height, width = 12, 8
    dataset = _DatasetStub(height, width)
    points, viewdirs = _geometry(height, width)
    model = _model()
    renderer = UltraNeRFSliceRenderer(dataset, query_chunk=13)
    valid_mask = torch.zeros((height, width), dtype=torch.bool, device=DEVICE)
    valid_mask[2:10, 2:6] = True

    maps = renderer.render_points(
        model,
        points,
        viewdirs,
        height=height,
        width=width,
        eval_seed=17,
        valid_mask=valid_mask,
    )

    for value in maps.values():
        assert torch.count_nonzero(value[~valid_mask]) == 0
