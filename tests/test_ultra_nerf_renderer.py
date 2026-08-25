from __future__ import annotations

import numpy as np
import pytest
import torch

from neuf.ultra_nerf_renderer import (
    UltraNeRFRenderer,
    exclusive_cumprod,
    gaussian_psf_kernel,
    make_generator,
)


def test_exclusive_cumprod_matches_numpy_and_uses_axial_dimension() -> None:
    values = torch.tensor([[0.5, 0.4, 0.3], [0.9, 0.8, 0.7]])
    actual = exclusive_cumprod(values, dim=1)
    expected = np.array([[1.0, 0.5, 0.2], [1.0, 0.9, 0.72]])
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual[:, 0], torch.ones(2), rtol=0, atol=0)


def test_gaussian_psf_has_official_shape_sum_and_orientation() -> None:
    kernel = gaussian_psf_kernel(3, lateral_std=2.0, axial_std=1.0)
    assert kernel.shape == (1, 1, 7, 7)
    torch.testing.assert_close(kernel.sum(), torch.tensor(1.0))

    coordinates = torch.arange(-3, 4, dtype=kernel.dtype)
    weights = kernel[0, 0]
    lateral_variance = torch.sum(weights * coordinates[:, None] ** 2)
    axial_variance = torch.sum(weights * coordinates[None, :] ** 2)
    assert lateral_variance > axial_variance


def test_fixed_indicators_match_analytic_transmission_and_echo() -> None:
    renderer = UltraNeRFRenderer(psf_half_size=0)
    width, height = 2, 4
    raw = torch.zeros(width, height, 5)
    raw[..., 0] = 2.0
    raw[..., 1] = torch.logit(torch.tensor(0.25))
    raw[..., 4] = torch.logit(torch.tensor(0.8))
    z = torch.arange(height, dtype=torch.float32)[None].repeat(width, 1) * 0.1
    border = torch.tensor([[0, 1, 0, 1], [1, 0, 0, 0]], dtype=torch.float32)
    scatter = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)

    maps = renderer(
        raw,
        z,
        border_indicator=border,
        scatterers_density=scatter,
    )
    expected_att = torch.exp(
        -2.0 * torch.tensor([[0.0, 0.1, 0.2, 0.3]]).repeat(width, 1)
    )
    expected_reflection = exclusive_cumprod(1.0 - 0.25 * border, dim=1)
    transmission = expected_att * expected_reflection

    torch.testing.assert_close(maps["attenuation_transmission"], expected_att)
    torch.testing.assert_close(maps["reflection_transmission"], expected_reflection)
    torch.testing.assert_close(maps["transmission"], transmission)
    torch.testing.assert_close(maps["b"], transmission * scatter * 0.8)
    torch.testing.assert_close(maps["r"], transmission * 0.25 * border)
    torch.testing.assert_close(maps["intensity_map"], maps["b"] + maps["r"])


def test_zero_border_and_scatter_produce_zero_echo() -> None:
    renderer = UltraNeRFRenderer()
    raw = torch.randn(5, 8, 5)
    z = torch.linspace(0, 0.07, 8)[None].repeat(5, 1)
    zeros = torch.zeros(5, 8)
    maps = renderer(
        raw,
        z,
        border_indicator=zeros,
        scatterers_density=zeros,
    )
    for name in ("r", "b", "intensity_map"):
        torch.testing.assert_close(maps[name], torch.zeros_like(maps[name]))


def test_impulse_psf_is_wider_laterally_than_axially() -> None:
    renderer = UltraNeRFRenderer()
    raw = torch.zeros(9, 9, 5)
    z = torch.linspace(0, 0.008, 9)[None].repeat(9, 1)
    impulse = torch.zeros(9, 9)
    impulse[4, 4] = 1
    zeros = torch.zeros_like(impulse)
    maps = renderer(
        raw,
        z,
        border_indicator=impulse,
        scatterers_density=zeros,
    )
    expected = torch.zeros_like(impulse)
    expected[1:8, 1:8] = renderer.psf_kernel[0, 0]
    torch.testing.assert_close(maps["psf_border"], expected)
    assert maps["psf_border"][2, 4] > maps["psf_border"][4, 2]


def test_seeded_sampling_is_reproducible() -> None:
    renderer = UltraNeRFRenderer()
    raw = torch.zeros(12, 16, 5)
    z = torch.linspace(0, 0.015, 16)[None].repeat(12, 1)

    def render(seed: int) -> dict[str, torch.Tensor]:
        return renderer(
            raw,
            z,
            border_generator=make_generator(raw.device, seed),
            scatter_generator=make_generator(raw.device, seed + 1),
        )

    first = render(17)
    second = render(17)
    different = render(19)
    torch.testing.assert_close(first["intensity_map"], second["intensity_map"], rtol=0, atol=0)
    assert not torch.equal(first["border_indicator"], different["border_indicator"])


def test_continuous_branches_and_coordinates_receive_gradients() -> None:
    renderer = UltraNeRFRenderer(psf_half_size=1)
    raw = torch.full((3, 5, 5), 0.4, requires_grad=True)
    z = torch.linspace(0, 0.004, 5)[None].repeat(3, 1).requires_grad_()
    ones = torch.ones(3, 5)
    maps = renderer(
        raw,
        z,
        border_indicator=ones,
        scatterers_density=ones,
    )
    maps["intensity_map"].sum().backward()
    assert raw.grad is not None and torch.all(torch.isfinite(raw.grad))
    assert z.grad is not None and torch.all(torch.isfinite(z.grad))
    assert raw.grad[..., 0].abs().sum() > 0
    assert raw.grad[..., 1].abs().sum() > 0
    assert raw.grad[..., 4].abs().sum() > 0
    assert z.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_renderer_smoke() -> None:
    device = torch.device("cuda")
    renderer = UltraNeRFRenderer().to(device)
    raw = torch.zeros(4, 8, 5, device=device, requires_grad=True)
    z = torch.linspace(0, 0.007, 8, device=device)[None].repeat(4, 1)
    maps = renderer(
        raw,
        z,
        border_generator=make_generator(device, 3),
        scatter_generator=make_generator(device, 4),
    )
    maps["intensity_map"].sum().backward()
    assert raw.grad is not None
    assert torch.all(torch.isfinite(maps["intensity_map"]))

