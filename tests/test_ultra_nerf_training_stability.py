from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from neuf.main import NeUF


class _WriterStub:
    def add_scalar(self, *_args, **_kwargs) -> None:
        pass


def test_ultra_loss_schedule_warms_up_then_reaches_official_weights() -> None:
    trainer = SimpleNamespace(
        ultra_mse_warmup_iters=500,
        ultra_loss_ramp_iters=1500,
        ultra_final_ms_ssim_weight=0.9,
    )
    assert NeUF._ultra_loss_weights(trainer, 0) == (1.0, 0.0)
    assert NeUF._ultra_loss_weights(trainer, 500) == (1.0, 0.0)
    mse_weight, ms_ssim_weight = NeUF._ultra_loss_weights(trainer, 1250)
    assert mse_weight == pytest.approx(0.55)
    assert ms_ssim_weight == pytest.approx(0.45)
    mse_weight, ms_ssim_weight = NeUF._ultra_loss_weights(trainer, 2000)
    assert mse_weight == pytest.approx(0.1)
    assert ms_ssim_weight == pytest.approx(0.9)

    trainer.ultra_final_ms_ssim_weight = 0.0
    assert NeUF._ultra_loss_weights(trainer, 10_000) == (1.0, 0.0)


def test_black_output_monitor_fails_fast_after_configured_patience() -> None:
    trainer = SimpleNamespace(
        renderer_name="ultra_nerf",
        ultra_collapse_threshold=1e-6,
        ultra_collapse_patience=3,
        _ultra_collapse_count=0,
        tb_writer=_WriterStub(),
    )
    black = torch.full((32, 1), 1e-12)
    NeUF._monitor_ultra_output(trainer, black, 10)
    NeUF._monitor_ultra_output(trainer, black, 11)
    with pytest.raises(RuntimeError, match="black-output collapse"):
        NeUF._monitor_ultra_output(trainer, black, 12)

    trainer._ultra_collapse_count = 2
    NeUF._monitor_ultra_output(trainer, torch.full((32, 1), 0.1), 13)
    assert trainer._ultra_collapse_count == 0
