# NeUF quality-first baseline

This note turns the geometry/imaging-model diagnosis into reproducible baselines.
It deliberately keeps the current point-sampled intensity field and avoids the
experimental ray marcher until an elevational resolution-cell model is available.

## What the current code actually does

- `neuf.main` imports `neuf.slice_renderer.SliceRenderer`, so training and
  validation query one 3D point per B-mode pixel.
- `neuf.slice_render_ray` is not imported by the trainer. Its `mean`, `sum`, and
  optical `alpha` modes are experimental and are not an ultrasound PSF model.
- `neuf.render_3d_volume_from_ckpt` directly queries a Cartesian world-coordinate
  grid. It does not call the legacy 2.5D interpolation code.
- `neuf.recons3d_exact_from_saved` remains a legacy/comparison path and should not
  be used to judge the native sharpness of a NeUF checkpoint.

## Direct float32 export

The direct exporter now defaults to `float32` MHD output:

```bash
python -m neuf.render_3d_volume_from_ckpt \
  --ckpt path/to/checkpoint.pkl \
  --output exports/direct_float \
  --output-exact \
  --save-volume-npy
```

Use `--output-type uint8` only for an explicit display-oriented comparison. That
mode min-max scales non-zero foreground values and therefore is not suitable for
quantitative intensity comparison.

## Structured training and pose regularization

`CurriculumRPS` implements a Random -> Patch -> Slice schedule. The two ratios are
durations, so `0.2 + 0.5` leaves the final `0.3` for full slices:

```bash
python -m neuf \
  --dataset path/to/baked_dataset.pkl \
  --encoding DUAL_HASH \
  --training-mode CurriculumRPS \
  --curriculum-random-ratio 0.2 \
  --curriculum-patch-ratio 0.5 \
  --patch-size 32 \
  --grad-weight 0.1
```

`grad_weight` is active only for structured Patch/Slice batches. It was previously
accepted by the CLI but not added to the training loss.

New training runs use a sigmoid output so normalized B-mode predictions remain in
`[0, 1]`. Checkpoints written before this option automatically retain their original
identity/linear output when loaded. Use `--intensity-activation identity` only for an
explicit ablation; forcing sigmoid while resuming an old checkpoint changes that
checkpoint's learned function.

For joint pose refinement, start pose updates after the initial field fit and use
both correction-magnitude and trajectory terms:

```bash
  --optimize-poses \
  --pose-anchor-first \
  --pose-start-iter 2000 \
  --pose-rotation-reg-weight 1e-4 \
  --pose-translation-reg-weight 1e-5 \
  --pose-velocity-reg-weight 1e-4 \
  --pose-acceleration-reg-weight 1e-4
```

The trajectory terms operate on adjacent six-dimensional corrections in stored
slice order. Rotation components use radians and translation components use mm;
the example weights are conservative starting values, not calibrated constants.

The Patient0 PBS launcher uses this quality-first schedule by default. Every value
can still be overridden through its corresponding environment variable.

The same launcher delays the single sagittal image until iteration 4000, ramps its
weight to a conservative `0.1` over 2000 iterations, and starts its independent pose
at iteration 5000. For the clean fixed-geometry ablation, submit with
`USE_SAGITTAL=0`; the CLI equivalents are `--sagittal-start-iter` and
`--sagittal-ramp-iters`.

## Minimum ablation order

1. Train with fixed tracked poses and no sagittal auxiliary loss.
2. Train the same configuration with delayed, regularized pose refinement.
3. Render training and held-out slices from both checkpoints.
4. Export both checkpoints with the direct float32 Cartesian exporter.
5. Compare the direct outputs against KNN on the same physical grid.
6. Only then compare against the legacy 2.5D reconstruction path.

Use whole held-out frames or angle ranges rather than random held-out pixels. Report
metrics separately for training views, held-out tracked views, and orthogonal views.

## Deferred model changes

An ultrasound resolution-cell renderer needs explicit axial, lateral, and
elevational bases plus calibrated PSF widths. The existing `viewdirs` vector is the
in-plane axial direction, so marching along it is not a slice-thickness model. A
future implementation should sample primarily along the rotated local z/elevational
axis and apply an anisotropic Gaussian kernel. It should be evaluated after the
point/geometry/export baselines above are stable.
