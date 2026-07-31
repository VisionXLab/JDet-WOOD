# PyTorch to Jittor porting notes

This document records the implementation choices that are required for numeric
and gradient parity with the Point2RBox-v3 PyTorch reference.

## MobileSAM

The native Jittor implementation is in `python/jdet/models/sam/`:

| File | Component |
|---|---|
| `tiny_vit.py` | TinyViT image encoder, MBConv blocks and window attention |
| `prompt_encoder.py` | point/box/mask prompt encoding and random positional encoding |
| `transformer.py` | two-way transformer |
| `mask_decoder.py` | mask tokens, hypernetworks and IoU prediction |
| `sam.py` | preprocessing and mask postprocessing |
| `predictor.py` | `SamPredictor` and longest-side resize |
| `build.py` | `sam_model_registry['vit_t']` and converted weight loading |

The public API intentionally matches MobileSAM, so the loss code can use
`sam_model_registry['vit_t']` and `SamPredictor` without an adapter layer.

### Weight conversion

PyTorch state dictionaries are converted to a pickle mapping parameter names
to NumPy arrays:

```bash
python tools/convert_torch_weights.py mobile_sam.pt weights/mobile_sam.pkl
```

`num_batches_tracked` entries are ignored because Jittor BatchNorm does not use
them. TinyViT `Conv2d_BN` keeps the original `c` and `bn` child names so the
remaining keys load directly. The random positional-encoding matrix is a real
checkpoint value, not a disposable initialization buffer; loading code and
tests verify that it is restored.

TinyViT caches indexed attention biases in evaluation mode. The cache is
refreshed after loading weights, preventing stale random biases from surviving
checkpoint restoration.

### Tensor semantics

- PyTorch `transpose(a, b)` swaps two dimensions. Jittor's transpose API is
  expressed as a full permutation in this port; all window partition/reverse
  paths therefore use explicit `permute` orders.
- Point masking is written with `where` and mask multiplication instead of
  in-place indexed assignment, preserving gradients.
- `repeat_interleave` in the mask decoder is implemented with explicit
  reshape/expand/reshape operations.
- Predictor resizing uses PIL rather than torchvision and keeps the reference
  longest-side coordinate transform.

The converted 439-tensor checkpoint reaches mask IoU 1.0000 against PyTorch in
CPU fp32 and at least 0.9982 on GPU in the repository tests.

## TED edge detector

TED is implemented in `python/jdet/models/edge/`. Its convolutional weights are
converted with:

```bash
python tools/convert_ted_weights.py ted.pth weights/ted.pkl
```

The output edge maps match the PyTorch implementation with maximum relative
error below 5.7e-6 in the validated CPU path.

## Jittor semantic differences

### Detach and in-place updates

`Var.stop_grad()` changes the variable itself; it is not equivalent to
PyTorch's `detach()`. Graph branches that require detached values use
`Var.detach()`. Loss-path mask updates are rebuilt out-of-place with `where` or
concatenation because indexed in-place writes can silently alter gradients.

### Batched linear algebra

Jittor's GPU `matmul` does not broadcast batch dimensions. Batch dimensions are
expanded explicitly before batched products. The covariance operations used by
Point2RBox are 2×2, so `python/jdet/ops/linalg2x2.py` provides closed-form
determinant, solve and symmetric eigendecomposition implementations without a
CuPy dependency.

Repeated eigenvalues do not define a unique eigenbasis. Degenerate tests compare
basis-independent quantities such as trace, determinant and reconstructed
matrices rather than individual eigenvectors.

### Reductions and grouping

Some Jittor reductions return shape `(1,)` rather than a scalar. Scalar lists
are concatenated instead of stacked. Instance grouping uses scatter reductions;
Python loops over tensors are avoided because they create one graph fragment per
element and can make dense batches stall.

### Normalization

Jittor's default GroupNorm computes variance as `E[x^2] - E[x]^2`. The port uses
the two-pass implementation in `python/jdet/models/utils/modules.py` to match
PyTorch and avoid cancellation. The detector explicitly redispatches
`backbone.train()` after Jittor's recursive mode switch so frozen stages and
`norm_eval=True` remain effective.

### Rotated geometry

- JDet's rotated RoIAlign angle direction matches mmcv
  `clockwise=True`; Point2RBox uses `aligned=True` explicitly.
- JDet rotated NMS returns retained indices in input order. Prediction code sorts
  by score before truncating to `max_per_img`.
- Rotated IoU polygon area is evaluated in centered coordinates. The shoelace
  formula is translation invariant, while centering avoids false areas caused by
  cancellation of large image-coordinate products.
- `torchvision.transforms.functional.resized_crop(..., antialias=True)` is
  reproduced by explicit antialiased bilinear weights in the scale augmentation
  path.

## Numeric validation

The repository stores fixed PyTorch goldens under `tests/parity/golden/`.
CPU fp32 checks use strict tolerances for losses and feature gradients. GPU
checks use looser aggregate tolerances because cuDNN algorithm selection and
TF32 can vary between processes even when the implementation is unchanged.
Config values, including the class-specific SAM mask filtering table, are
checked separately with zero tolerance.
