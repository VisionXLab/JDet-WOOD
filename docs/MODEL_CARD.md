---
license: apache-2.0
tags:
  - object-detection
  - oriented-object-detection
  - jittor
  - remote-sensing
  - dota
---

# Point2RBox-v3 (Jittor reproduction)

This is a **Jittor** reproduction of
[Point2RBox-v3](https://github.com/VisionXLab/Point2RBox-v3)
(point-supervised oriented object detection with SAM-guided pseudo-label
refinement, [arXiv:2509.26281](https://arxiv.org/pdf/2509.26281)), built on
JDet. Code: <https://github.com/mingqian-233/Point2RBox-v3-jittor>.

## Results (DOTA-v1.0, official test server, mAP50)

| Model | Paper (PyTorch) | This repo (Jittor) |
|---|---|---|
| Point2RBox-v3 end-to-end (12 ep) | 59.61 | **59.52** |
| Point2RBox-v3 two-stage (rotated-FCOS) | 66.09 | **65.50** |

The end-to-end number is the official DOTA-v1.0 Task1 test-server mAP50
(`0.5952330691632053`) and passes the project's paper−2.0 acceptance threshold
(57.61). Per-class AP table: TBD.

Local diagnostic only: the norm-eval-correct end-to-end checkpoint scores
**66.6056 mAP50** on the unmerged 1024×1024 trainval-patch protocol. The
same-machine full upstream PyTorch run scores **65.60**, while the published
upstream log reports **66.70**. This is not the official DOTA test-server
protocol; the official server result is reported in the table above.

The completed two-stage checkpoint scores **75.7029 mAP50** on the same local
trainval-patch diagnostic and **65.50 mAP50** on the official DOTA-v1.0 Task1
test server. The latter passes the project's paper−2.0 acceptance threshold
of 64.09.

The generated stage-2 pseudo boxes were also matched directly against all
245,953 trainval GT boxes. They achieve mean rotated IoU **0.7330**,
recall@0.5 **91.03%**, and recall@0.75 **56.58%**. The corresponding
official-code PyTorch v2 baseline is 0.7295 / 91.25% / 55.39%.

## Files

- `checkpoints/point2rbox_v3_1x_dota_ckpt_12.pkl` — end-to-end model
- `checkpoints/rotated_fcos_1x_dota_using_pseudo_ckpt_12.pkl` — stage-2 model
- `weights/mobile_sam.pkl` — MobileSAM converted weights (439 tensors, numpy
  pickle; native Jittor port, mask IoU 1.0000 vs PyTorch on CPU fp32)
- `weights/ted.pkl` — TED edge detector converted weights (parity 5.7e-6)
- `logs/` — training logs

## Training environment

- Jittor 1.3.8.5, numpy 1.26.4 (pinned — numpy≥2 silently corrupts jt.array
  inputs), CUDA 11.2 toolchain (g++-10), single A100 80GB, official
  hyper-parameters verbatim (end-to-end batch=2; stage-2 batch=4; AdamW
  5e-5, grad-clip 35, 12 epochs, LinearLR warmup 500 iters + MultiStepLR
  [8,11]).
- Config parity is locked by a zero-tolerance L0 test suite (17 tests,
  including the 74-line per-class SAM rule table byte-copied from upstream).

## Known differences from the upstream paper run

1. **SAM attention bias**: the upstream `mobile_sam` package calls `eval()`
   before `load_state_dict`, leaving TinyViT's non-persistent attention-bias
   cache at its random initialisation — i.e. the upstream training used an
   unseeded random attention bias (unreproducible across runs). This repo
   loads the weights correctly. Direction of the effect on mAP is unknown
   (expected neutral-to-positive). See `docs/config_parity.md`.
2. Upstream's copy-paste stage-2 branch is dead code (condition can never be
   true); reproduced verbatim.
3. `num_workers=0` instead of 2 (Jittor multi-process dataloader deadlock);
   pure infra, no effect on training math.
4. GPU-mode numeric noise vs PyTorch golden is at the cuDNN-TF32 level
   (~1e-3); CPU-mode strict-fp32 parity is 1e-5 to bit-exact per component.
