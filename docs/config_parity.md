# Configuration parity

The files in `configs/point2rbox_v3/` reproduce the official mmrotate
Point2RBox-v3 configuration. `tests/parity/test_L0_v3_config.py` compares the
flattened values with stored reference goldens.

## Framework mappings

| PyTorch/mmrotate | Jittor/JDet | Equivalent behavior |
|---|---|---|
| `mmdet.ResNet(out_indices=...)` | `Resnet50(return_stages=...)` | same feature stages |
| torchvision ResNet-50 initialization | `pretrained=True` | same pretrained source |
| `mmdet.FPN` | `FPN` | matching channels and levels |
| `mmdet.FocalLoss` | `MMDetFocalLoss` | mmdetection reduction semantics |
| `GWDLoss` | `GDLoss(loss_type='gwd')` | matching Gaussian distance |
| AdamW + `clip_grad` | JDet AdamW with `grad_clip` | global L2 clip at 35 |
| LinearLR + MultiStepLR | `LinearWarmupMultiStepLR` | pointwise-equal LR sequence |
| `SetEpochInfoHook` | runner `model.set_epoch(epoch)` | epoch switches preserved |
| mmrotate qbox transforms | `P2RV2DOTADataset` | qbox/rbox and point labels |
| mmrotate resize/flip | `MMRotateResize` / `MMRotateRandomFlip` | coordinate and angle parity |

## Locked training values

- 12 epochs, evaluation at epoch 12, checkpoint every epoch.
- AdamW learning rate `5e-5`, gradient clip `35`.
- Linear warmup from factor `1/3` for 500 iterations; learning-rate milestones
  at epochs 8 and 11 with factor 0.1.
- End-to-end batch size 2 and second-stage batch size 4 on one GPU.
- End-to-end weight decay 0.05; second-stage weight decay 0.005.
- Five FPN strides `[8, 16, 32, 64, 128]` for v3.
- Self-supervision probabilities `[0.68, 0.07, 0.25]`.
- Epoch 6 switches for edge supervision, pseudo-label assignment and copy-paste
  routing. The upstream key spelling
  `label_assign_pseudo_label_switch_eopch` is intentionally preserved.
- Validation points to the reference trainval split, matching the official
  diagnostic protocol.

## SAM filtering configuration

`configs/point2rbox_v3/_base_sam-dotav1-0.py` preserves the complete
class-specific filtering table from the reference. The L0 test compares every
key, value and tuple/list type. Notable intentional values include:

- classes 3, 8 and 10 use circularity weight `-3` with circularity penalty 100;
- prompt points outside a mask receive the reference hard center-alignment
  penalty;
- the internal fallback filter table is kept separate from the config table,
  because the reference values differ and the configured training path always
  passes the explicit table.

## Infrastructure-only adaptation

The validated Jittor setup uses `num_workers=0`. Jittor 1.3.8.5 can deadlock in
the multiprocessing dataset ring buffer for this variable-instance workload.
This changes loading concurrency only; sample definitions, transforms and
training math remain unchanged.

## Upstream MobileSAM cache behavior

The upstream builder enters evaluation mode before loading the TinyViT state
dict, which can leave a cached attention-bias tensor derived from initialization
values. The Jittor builder loads weights first and then refreshes the evaluation
cache. This is the deterministic checkpoint-loading behavior documented in
[porting_notes.md](porting_notes.md).
