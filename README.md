# Point2RBox-v3 for Jittor

Jittor/JDet implementation of
[Point2RBox-v3](https://github.com/VisionXLab/Point2RBox-v3), a
point-supervised oriented object detector with SAM-guided pseudo-label
refinement. The repository contains the end-to-end model, pseudo-label export,
the rotated-FCOS second stage, DOTA evaluation utilities, converted MobileSAM
and TED integrations, and numeric parity tests against the PyTorch reference.

## Results

DOTA-v1.0 Task1, mAP50 on the official test server:

| Model | Paper | Jittor | Checkpoint |
|---|---:|---:|---|
| Point2RBox-v3 end-to-end | 59.61 | **59.52** | [download](https://huggingface.co/Mingqian-233/Point2RBox-v3-jittor/blob/main/checkpoints/point2rbox_v3_1x_dota_ckpt_12.pkl) |
| Point2RBox-v3 + rotated-FCOS | 66.09 | **65.50** | [download](https://huggingface.co/Mingqian-233/Point2RBox-v3-jittor/blob/main/checkpoints/rotated_fcos_1x_dota_using_pseudo_ckpt_12.pkl) |

Both results satisfy the reproduction criterion of paper mAP minus 2.0. The
corresponding local 1024×1024 trainval-patch diagnostics are 66.6056 and
75.7029 mAP50; these local values are not the official test-server metric.

Converted weights and training logs are available in the
[Hugging Face repository](https://huggingface.co/Mingqian-233/Point2RBox-v3-jittor).

## Environment

The validated environment uses Python 3.10, Jittor 1.3.8.5, NumPy 1.26.4,
CUDA 11.2 and g++-10. NumPy must remain below version 2 with this Jittor
release.

```bash
pip install -r requirements.txt
export cc_path=/usr/bin/g++-10
export PYTHONPATH="$PWD:$PWD/python"
```

See [docs/environment.md](docs/environment.md) for compatibility notes.

## Data and weights

Prepare DOTA-v1.0 with 1024×1024 patches, gap 200. The validated split contains
21,046 trainval patches and 10,833 test patches. Dataset details and expected
layout are in [docs/data.md](docs/data.md). Update the dataset paths in the
selected config when using a different data root.

Place the converted auxiliary weights at:

```text
weights/mobile_sam.pkl
weights/ted.pkl
```

The files can be downloaded from Hugging Face. To convert the original PyTorch
weights instead:

```bash
python tools/convert_torch_weights.py /path/to/mobile_sam.pt weights/mobile_sam.pkl
python tools/convert_ted_weights.py /path/to/ted.pth weights/ted.pkl
```

`mobile_sam.pt` is a repository symlink to `weights/mobile_sam.pkl`, matching
the path expected by the model configuration.

## Training

End-to-end training:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/run_net.py \
  --config-file configs/point2rbox_v3/point2rbox_v3_1x_dota.py \
  --task train
```

Generate pseudo labels from the end-to-end checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_pseudo_labels.py \
  --config-file configs/point2rbox_v3/point2rbox_v3_pseudo_generator_dota.py \
  --ckpt work_dirs/point2rbox_v3_1x_dota/checkpoints/ckpt_12.pkl
```

Train the second-stage rotated-FCOS detector:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/run_net.py \
  --config-file configs/point2rbox_v3/rotated_fcos_1x_dota_using_pseudo.py \
  --task train
```

The configs reproduce the reference hyperparameters, including the 500-iter
linear warmup, epoch milestones 8 and 11, gradient clipping at 35, and the
class-specific SAM filtering table.

## Evaluation

Set `resume_path` in the selected config to the checkpoint and run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/run_net.py \
  --config-file configs/point2rbox_v3/point2rbox_v3_1x_dota.py \
  --task test
```

Convert JDet's test pickle and merge patches with the reference mmrotate DOTA
metric:

```bash
python tools/convert_test_results.py \
  --test-pkl work_dirs/point2rbox_v3_1x_dota/test/test_12.pkl \
  --out work_dirs/point2rbox_v3_1x_dota/test/merge_input.pkl

PYTHONPATH=/path/to/Point2RBox-v3-reference python tools/merge_dota_submission.py \
  --results work_dirs/point2rbox_v3_1x_dota/test/merge_input.pkl \
  --out work_dirs/point2rbox_v3_1x_dota/submission/Task1
```

The resulting `Task1.zip` contains the 15 standard DOTA class files.

## Verification

The parity suite covers config values, geometry and rotated ops, losses and
gradients, detector routing, MobileSAM, TED, pseudo-label serialization and
dataset adapters.

```bash
PYTHONPATH="$PWD:$PWD/python" python -m pytest \
  tests/parity tests/test_v3_norm_eval.py -q
python tests/test_sam.py
python tests/test_ted.py
```

Technical translation details, including the native MobileSAM port, are in
[docs/porting_notes.md](docs/porting_notes.md). Exact config mappings are in
[docs/config_parity.md](docs/config_parity.md).

## Acknowledgements

This implementation is built on
[JDet](https://github.com/Jittor/JDet),
[Wholly-WOOD for Jittor](https://github.com/VisionXLab/whollywood-jittor),
[Point2RBox-v2 for Jittor](https://github.com/mingqian-233/Point2RBox-v2-jittor),
[MobileSAM](https://github.com/ChaoningZhang/MobileSAM), and the original
[Point2RBox-v3](https://github.com/VisionXLab/Point2RBox-v3).

Released under the Apache License 2.0. See [LICENSE.txt](LICENSE.txt).
