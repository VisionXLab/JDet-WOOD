<div align="center">
  <img src="assets/jdet-wood-logo.png" width="280" alt="JDet-WOOD Pami logo">
  <h1>JDet-WOOD</h1>
  <p><strong>Unified Jittor implementations for weakly supervised oriented object detection</strong></p>
</div>

JDet-WOOD brings six related weakly supervised oriented object detection methods
into one installable [JDet](https://github.com/Jittor/JDet) codebase. All models
share the same `python/jdet` package, runner, datasets, rotated operators, and
command-line interface; select a method by changing only the config file.

## Supported models

| Method | Supervision | JDet model type | Primary config |
|---|---|---|---|
| H2RBox | horizontal boxes | `H2RBox` | [`h2rbox_obb_r50_adamw_fpn_1x_dota.py`](configs/whollywood/h2rbox_obb_r50_adamw_fpn_1x_dota.py) |
| H2RBox-v2 | horizontal boxes | `H2RBoxV2P` | [`h2rbox_v2p_obb_r50_adamw_fpn_1x_dota.py`](configs/whollywood/h2rbox_v2p_obb_r50_adamw_fpn_1x_dota.py) |
| Wholly-WOOD | points, HBoxes, RBoxes, or mixed labels | `WhollyWood` | [`whollywood_obb_r50_adamw_fpn_1x_dota.py`](configs/whollywood/whollywood_obb_r50_adamw_fpn_1x_dota.py) |
| Point2RBox | points | `Point2RBox` | [`point2rbox_obb_r50_adamw_fpn_1x_dota.py`](configs/whollywood/point2rbox_obb_r50_adamw_fpn_1x_dota.py) |
| Point2RBox-v2 | points | `Point2RBoxV2` | [`point2rbox_v2_final_fixed.py`](configs/point2rbox_v2/point2rbox_v2_final_fixed.py) |
| Point2RBox-v3 | points | `Point2RBoxV3` | [`point2rbox_v3_1x_dota.py`](configs/point2rbox_v3/point2rbox_v3_1x_dota.py) |

The Wholly-WOOD family lives under `configs/whollywood`, while the newer
Point2RBox releases keep their stage-1, pseudo-label, and stage-2 configs under
`configs/point2rbox_v2` and `configs/point2rbox_v3`.

## Installation

Validated environment:

- Linux, Python 3.10
- Jittor 1.3.8.5
- NumPy 1.26.4 (NumPy 2.x is not supported by this Jittor release)
- CUDA 11.2 and g++-10 for the validated GPU setup

```bash
git clone https://github.com/VisionXLab/JDet-WOOD.git
cd JDet-WOOD
bash scripts/setup_env.sh
conda activate jdet-wood
export PYTHONPATH="$PWD:$PWD/python"
```

For an existing compatible environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
export cc_path=/usr/bin/g++-10
export PYTHONPATH="$PWD:$PWD/python"
```

See [`docs/environment.md`](docs/environment.md) for compiler, CUDA, and Jittor
compatibility notes.

## Data and auxiliary weights

The released DOTA-v1.0 configs use 1024×1024 patches with a 200-pixel gap and
expect the validated split under `/root/data/split_ss_dota`:

```text
/root/data/split_ss_dota/
├── trainval/
│   ├── images/
│   └── annfiles/
└── test/
    └── images/
```

Update the dataset paths in the selected config if your data is elsewhere.
The JDet preprocessing utilities and additional supported datasets are
documented in [`docs/data.md`](docs/data.md) and [`docs/dota.md`](docs/dota.md).

Point2RBox-v3 additionally expects converted MobileSAM and TED weights:

```text
weights/mobile_sam.pkl
weights/ted.pkl
```

Convert original PyTorch weights when needed:

```bash
python tools/convert_torch_weights.py /path/to/mobile_sam.pt weights/mobile_sam.pkl
python tools/convert_ted_weights.py /path/to/ted.pth weights/ted.pkl
```

Converted weights, checkpoints, and logs are available from the
[Point2RBox-v3 Jittor collection](https://huggingface.co/Mingqian-233/Point2RBox-v3-jittor)
and the
[Point2RBox-v2 Jittor collection](https://huggingface.co/Mingqian-233/Point2RBox-v2-jittor).

## Train and test

Every model uses the same entry point:

```bash
python tools/run_net.py --config-file <config> --task train
python tools/run_net.py --config-file <config> --task test
```

Examples:

```bash
# Wholly-WOOD, H2RBox, H2RBox-v2, and Point2RBox
python tools/run_net.py --config-file configs/whollywood/whollywood_obb_r50_adamw_fpn_1x_dota.py --task train
python tools/run_net.py --config-file configs/whollywood/h2rbox_obb_r50_adamw_fpn_1x_dota.py --task train
python tools/run_net.py --config-file configs/whollywood/h2rbox_v2p_obb_r50_adamw_fpn_1x_dota.py --task train
python tools/run_net.py --config-file configs/whollywood/point2rbox_obb_r50_adamw_fpn_1x_dota.py --task train

# Point2RBox-v2 and Point2RBox-v3 stage 1
python tools/run_net.py --config-file configs/point2rbox_v2/point2rbox_v2_final_fixed.py --task train
python tools/run_net.py --config-file configs/point2rbox_v3/point2rbox_v3_1x_dota.py --task train
```

Set `resume_path` in the selected config to evaluate a downloaded checkpoint,
then run with `--task val` or `--task test`.

### Point2RBox-v2 two-stage workflow

```bash
# 1. Train the end-to-end point-supervised model.
CUDA_VISIBLE_DEVICES=0 python tools/run_net.py \
  --config-file configs/point2rbox_v2/point2rbox_v2_final_fixed.py \
  --task train

# 2. Export rotated pseudo labels.
CUDA_VISIBLE_DEVICES=0 python tools/generate_pseudo_labels.py \
  --config configs/point2rbox_v2/point2rbox_v2_pseudo_generator_dota.py \
  --ckpt work_dirs/point2rbox_v2_1x_dota_final_fixed/checkpoints/ckpt_12.pkl \
  --out /root/data/split_ss_dota/point2rbox_v2_pseudo_labels

# 3. Train rotated FCOS from the pseudo labels.
CUDA_VISIBLE_DEVICES=0 python tools/run_net.py \
  --config-file configs/point2rbox_v2/rotated_fcos_1x_dota_using_pseudo.py \
  --task train
```

`tools/auto_stage2_pipeline.sh` automates the same workflow and accepts
`GPU_ID`, `ENV_NAME`, `STAGE1_PID`, and `OUT_DIR` environment overrides.

### Point2RBox-v3 two-stage workflow

```bash
# 1. Train the end-to-end detector.
CUDA_VISIBLE_DEVICES=0 python tools/run_net.py \
  --config-file configs/point2rbox_v3/point2rbox_v3_1x_dota.py \
  --task train

# 2. Export SAM-refined pseudo labels.
CUDA_VISIBLE_DEVICES=0 python tools/export_pseudo_labels.py \
  --config-file configs/point2rbox_v3/point2rbox_v3_pseudo_generator_dota.py \
  --ckpt work_dirs/point2rbox_v3_1x_dota/checkpoints/ckpt_12.pkl

# 3. Train rotated FCOS from the pseudo labels.
CUDA_VISIBLE_DEVICES=0 python tools/run_net.py \
  --config-file configs/point2rbox_v3/rotated_fcos_1x_dota_using_pseudo.py \
  --task train
```

## Reproduced Point2RBox results

DOTA-v1.0 Task1 mAP50 on the official test server:

| Model | Paper | Jittor | Checkpoint |
|---|---:|---:|---|
| Point2RBox-v2 end-to-end | 51.00 | 48.95 | [download](https://huggingface.co/Mingqian-233/Point2RBox-v2-jittor/resolve/main/stage1/point2rbox_v2_stage1_ckpt_12.pkl) |
| Point2RBox-v2 + rotated FCOS | 62.61 | **59.39** | [download](https://huggingface.co/Mingqian-233/Point2RBox-v2-jittor/resolve/main/stage2/rotated_fcos_stage2_ckpt_12.pkl) |
| Point2RBox-v3 end-to-end | 59.61 | **59.52** | [download](https://huggingface.co/Mingqian-233/Point2RBox-v3-jittor/resolve/main/checkpoints/point2rbox_v3_1x_dota_ckpt_12.pkl) |
| Point2RBox-v3 + rotated FCOS | 66.09 | **65.50** | [download](https://huggingface.co/Mingqian-233/Point2RBox-v3-jittor/resolve/main/checkpoints/rotated_fcos_1x_dota_using_pseudo_ckpt_12.pkl) |

## Verification

The test suite covers registry/config integration, Jittor numerics, rotated
geometry and losses, optimizer resume, datasets, MobileSAM, TED, pseudo-label
serialization, and v2/v3 detector routing.

```bash
export PYTHONPATH="$PWD:$PWD/python"
python -m pytest tests/test_jdet_wood_registry.py tests/smoke -q
python -m pytest tests/parity tests/test_v3_norm_eval.py -q
python tests/test_sam.py
python tests/test_ted.py
```

GPU-only parity tests require the validated CUDA environment. The repository
does not bundle DOTA data or released model checkpoints.

## Source snapshots

JDet-WOOD was unified from these VisionXLab repositories:

| Source | Imported commit | Role |
|---|---|---|
| [h2rbox-jittor](https://github.com/VisionXLab/h2rbox-jittor) | `90e756ba375cfa74ad55bf57527ed17cb4d1ebbe` | H2RBox provenance |
| [whollywood-jittor](https://github.com/VisionXLab/whollywood-jittor) | `6bca5e07d5ea60ba2f22f06ea90961e0b4235b37` | Wholly-WOOD family |
| [Point2RBox-v2-jittor](https://github.com/VisionXLab/Point2RBox-v2-jittor) | `66bf12fa6764c44da4046018159b1d7e56b9c249` | v2 release fixes and pipeline |
| [Point2RBox-v3-jittor](https://github.com/VisionXLab/Point2RBox-v3-jittor) | `d309b47f060bae040f7889cfd187c2ffc393db5f` | unified base and v3 implementation |

Please cite the corresponding method papers when using a model. BibTeX entries
for the Wholly-WOOD, H2RBox, H2RBox-v2, and Point2RBox family are collected in
the [Point2RBox-v2 source README](https://github.com/VisionXLab/Point2RBox-v2-jittor#citation).

## Acknowledgements and license

Built on [Jittor](https://github.com/Jittor/jittor),
[JDet](https://github.com/Jittor/JDet),
[MobileSAM](https://github.com/ChaoningZhang/MobileSAM), and the VisionXLab
weakly supervised oriented detection projects listed above.

Released under the Apache License 2.0. See [`LICENSE.txt`](LICENSE.txt).
