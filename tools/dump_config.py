"""dump_config.py — 在 p2r-torch 环境把 3 个官方 config 解析后展平成 JSON golden（L0）。

展平规则：嵌套 dict/list 以点号路径为 key（list 用下标），值为 JSON 标量。
覆盖 L0 要求的全部关键子树：model / pipeline（含顺序）/ 三个 dataloader /
optim_wrapper（含 clip_grad）/ param_scheduler / train_cfg / test_cfg /
custom_hooks / default_hooks。

用法：
    conda activate p2r-torch
    python tools/dump_config.py   # 产出 tests/parity/golden/config_<name>.json
"""
import json
import os

REPO = os.environ.get('P2R_REF', '/root/ref/Point2RBox-v3')
CONFIGS = [
    'configs/point2rbox_v2/point2rbox_v2-1x-dota.py',
    'configs/point2rbox_v2/point2rbox_v2-pseudo-generator-dota.py',
    'configs/point2rbox_v2/rotated-fcos-1x-dota-using-pseudo.py',
]
# L0 比对的顶层 key（全量子树逐值比对）
KEYS = ['model', 'train_pipeline', 'test_pipeline', 'train_dataloader', 'val_dataloader',
        'test_dataloader', 'optim_wrapper', 'param_scheduler', 'train_cfg', 'val_cfg',
        'test_cfg', 'custom_hooks', 'default_hooks', 'val_evaluator', 'test_evaluator',
        'env_cfg', 'data_root', 'dataset_type']

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'tests', 'parity', 'golden')


def flatten(node, prefix, out):
    if isinstance(node, dict):
        if not node:
            out[prefix] = {}
        for k in sorted(node):
            flatten(node[k], f'{prefix}.{k}' if prefix else str(k), out)
    elif isinstance(node, (list, tuple)):
        out[f'{prefix}.__len__'] = len(node)
        for i, v in enumerate(node):
            flatten(v, f'{prefix}[{i}]', out)
    else:
        out[prefix] = node


def main():
    import sys
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    from mmengine.config import Config

    os.makedirs(OUT_DIR, exist_ok=True)
    for c in CONFIGS:
        cfg = Config.fromfile(os.path.join(REPO, c)).to_dict()
        flat = {}
        for key in KEYS:
            if key in cfg:
                flatten(cfg[key], key, flat)
        name = os.path.splitext(os.path.basename(c))[0]
        out = os.path.join(OUT_DIR, f'config_{name}.json')
        with open(out, 'w') as f:
            json.dump(flat, f, indent=1, sort_keys=True, default=str)
        print(f'{name}: {len(flat)} 个扁平键 -> {out}')


if __name__ == '__main__':
    main()
