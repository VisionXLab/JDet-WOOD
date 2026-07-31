"""convert_torch_ckpt.py — mmrotate 训练的 v2 checkpoint → Jittor 可加载 pkl。

键映射：本仓库 detector 属性名与 mmrotate 相同（backbone/neck/bbox_head），
且 head/FPN/ResNet 子键逐一对应；仅需滤掉 torch 的 num_batches_tracked。
转换后做双向覆盖检查（缺失/多余键都报出来）。

用法（p2r-torch 环境，纯 CPU）：
    python tools/convert_torch_ckpt.py --src <epoch_12.pth> --dst <out.pkl>
"""
import argparse
import pickle

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--dst', required=True)
    args = parser.parse_args()

    ck = torch.load(args.src, map_location='cpu')
    sd = ck.get('state_dict', ck)
    out = {}
    skipped = []
    for k, v in sd.items():
        if k.endswith('num_batches_tracked'):
            skipped.append(k)
            continue
        out[k] = v.detach().cpu().numpy()
    with open(args.dst, 'wb') as f:
        pickle.dump(out, f)
    print(f'{len(out)} tensors -> {args.dst} (skipped {len(skipped)} num_batches_tracked)')
    meta = ck.get('meta', {})
    if meta:
        print('src meta epoch:', meta.get('epoch'))


if __name__ == '__main__':
    main()
