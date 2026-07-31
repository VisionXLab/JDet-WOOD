"""convert_ted_weights.py — p2r-torch 环境把 ted.pth 转成 jittor 可加载的 ted.pkl。

同时 dump 一份 TED 前向 golden（固定输入 → 4 个输出）供 L2 parity 测试。
用法：
    conda activate p2r-torch
    python tools/convert_ted_weights.py
产出：
    third_parties/ted/ted.pkl                  （numpy dict 权重）
    tests/parity/golden/ted_forward.npz        （前向 golden）
"""
import os
import pickle
import sys

import numpy as np
import torch

REPO = os.environ.get('P2R_REF', '/root/ref/Point2RBox-v3')
sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
PKL = os.path.join(HERE, '..', 'third_parties', 'ted', 'ted.pkl')
GOLDEN = os.path.join(HERE, '..', 'tests', 'parity', 'golden', 'ted_forward.npz')


def main():
    from third_parties.ted.ted import TED

    sd = torch.load(os.path.join(REPO, 'third_parties/ted/ted.pth'),
                    map_location='cpu')
    np_sd = {k: v.detach().cpu().numpy() for k, v in sd.items()}
    with open(PKL, 'wb') as f:
        pickle.dump(np_sd, f)
    print(f'weights: {len(np_sd)} tensors -> {PKL}')

    model = TED()
    model.load_state_dict(sd)
    model.eval()
    rng = np.random.RandomState(41)
    x = rng.uniform(-2, 2, (2, 3, 128, 128)).astype(np.float32)
    with torch.no_grad():
        outs = model(torch.tensor(x))
    np.savez(GOLDEN, x=x, **{f'out{i}': o.numpy() for i, o in enumerate(outs)})
    print(f'golden: 4 outputs -> {GOLDEN}')
    print('out3 (detector 用的边缘图) mean:', outs[3].mean().item())


if __name__ == '__main__':
    main()
