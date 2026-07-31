"""L1 parity test: Jittor TED vs PyTorch golden reference.

Prerequisites (run in p2r-torch env first):
    python tools/convert_torch_weights.py \
        /root/ref/Point2RBox-v3/third_parties/ted/ted.pth weights/ted.pkl
    python tools/dump_ted_reference.py --ted-dir ... --out tests/data/ted_reference.npz

Then run in p2r-jittor env:
    python tests/test_ted.py

Acceptance (plan M2'): relative error < 1e-3 on the edge maps.
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import jittor as jt  # noqa: E402
from jdet.models.edge.ted import TED  # noqa: E402

REF = os.path.join(os.path.dirname(__file__), 'data', 'ted_reference.npz')
WEIGHTS = os.path.join(os.path.dirname(__file__), '..', 'weights', 'ted.pkl')
RTOL = 1e-3


def rel_err(a, b):
    return np.abs(a - b).max() / (np.abs(b).max() + 1e-12)


def main():
    ref = np.load(REF)
    with open(WEIGHTS, 'rb') as f:
        params = pickle.load(f)

    model = TED()
    model.load_parameters({k: jt.array(v) for k, v in params.items()})
    model.eval()

    n_checked = 0
    worst = 0.0
    for prefix in ('rand', 'img'):
        key = f'input_{prefix}'
        if key not in ref:
            continue
        x = jt.array(ref[key])
        with jt.no_grad():
            outs = model(x)
        for i, o in enumerate(outs):
            golden = ref[f'out_{prefix}_{i}']
            e = rel_err(o.numpy(), golden)
            worst = max(worst, e)
            status = 'OK ' if e < RTOL else 'FAIL'
            print(f'[{status}] {prefix} out[{i}]: rel_err={e:.3e}')
            assert e < RTOL, f'{prefix} out[{i}] rel_err {e:.3e} >= {RTOL}'
            n_checked += 1

    print(f'PASS: {n_checked} outputs, worst rel_err={worst:.3e}')


if __name__ == '__main__':
    main()
