"""B-side L1 test for A's ops/linalg2x2: the w==h degenerate case.

Plan M5' acceptance explicitly requires B to independently re-verify the
square-object case (sigma = a*I, i.e. b==0 and equal eigenvalues), where a
naive eigh gradient produces NaN (division by lambda1-lambda2 == 0).

Run in p2r-jittor:  python tests/test_linalg2x2_degenerate.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import jittor as jt  # noqa: E402
from jdet.ops.linalg2x2 import eigh_2x2  # noqa: E402


def check(name, sigma_np):
    sigma = jt.array(sigma_np.astype(np.float32))
    L, V = eigh_2x2(sigma)
    # reconstruction: V diag(L) V^T == sigma
    rec = jt.matmul(jt.matmul(V, jt.init.eye(2)[None] * L[:, None, :]),
                    V.transpose(0, 2, 1))
    rec_err = float(jt.abs(rec - sigma).max())
    # gradient of a scalar functional through both L and V must be finite
    loss = (L.sqrt().sum() + (V * V).sum())
    g = jt.grad(loss, sigma)
    g_np = g.numpy()
    finite = np.isfinite(g_np).all()
    status = 'OK ' if (rec_err < 1e-4 and finite) else 'FAIL'
    print(f'[{status}] {name}: rec_err={rec_err:.2e} grad_finite={finite} '
          f'grad_max={np.abs(g_np).max():.3e}')
    assert rec_err < 1e-4, (name, rec_err)
    assert finite, (name, 'NaN/Inf gradient')


def main():
    # exact square: sigma = a*I (the plan's w==h landmine)
    check('exact w==h (aI)', np.stack([np.diag([64.0, 64.0]),
                                       np.diag([1024.0, 1024.0])]))
    # near-degenerate: eigenvalue gap 1e-6
    check('near-degenerate', np.stack([
        np.array([[100.0, 1e-7], [1e-7, 100.000001]])]))
    # generic anisotropic + rotated (sanity)
    a, b, c = 200.0, 40.0, 80.0
    check('generic', np.array([[[a, b], [b, c]]]))
    # tiny values (downsampled sigma / D**2 territory)
    check('tiny', np.array([[[1e-3, 0.0], [0.0, 1e-3]]]))
    print('PASS: eigh_2x2 degenerate cases safe')


if __name__ == '__main__':
    main()
