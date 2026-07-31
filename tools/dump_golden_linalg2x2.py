"""dump_golden_linalg2x2.py — torch.linalg.{eigh,solve} 的 golden（M3 地基）。

用例矩阵（L1 必须覆盖的退化情形，PLAN §7/§8）：
    一般对称正定、w==h 正方形（b=0 且 a==c）、near-isotropic、
    b=0 但 a!=c、极小特征值（面积→0）、大条件数。
梯度：eigh 的 L 对输入的梯度（V 的梯度不比——特征向量符号任意，且
v2 的使用场景里 V 均在 detach 分支）；solve 的输出梯度。
"""
import os

import numpy as np
import torch

OUT = os.path.join(os.path.dirname(__file__), '..', 'tests', 'parity', 'golden',
                   'linalg2x2.npz')


def make_sigmas():
    rng = np.random.RandomState(11)
    mats = []
    for _ in range(10):  # 一般情形：R diag(s²) R^T
        w, h = rng.uniform(4, 100, 2)
        t = rng.uniform(-np.pi, np.pi)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        mats.append(R @ np.diag([(w / 2) ** 2, (h / 2) ** 2]) @ R.T)
    mats.append(np.eye(2) * 256.0)                     # w==h 完全各向同性
    mats.append(np.eye(2) * 256.0 + np.array([[0, 1e-6], [1e-6, 0]]))  # 近各向同性
    mats.append(np.diag([400.0, 25.0]))                # b=0, a>c
    mats.append(np.diag([25.0, 400.0]))                # b=0, a<c
    mats.append(np.eye(2) * 1e-6)                      # 面积趋 0
    mats.append(np.diag([1e6, 1e-2]))                  # 大条件数
    return np.stack(mats).astype(np.float32)


def main():
    sig_np = make_sigmas()
    sig = torch.tensor(sig_np, requires_grad=True)
    L, V = torch.linalg.eigh(sig)
    # L 的梯度（加权和，权重固定）
    wgt = torch.tensor(np.linspace(0.5, 2.0, L.numel()).reshape(L.shape).astype(np.float32))
    (L * wgt).sum().backward()
    L_grad = sig.grad.detach().numpy().copy()

    rng = np.random.RandomState(12)
    b_np = rng.uniform(-5, 5, (len(sig_np), 2, 1)).astype(np.float32)
    A = torch.tensor(sig_np, requires_grad=True)
    B = torch.tensor(b_np)
    X = torch.linalg.solve(A, B)
    X.sum().backward()

    np.savez(OUT, sigma=sig_np,
             eigvals=L.detach().numpy(), eigvecs=V.detach().numpy(),
             eig_wgt=wgt.numpy(), eigval_grad=L_grad,
             solve_b=b_np, solve_x=X.detach().numpy(),
             solve_grad=A.grad.detach().numpy())
    print('saved ->', OUT, '| cases:', len(sig_np))


if __name__ == '__main__':
    main()
