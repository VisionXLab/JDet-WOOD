"""2×2 线性代数闭式解（可导），替代 Jittor 缺失的 torch.linalg.eigh / solve / diag_embed。

Point2RBox-v2/v3 的 Gaussian losses 作用于 2×2 对称协方差矩阵：
    eigh_2x2   : 对称 2×2 batch 特征分解，特征值升序（对齐 torch.linalg.eigh）。
                 特征向量存在符号任意性——与 torch 的 parity 只保证
                 V @ diag(L) @ V^T 重建一致、|<v_jt, v_torch>| = 1。
    solve_2x2  : 2×2 batch 线性方程组 A X = B（伴随矩阵 / det）。
    diag_embed_2x2 : (..., 2) → (..., 2, 2) 对角阵。

退化保护（w==h 的正方形目标 → b≈0 且 a≈c，判别式为 0，naive sqrt 反向 NaN）：
    disc = sqrt(clamp(delta² + b², EPS))，EPS=1e-24；
    特征向量按 delta 符号选伴随列，模长过小时回退单位阵。
"""
import jittor as jt

EPS = 1e-24  # 必须远小于最小可能特征值的平方（面积趋 0 的框 ~1e-6）；|delta|/disc<=1 保证梯度有界


def diag_embed_2x2(d):
    """(..., 2) -> (..., 2, 2)，对角线为 d。"""
    zero = jt.zeros_like(d[..., 0])
    m = jt.stack([d[..., 0], zero, zero, d[..., 1]], dim=-1)
    return m.reshape(d.shape[:-1] + (2, 2))


def eigh_2x2(sigma):
    """对称 2×2 batch 特征分解，返回 (L, V)。

    L: (..., 2) 特征值升序；V: (..., 2, 2)，V[..., :, i] 是 L[..., i] 的特征向量。
    与 torch.linalg.eigh 的约定一致（升序 + 列向量）。
    """
    a = sigma[..., 0, 0]
    # 对称化读取：保证反向梯度在 [0,1]/[1,0] 两个位置对称分摊（与 torch.linalg.eigh 一致）
    b = (sigma[..., 0, 1] + sigma[..., 1, 0]) / 2
    c = sigma[..., 1, 1]

    mean = (a + c) / 2
    delta = (a - c) / 2
    disc = jt.sqrt((delta * delta + b * b).clamp(EPS))

    l1 = mean - disc  # 小特征值
    l2 = mean + disc  # 大特征值
    L = jt.stack([l1, l2], dim=-1)

    # 大特征值 λ2 的特征向量：(A - λ2 I) v = 0 的两个伴随列解
    #   cand_a = [disc + delta, b]   （delta >= 0，即 a >= c 时数值稳定）
    #   cand_b = [b, disc - delta]   （delta <  0，即 a <  c 时数值稳定）
    use_a = (delta >= 0).unsqueeze(-1)
    v2 = jt.where(use_a,
                  jt.stack([disc + delta, b], dim=-1),
                  jt.stack([b, disc - delta], dim=-1))
    norm = jt.sqrt((v2 * v2).sum(-1, keepdims=True).clamp(EPS))
    # 完全退化（a==c 且 b==0，各向同性）：两个候选都是零向量 → 回退 [1, 0]
    degenerate = norm < 1e-6
    e1 = jt.stack([jt.ones_like(b), jt.zeros_like(b)], dim=-1)
    v2 = jt.where(degenerate, e1, v2 / norm)
    # 小特征值向量 = v2 逆时针旋转 90°，保证正交且 det(V)=1
    v1 = jt.stack([-v2[..., 1], v2[..., 0]], dim=-1)

    V = jt.stack([v1, v2], dim=-1)  # 列向量：V[..., :, 0]=v1, V[..., :, 1]=v2
    return L, V


def inv_2x2(m):
    """2×2 batch 逆矩阵：adj(A) / det(A)，det 用带符号的 eps 保护。"""
    a = m[..., 0, 0]
    b = m[..., 0, 1]
    c = m[..., 1, 0]
    d = m[..., 1, 1]
    det = a * d - b * c
    det = det + jt.where(det >= 0, jt.float32(EPS), jt.float32(-EPS))  # EPS=1e-24 不影响 1e-12 量级的真实 det
    adj = jt.stack([d, -b, -c, a], dim=-1).reshape(m.shape[:-2] + (2, 2))
    return adj / det.unsqueeze(-1).unsqueeze(-1)


def solve_2x2(A, B):
    """解 A X = B。A: (..., 2, 2)，B: (..., 2, k) 或 (..., 2)。

    GPU 的 cublas_batched_matmul 不做 batch 广播（CPU 会），batch 维不一致时
    显式 expand（系数矩阵的 batch 需为 1 或与右端项相同）。"""
    squeeze = B.ndim == A.ndim - 1
    if squeeze:
        B = B.unsqueeze(-1)
    inv = inv_2x2(A)
    if tuple(inv.shape[:-2]) != tuple(B.shape[:-2]):
        inv = inv.expand(tuple(B.shape[:-2]) + (2, 2))
    out = jt.matmul(inv, B)
    return out.squeeze(-1) if squeeze else out
