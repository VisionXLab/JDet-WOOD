"""dump_golden_p2rv2_loss.py — p2r-torch 环境 dump Point2RBox-v2 四个 loss 的 golden。

按 M3 移植顺序逐段追加：
    §1 GaussianOverlapLoss（含 gwd_sigma_loss / bhattacharyya）
    §2 VoronoiWatershedLoss（后续追加）
    §3 EdgeLoss（后续追加）
    §4 Point2RBoxV2ConsistencyLoss（后续追加）

随机数策略（PLAN §7）：输入全部由固定 seed 的 numpy 生成并存进 npz，
两边从同一 npz 读，不依赖框架内 RNG。
"""
import os
import sys

import numpy as np
import torch

REPO = os.environ.get('P2R_REF', '/root/ref/Point2RBox-v3')
sys.path.insert(0, REPO)

OUT = os.path.join(os.path.dirname(__file__), '..', 'tests', 'parity', 'golden',
                   'p2rv2_loss.npz')


def make_gaussians(n, seed):
    """mu (N,2) + sigma (N,2,2)，含 w==h 正方形与部分高重叠对。"""
    rng = np.random.RandomState(seed)
    mu = rng.uniform(100, 900, (n, 2)).astype(np.float32)
    mu[1] = mu[0] + rng.uniform(-6, 6, 2)   # 制造高重叠对
    mu[3] = mu[2] + rng.uniform(-4, 4, 2)
    sigmas = []
    for i in range(n):
        w, h = rng.uniform(8, 120, 2)
        if i == 0:
            w = h = 48.0                     # w==h 正方形
        t = rng.uniform(-np.pi / 2, np.pi / 2)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        sigmas.append(R @ np.diag([(w / 2) ** 2, (h / 2) ** 2]) @ R.T)
    return mu, np.stack(sigmas).astype(np.float32)


def main():
    from mmrotate.models.losses.point2rbox_v2_loss import (
        GaussianOverlapLoss, gwd_sigma_loss)

    out = {}

    # ---- §1 GaussianOverlapLoss（官方 config: loss_weight=10.0, lamb=0；
    #        另测默认 lamb=1e-4 路径）----
    mu_np, sigma_np = make_gaussians(12, 21)
    out['gol_mu'], out['gol_sigma'] = mu_np, sigma_np
    for tag, kwargs in [('cfg', dict(loss_weight=10.0, lamb=0)),
                        ('lamb', dict(loss_weight=1.0, lamb=1e-4))]:
        loss_fn = GaussianOverlapLoss(**kwargs)
        mu = torch.tensor(mu_np, requires_grad=True)
        sigma = torch.tensor(sigma_np, requires_grad=True)
        loss = loss_fn((mu, sigma))
        loss.backward()
        out[f'gol_{tag}_loss'] = loss.item()
        out[f'gol_{tag}_mu_grad'] = mu.grad.detach().numpy()
        out[f'gol_{tag}_sigma_grad'] = sigma.grad.detach().numpy()

    # gwd_sigma_loss 单测（Voronoi 依赖）
    _, sig_a = make_gaussians(8, 22)
    _, sig_b = make_gaussians(8, 23)
    sa = torch.tensor(sig_a, requires_grad=True)
    l = gwd_sigma_loss(sa, torch.tensor(sig_b), reduction='mean')
    l.backward()
    out['gws_a'], out['gws_b'] = sig_a, sig_b
    out['gws_loss'] = l.item()
    out['gws_grad'] = sa.grad.detach().numpy()

    # ---- §2 VoronoiWatershedLoss（官方路径：voronoi='standard', w=5.0）----
    from mmrotate.models.losses.point2rbox_v2_loss import (
        VoronoiWatershedLoss, EdgeLoss, Point2RBoxV2ConsistencyLoss)

    rng = np.random.RandomState(31)
    Hh = Ww = 128
    image_np = rng.uniform(0, 255, (3, Hh, Ww)).astype(np.float32)
    # 平滑一点，让 watershed 行为不至于纯噪声
    import cv2
    for c in range(3):
        image_np[c] = cv2.GaussianBlur(image_np[c], (9, 9), 3)
    mu_v, sigma_v = make_gaussians(6, 32)
    mu_v = (mu_v / 900 * 100 + 14).astype(np.float32)   # 落进 128×128
    sigma_v = (sigma_v / 16).astype(np.float32)
    label_v = np.array([0, 1, 9, 11, 2, 7], dtype=np.int64)
    pos_thres_v = np.full((15,), 0.994, dtype=np.float32)
    neg_thres_v = np.full((15,), 0.005, dtype=np.float32)
    pos_thres_v[[2, 11]] = 0.999
    neg_thres_v[[2, 11]] = 0.6
    pos_thres_v[[7, 8, 10, 14]] = 0.95
    neg_thres_v[[7, 8, 10, 14]] = 0.005
    out['vws_image'], out['vws_mu'], out['vws_sigma'] = image_np, mu_v, sigma_v
    out['vws_label'] = label_v
    out['vws_pos'], out['vws_neg'] = pos_thres_v, neg_thres_v

    loss_fn = VoronoiWatershedLoss(loss_weight=5.0)
    mu_t = torch.tensor(mu_v)  # mu 只进 detach 分支（L_target 整体 detach），无梯度
    sigma_t = torch.tensor(sigma_v, requires_grad=True)
    loss = loss_fn((mu_t, sigma_t), torch.tensor(label_v), torch.tensor(image_np),
                   torch.tensor(pos_thres_v), torch.tensor(neg_thres_v),
                   voronoi='standard')
    loss.backward()
    out['vws_loss'] = loss.item()
    out['vws_sigma_grad'] = sigma_t.grad.detach().numpy()
    out['vws_markers'] = loss_fn.vis[1].detach().numpy()

    # ---- §3 EdgeLoss（官方 w=0.3）----
    rng = np.random.RandomState(33)
    edge_np = rng.uniform(0, 1, (1, 1, 256, 256)).astype(np.float32)
    for _ in range(2):
        edge_np[0, 0] = cv2.GaussianBlur(edge_np[0, 0], (7, 7), 2)
    boxes1 = np.stack([
        rng.uniform(60, 200, 8), rng.uniform(60, 200, 8),
        rng.uniform(20, 60, 8), rng.uniform(12, 40, 8),
        rng.uniform(-np.pi / 2, np.pi / 2, 8)], 1).astype(np.float32)
    boxes1[0, 2] = boxes1[0, 3] = 32.0
    out['edge_map'], out['edge_boxes'] = edge_np, boxes1
    loss_fn = EdgeLoss(loss_weight=0.3)
    b1 = torch.tensor(boxes1, requires_grad=True)
    loss = loss_fn([b1], torch.tensor(edge_np))
    loss.backward()
    out['edge_loss'] = loss.item()
    out['edge_grad'] = b1.grad.detach().numpy()

    # ---- §4 Point2RBoxV2ConsistencyLoss（rot / flp / sca 三条路径）----
    rng = np.random.RandomState(34)
    n = 10
    _, gaus_o = make_gaussians(n, 35)
    gaus_o = np.linalg.cholesky(gaus_o + 1e-3 * np.eye(2)).astype(np.float32)  # 保持可乘方
    _, gaus_t = make_gaussians(n, 36)
    gaus_t = np.linalg.cholesky(gaus_t + 1e-3 * np.eye(2)).astype(np.float32)
    ang_o = rng.uniform(-np.pi / 2, np.pi / 2, n).astype(np.float32)
    ang_t = rng.uniform(-np.pi / 2, np.pi / 2, n).astype(np.float32)
    sq = np.zeros(n, dtype=bool)
    sq[[1, 4]] = True
    out['con_gaus_o'], out['con_gaus_t'] = gaus_o, gaus_t
    out['con_ang_o'], out['con_ang_t'], out['con_sq'] = ang_o, ang_t, sq
    loss_fn = Point2RBoxV2ConsistencyLoss(loss_weight=1.0)
    for aug_type, aug_val in [('rot', 0.6), ('flp', 0.0), ('sca', 1.3)]:
        go = torch.tensor(gaus_o, requires_grad=True)
        gt_ = torch.tensor(gaus_t, requires_grad=True)
        ao = torch.tensor(ang_o, requires_grad=True)
        at = torch.tensor(ang_t, requires_grad=True)
        loss = loss_fn((go, ao), (gt_, at), torch.tensor(sq), aug_type, aug_val)
        loss.backward()
        out[f'con_{aug_type}_loss'] = loss.item()
        out[f'con_{aug_type}_go_grad'] = go.grad.detach().numpy()
        out[f'con_{aug_type}_ao_grad'] = ao.grad.detach().numpy()

    np.savez(OUT, **out)
    print('saved ->', OUT)
    print('gol_cfg_loss =', out['gol_cfg_loss'], '| gol_lamb_loss =', out['gol_lamb_loss'])
    print('gws_loss =', out['gws_loss'])
    print('vws_loss =', out['vws_loss'])
    print('edge_loss =', out['edge_loss'])
    print('con losses:', out['con_rot_loss'], out['con_flp_loss'], out['con_sca_loss'])


if __name__ == '__main__':
    main()
