"""dump_golden_gdloss.py — p2r-torch 环境 dump GDLoss(gwd) 的前向+梯度 golden（M2.5 核对）。

输入覆盖 L1 要求的退化用例：一般框、w==h 正方形、角度 ±π/2 边界、极小面积、大长宽比。
产出 tests/parity/golden/gdloss_gwd.npz。
"""
import os
import sys

import numpy as np
import torch

REPO = os.environ.get('P2R_REF', '/root/ref/Point2RBox-v3')
sys.path.insert(0, REPO)

OUT = os.path.join(os.path.dirname(__file__), '..', 'tests', 'parity', 'golden',
                   'gdloss_gwd.npz')


def make_boxes():
    rng = np.random.RandomState(42)
    n = 16
    xy = rng.uniform(100, 900, (n, 2))
    wh = rng.uniform(8, 200, (n, 2))
    r = rng.uniform(-np.pi / 2, np.pi / 2, (n, 1))
    pred = np.concatenate([xy, wh, r], axis=1)
    # 退化用例
    pred[0, 2] = pred[0, 3] = 64.0          # w == h 正方形
    pred[1, 4] = np.pi / 2                  # 角度上边界
    pred[2, 4] = -np.pi / 2                 # 角度下边界
    pred[3, 2:4] = [1e-3, 1e-3]             # 极小面积
    pred[4, 2:4] = [400.0, 4.0]             # 大长宽比
    target = pred + rng.uniform(-8, 8, pred.shape) * np.array([1, 1, 0.5, 0.5, 0.02])
    target[:, 2:4] = np.abs(target[:, 2:4]) + 1e-3
    return pred.astype(np.float32), target.astype(np.float32)


def main():
    from mmrotate.models.losses.gaussian_dist_loss import (GDLoss,
                                                           xy_wh_r_2_xy_sigma)

    pred_np, target_np = make_boxes()

    # 1) xy_wh_r_2_xy_sigma 前向 + 对 pred 的梯度
    pred = torch.tensor(pred_np, requires_grad=True)
    xy, sigma = xy_wh_r_2_xy_sigma(pred)
    (xy.sum() + sigma.sum()).backward()
    sigma_grad = pred.grad.detach().numpy().copy()

    # 2) GDLoss(gwd, loss_weight=5.0)（官方 config 用法）前向 + 梯度
    loss_fn = GDLoss(loss_type='gwd', loss_weight=5.0)
    pred2 = torch.tensor(pred_np, requires_grad=True)
    loss = loss_fn(pred2, torch.tensor(target_np))
    loss.backward()

    # 3) 带 weight + avg_factor 的路径（head 的实际调用形态）
    rng = np.random.RandomState(7)
    weight_np = rng.uniform(0, 1, (len(pred_np),)).astype(np.float32)
    weight_np[5:8] = 0.0  # 部分零权重
    avg_factor = 9.0
    pred3 = torch.tensor(pred_np, requires_grad=True)
    loss_w = loss_fn(pred3, torch.tensor(target_np),
                     weight=torch.tensor(weight_np), avg_factor=avg_factor)
    loss_w.backward()

    np.savez(OUT,
             pred=pred_np, target=target_np,
             xy=xy.detach().numpy(), sigma=sigma.detach().numpy(),
             xy_sigma_grad=sigma_grad,
             gwd_loss=loss.item(), gwd_grad=pred2.grad.detach().numpy(),
             weight=weight_np, avg_factor=avg_factor,
             gwd_loss_weighted=loss_w.item(),
             gwd_grad_weighted=pred3.grad.detach().numpy())
    print('saved ->', OUT)
    print('gwd_loss =', loss.item(), '| weighted =', loss_w.item())
    print('grad norm =', np.linalg.norm(pred2.grad.numpy()))


if __name__ == '__main__':
    main()
