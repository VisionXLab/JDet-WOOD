"""dump_golden_head.py — torch 侧 dump Point2RBoxV2Head 的整体 parity golden（M4 验收）。

固定权重 + 固定输入（feat/gt/bids/images）→ loss_dict 各项 + forward 输出。
epoch=1：edge/copy-paste 未启动（这两条路径的组件级 parity 已在 L2 覆盖），
关注 cls/bbox/vor/ovl/ss 五项主 loss 的端到端一致性。

用法：cd /root/ref/Point2RBox-v3 && PYTHONPATH=. python <this>
"""
import os
import sys

import numpy as np
import torch

REPO = os.environ.get('P2R_REF', '/root/ref/Point2RBox-v3')
sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'tests', 'parity', 'golden', 'head_parity.npz')

HEAD_CFG = dict(
    num_classes=15, in_channels=128, feat_channels=128, strides=[8],
    square_cls=[1, 9, 11],
    edge_loss_cls=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13],
    post_process={11: 1.2},
    voronoi_type='standard',
    voronoi_thres=dict(default=[0.994, 0.005],
                       override=(([2, 11], [0.999, 0.6]),
                                 ([7, 8, 10, 14], [0.95, 0.005]))),
    loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0),
    loss_overlap=dict(type='GaussianOverlapLoss', loss_weight=10.0, lamb=0),
)


def make_inputs():
    rng = np.random.RandomState(51)
    B, C, H, W = 4, 128, 32, 32  # dual-stream 后 batch=4（原 2 + aug 2）
    feat = rng.randn(B, C, H, W).astype(np.float32) * 0.5
    images = rng.uniform(-2, 2, (B, 3, 256, 256)).astype(np.float32)
    gts = []
    offset = 1
    for i in range(2):
        n = 5
        rb = np.stack([
            rng.uniform(40, 210, n), rng.uniform(40, 210, n),
            rng.uniform(16, 56, n), rng.uniform(12, 40, n),
            rng.uniform(-1.4, 1.4, n)], 1).astype(np.float32)
        rb[0, 4] = 0.0
        labels = rng.randint(0, 15, n).astype(np.int64)
        bids = np.zeros((n, 4), dtype=np.int64)
        bids[:, 0] = i
        bids[:, 3] = np.arange(n) + offset
        bids[2, 1] = 1  # 一个 syn 实例（copy-paste 路径的 target 分支）
        offset += n
        gts.append((rb, labels, bids))
    # aug 批：同实例 view=1（rot 0.5）
    rot = 0.5
    for i in range(2):
        rb, labels, bids = gts[i]
        rb2 = rb.copy()
        c, s = np.cos(rot), np.sin(rot)
        ctr = np.float32([128, 128])
        xy = (rb2[:, :2] - ctr) @ np.float32([[c, -s], [s, c]]).T + ctr
        rb2 = np.concatenate([xy, rb2[:, 2:4], rb2[:, 4:5] + rot], 1).astype(np.float32)
        bids2 = bids.copy()
        bids2[:, 0] = i + 2
        bids2[:, 2] = 1
        gts.append((rb2, labels, bids2))
    return feat, images, gts, ('rot', rot)


def main():
    from mmrotate.utils import register_all_modules
    register_all_modules(True)
    from mmrotate.models.dense_heads.point2rbox_v2_head import Point2RBoxV2Head
    from mmrotate.structures.bbox import RotatedBoxes
    from mmengine.structures import InstanceData

    torch.manual_seed(3)
    head = Point2RBoxV2Head(**HEAD_CFG)
    head.epoch = 1
    head.train()

    feat_np, images_np, gts, ss = make_inputs()
    head.images = torch.tensor(images_np)

    batch_gt, metas = [], []
    for rb, labels, bids in gts:
        gi = InstanceData()
        gi.bboxes = RotatedBoxes(torch.tensor(rb))
        gi.labels = torch.tensor(labels)
        gi.bids = torch.tensor(bids)
        batch_gt.append(gi)
    metas = [dict(ss=ss)] * 4

    feat = torch.tensor(feat_np, requires_grad=True)
    outs = head((feat,))
    losses = head.loss_by_feat(*outs, batch_gt, metas)
    total = sum(v.sum() for v in losses.values())
    total.backward()

    out = dict(feat=feat_np, images=images_np, ss_val=ss[1],
               feat_grad=feat.grad.detach().numpy())
    for i, (rb, labels, bids) in enumerate(gts):
        out[f'gt{i}_rb'], out[f'gt{i}_lb'], out[f'gt{i}_bid'] = rb, labels, bids
    for k, v in losses.items():
        out[f'loss_{k}'] = float(v.sum().item())
    sd = {k: v.detach().numpy() for k, v in head.state_dict().items()}
    np.savez(OUT, **out, **{f'w__{k}': v for k, v in sd.items()})
    print('saved ->', OUT)
    for k, v in losses.items():
        print(' ', k, '=', float(v.sum().item()))


if __name__ == '__main__':
    main()
