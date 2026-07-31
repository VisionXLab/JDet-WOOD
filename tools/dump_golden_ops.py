"""dump_golden_ops.py — p2r-torch 环境 dump M2.5 复用件的 golden（L1 层）。

覆盖：
  1. PSCCoder(le90, dual_freq=False, num_step=3, thr_mod=0) encode/decode（官方 config 参数）
  2. mmcv.ops.box_iou_rotated（le90 弧度框）
  3. mmcv.ops.nms_rotated（iou_threshold=0.1，官方 test_cfg 值）
  4. mmcv.ops.RoIAlignRotated(out_size=7&49, spatial_scale, clockwise=True)
     —— B 问的角度约定，EdgeLoss 依赖 out_size=49

产出 tests/parity/golden/ops_misc.npz
"""
import os
import sys

import numpy as np
import torch

REPO = os.environ.get('P2R_REF', '/root/ref/Point2RBox-v3')
sys.path.insert(0, REPO)

OUT = os.path.join(os.path.dirname(__file__), '..', 'tests', 'parity', 'golden',
                   'ops_misc.npz')


def make_rboxes(n, seed):
    rng = np.random.RandomState(seed)
    xy = rng.uniform(50, 200, (n, 2))
    wh = rng.uniform(8, 80, (n, 2))
    r = rng.uniform(-np.pi / 2, np.pi / 2, (n, 1))
    b = np.concatenate([xy, wh, r], 1).astype(np.float32)
    b[0, 2] = b[0, 3] = 32.0   # 正方形
    b[1, 4] = np.pi / 2        # 角度边界
    b[2, 4] = -np.pi / 2
    return b


def main():
    from mmcv.ops import box_iou_rotated, nms_rotated, RoIAlignRotated
    from mmrotate.models.task_modules.coders.angle_coder import PSCCoder

    out = {}

    # ---- 1. PSCCoder（官方 config：dual_freq=False, num_step=3, thr_mod=0）----
    coder = PSCCoder(angle_version='le90', dual_freq=False, num_step=3, thr_mod=0)
    angles = np.linspace(-np.pi / 2, np.pi / 2, 61).astype(np.float32).reshape(-1, 1)
    enc = coder.encode(torch.tensor(angles))
    dec = coder.decode(torch.tensor(enc.numpy()), keepdim=True)
    out['psc_angles'] = angles
    out['psc_encoded'] = enc.numpy()
    out['psc_decoded'] = dec.numpy()
    assert coder.encode_size == 3

    # ---- 2. box_iou_rotated ----
    b1, b2 = make_rboxes(12, 1), make_rboxes(10, 2)
    iou = box_iou_rotated(torch.tensor(b1), torch.tensor(b2))
    out['iou_boxes1'], out['iou_boxes2'], out['iou'] = b1, b2, iou.numpy()

    # ---- 3. nms_rotated（官方 test_cfg: iou_threshold=0.1）----
    rng = np.random.RandomState(3)
    base = make_rboxes(8, 4)
    jitter = base.copy()
    jitter[:, :2] += rng.uniform(-4, 4, (8, 2))  # 制造高重叠对
    boxes = np.concatenate([base, jitter], 0).astype(np.float32)
    scores = rng.uniform(0.1, 1.0, (16,)).astype(np.float32)
    dets, keep = nms_rotated(torch.tensor(boxes), torch.tensor(scores), 0.1)
    out['nms_boxes'], out['nms_scores'] = boxes, scores
    out['nms_keep'] = keep.numpy()
    out['nms_dets'] = dets.numpy()

    # ---- 4. RoIAlignRotated（clockwise=True，mmrotate 全系用法）----
    rng = np.random.RandomState(5)
    feat = rng.randn(1, 4, 64, 64).astype(np.float32)
    # rois: (batch_idx, cx, cy, w, h, angle)，含 0 角、正角、负角、正方形
    rois = np.array([
        [0, 32, 32, 24, 16, 0.0],
        [0, 32, 32, 24, 16, 0.7854],
        [0, 20, 40, 30, 12, -0.5],
        [0, 45, 20, 20, 20, 1.2],
    ], dtype=np.float32)
    for size_key, osize in [('7', 7), ('49', 49)]:
        op = RoIAlignRotated(osize, spatial_scale=1.0, sampling_ratio=2,
                             clockwise=True)
        res = op(torch.tensor(feat), torch.tensor(rois))
        out[f'ra_out_{size_key}'] = res.numpy()
    out['ra_feat'], out['ra_rois'] = feat, rois

    np.savez(OUT, **out)
    print('saved ->', OUT)
    for k, v in out.items():
        print(f'  {k}: {np.asarray(v).shape}')


if __name__ == '__main__':
    main()
