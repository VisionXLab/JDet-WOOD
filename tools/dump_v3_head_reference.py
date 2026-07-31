"""Golden reference for Point2RBoxV3Head stage-1 parity (forward + targets).

Run in p2r-torch:
    PYTHONPATH=/root/ref/Point2RBox-v3 python tools/dump_v3_head_reference.py \
        --out tests/data/v3_head_reference.npz

Dumps: full state_dict, 5-level random FPN feats (B=2), forward outputs,
grid priors, and get_targets golden for synthetic gts (square / rotated /
overlapping / empty-image cases) with (N,4) bids.
"""
import argparse
import os

import numpy as np
import torch


BBOX_HEAD_CFG = dict(
    num_classes=15,
    in_channels=256,
    feat_channels=256,
    strides=[8, 16, 32, 64, 128],
    use_adaptive_scale=False,
    edge_loss_start_epoch=6,
    joint_angle_start_epoch=1,
    voronoi_type='standard',
    voronoi_thres=dict(
        default=[0.994, 0.005],
        override=(([2, 11], [0.999, 0.6]), ([7, 8, 10, 14], [0.95, 0.005]))),
    square_cls=[1, 9, 11],
    edge_loss_cls=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13],
    post_process={11: 1.2},
    angle_coder=dict(
        type='PSCCoder', angle_version='le90', dual_freq=False, num_step=3,
        thr_mod=0),
    loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0,
                  alpha=0.25, loss_weight=1.0),
    loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0),
    loss_overlap=dict(type='GaussianOverlapLoss', loss_weight=10.0, lamb=0),
    loss_voronoi=dict(type='VoronoiWatershedLoss', loss_weight=5.0),
    loss_bbox_edg=dict(type='EdgeLoss', loss_weight=0.3),
    loss_ss=dict(type='Point2RBoxV2ConsistencyLoss', loss_weight=1.0),
)

# synthetic gts: img0 has 4 boxes, img1 empty
GT0_BBOXES = np.array([
    [100.0, 120.0, 40.0, 40.0, 0.0],        # exact square
    [60.0, 60.0, 80.0, 30.0, 0.6],          # rotated rect
    [100.0, 120.0, 200.0, 160.0, -0.4],     # large, overlaps square
    [200.0, 40.0, 24.0, 12.0, 1.2],         # small far box
], np.float32)
GT0_LABELS = np.array([9, 4, 3, 5], np.int64)
GT0_BIDS = np.array([[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 3], [0, 1, 0, 4]],
                    np.int64)
GT1_BBOXES = np.array([
    [30.0, 200.0, 50.0, 20.0, -1.0],
    [180.0, 180.0, 60.0, 60.0, 0.3],
], np.float32)
GT1_LABELS = np.array([0, 11], np.int64)
GT1_BIDS = np.array([[1, 0, 0, 5], [1, 0, 1, 6]], np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    args = p.parse_args()

    from mmengine.registry import init_default_scope
    init_default_scope('mmrotate')
    from mmrotate.models.dense_heads.point2rbox_v3_head import Point2RBoxV3Head
    from mmengine.structures import InstanceData
    from mmrotate.structures.bbox import RotatedBoxes

    torch.manual_seed(0)
    head = Point2RBoxV3Head(**BBOX_HEAD_CFG)
    head.eval()

    arrays = {}
    for k, v in head.state_dict().items():
        arrays[f'sd::{k}'] = v.detach().cpu().numpy()

    # forward on small feature maps (input 256x256)
    sizes = [(32, 32), (16, 16), (8, 8), (4, 4), (2, 2)]
    feats = [torch.randn(2, 256, h, w) for h, w in sizes]
    for i, f in enumerate(feats):
        arrays[f'feat_{i}'] = f.numpy()
    with torch.no_grad():
        cls_scores, bbox_preds, angle_preds = head(feats)
    for i in range(5):
        arrays[f'cls_{i}'] = cls_scores[i].numpy()
        arrays[f'bbox_{i}'] = bbox_preds[i].numpy()
        arrays[f'angle_{i}'] = angle_preds[i].numpy()

    # priors + targets
    priors = head.prior_generator.grid_priors(sizes, dtype=torch.float32,
                                              device='cpu')
    for i, pr in enumerate(priors):
        arrays[f'priors_{i}'] = pr.numpy()

    gi0 = InstanceData()
    gi0.bboxes = RotatedBoxes(torch.from_numpy(GT0_BBOXES))
    gi0.labels = torch.from_numpy(GT0_LABELS)
    gi0.bids = torch.from_numpy(GT0_BIDS)
    # NOTE ref quirk: an empty-gt image makes ref get_targets CRASH (its
    # empty branch returns (N,4) while non-empty returns (N,5) -> cat fails).
    # Dead path in practice (filter_empty_gt=True). Golden uses 2 non-empty
    # images; the port keeps the same faithful (N,4) empty branch.
    gi1 = InstanceData()
    gi1.bboxes = RotatedBoxes(torch.from_numpy(GT1_BBOXES))
    gi1.labels = torch.from_numpy(GT1_LABELS)
    gi1.bids = torch.from_numpy(GT1_BIDS)

    labels, bbox_targets, bid_targets = head.get_targets(priors, [gi0, gi1])
    for i in range(5):
        arrays[f'tgt_labels_{i}'] = labels[i].numpy()
        arrays[f'tgt_bbox_{i}'] = bbox_targets[i].numpy()
        arrays[f'tgt_bid_{i}'] = bid_targets[i].numpy()

    arrays['gt0_bboxes'] = GT0_BBOXES
    arrays['gt1_bboxes'] = GT1_BBOXES
    arrays['gt1_labels'] = GT1_LABELS
    arrays['gt1_bids'] = GT1_BIDS
    arrays['gt0_labels'] = GT0_LABELS
    arrays['gt0_bids'] = GT0_BIDS

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f'saved {len(arrays)} arrays -> {args.out}')


if __name__ == '__main__':
    main()
