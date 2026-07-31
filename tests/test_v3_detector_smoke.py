"""Build + forward_train smoke for the Point2RBoxV3 detector (M5 stage).

Runs the full train-path plumbing on synthetic input at epoch 0:
dual-stream assembly -> voronoi pseudo-label generation (gaussian-orientation,
via head.predict assist) -> bids fill-back -> full 6-component loss dict (stage-2 landed).


Run in p2r-jittor:  CUDA_VISIBLE_DEVICES=1 python tests/test_v3_detector_smoke.py
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, ROOT)  # for the root-level third_parties package
os.chdir(ROOT)

import jittor as jt  # noqa: E402
from jdet.utils.registry import MODELS, build_from_cfg  # noqa: E402
import jdet.models  # noqa: F401,E402  (trigger registrations)

MODEL_CFG = dict(
    type='Point2RBoxV3',
    backbone=dict(
        type='Resnet50',
        frozen_stages=1,
        norm_eval=True,
        return_stages=['layer1', 'layer2', 'layer3', 'layer4'],
        pretrained=False),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
        num_outs=5),
    bbox_head=dict(
        type='Point2RBoxV3Head',
        num_classes=15,
        in_channels=256,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        use_adaptive_scale=False,
        edge_loss_start_epoch=6,
        joint_angle_start_epoch=1,
        voronoi_type='gaussian-orientation',
        voronoi_thres=dict(
            default=[0.994, 0.005],
            override=(([2, 11], [0.999, 0.6]), ([7, 8, 10, 14], [0.95, 0.005]))),
        square_cls=[1, 9, 11],
        edge_loss_cls=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13],
        post_process={11: 1.2},
        angle_coder=dict(
            type='PSCCoder',
            angle_version='le90',
            dual_freq=False,
            num_step=3,
            thr_mod=0),
        loss_cfg_placeholder=None),
)


def make_targets(n_imgs, n_gt=4, seed=0):
    rng = np.random.RandomState(seed)
    targets = []
    for i in range(n_imgs):
        cx = rng.uniform(200, 800, n_gt)
        cy = rng.uniform(200, 800, n_gt)
        wh = rng.uniform(30, 80, (n_gt, 2))
        ang = rng.uniform(-1.5, 1.5, n_gt)
        rb = np.stack([cx, cy, wh[:, 0], wh[:, 1], ang], 1).astype(np.float32)
        targets.append(dict(
            rboxes=jt.array(rb),
            labels=jt.array(rng.randint(0, 15, n_gt).astype(np.int32)),
        ))
    return targets


def main():
    cfg = dict(MODEL_CFG)
    cfg['bbox_head'] = {k: v for k, v in cfg['bbox_head'].items()
                        if k != 'loss_cfg_placeholder'}
    model = build_from_cfg(cfg, MODELS)
    model.train()
    model.set_epoch(0)

    images = jt.array(np.random.RandomState(1).randn(
        2, 3, 1024, 1024).astype(np.float32))
    targets = make_targets(2)

    # stage-2 落地后 loss 已实现：全链路应产出 6 分量有限 loss dict
    losses = model.forward_train(images, targets)
    expect = {'loss_cls', 'loss_bbox', 'loss_bbox_vor', 'loss_bbox_ovl',
              'loss_bbox_edg', 'loss_ss'}
    assert set(losses) == expect, set(losses)
    for k, v in losses.items():
        assert bool(np.isfinite(float(v.item()))), (k, float(v.item()))

    # plumbing state checks
    assert model.bbox_head.images_no_copypaste.shape == (4, 3, 1024, 1024)
    assert model.bbox_head.images.shape == (4, 3, 1024, 1024)
    for i, t in enumerate(model._last_targets_all):
        assert t['rboxes'].shape[1] == 5
        assert t['bids'].shape == (t['rboxes'].shape[0], 4)
        assert t['ss'][0] in ('rot', 'flp', 'sca')
        # fill-back happened: rboxes wh 不再全是 gt 初值（voronoi 产的伪框）
    b0 = model._last_targets_all[0]['bids'].numpy()
    b2 = model._last_targets_all[2]['bids'].numpy()
    assert (b0[:, 0] == 0).all() and (b2[:, 0] == 2).all(), 'aug batch offset'
    assert (b2[:, 2] == 1).all(), 'aug view flag'
    print('PASS: v3 detector build + train plumbing smoke '
          f"(ss={model._last_targets_all[0]['ss'][0]}, "
          f"n_targets={len(model._last_targets_all)})")


if __name__ == '__main__':
    main()
