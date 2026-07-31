"""Stage-1 parity for Point2RBoxV3Head: forward (rtol 1e-5), priors (exact),
get_targets (labels/bids bitexact, bbox rtol 1e-6), plus predict-path smoke
and the sort-before-truncate assertion.

Prereq (p2r-torch): PYTHONPATH=/root/ref/Point2RBox-v3 \
    python tools/dump_v3_head_reference.py --out tests/data/v3_head_reference.npz
Run (p2r-jittor): python tests/test_v3_head_stage1.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import jittor as jt  # noqa: E402
from jdet.models.roi_heads.point2rbox_v3_head import (  # noqa: E402
    Point2RBoxV3Head)

REF = os.path.join(os.path.dirname(__file__), 'data', 'v3_head_reference.npz')

CFG = dict(
    num_classes=15, in_channels=256, feat_channels=256,
    strides=[8, 16, 32, 64, 128],
    use_adaptive_scale=False, edge_loss_start_epoch=6,
    joint_angle_start_epoch=1, voronoi_type='standard',
    voronoi_thres=dict(
        default=[0.994, 0.005],
        override=(([2, 11], [0.999, 0.6]), ([7, 8, 10, 14], [0.95, 0.005]))),
    square_cls=[1, 9, 11],
    edge_loss_cls=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13],
    post_process={11: 1.2},
    angle_coder=dict(type='PSCCoder', angle_version='le90', dual_freq=False,
                     num_step=3, thr_mod=0),
    loss_cls=dict(type='MMDetFocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25,
                  loss_weight=1.0),
    loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0),
    loss_overlap=dict(type='GaussianOverlapLoss', loss_weight=10.0, lamb=0),
    loss_voronoi=dict(type='VoronoiWatershedLoss', loss_weight=5.0),
    loss_bbox_edg=dict(type='EdgeLoss', loss_weight=0.3),
    loss_ss=dict(type='Point2RBoxV2ConsistencyLoss', loss_weight=1.0),
    test_cfg=dict(nms_pre=2000, min_bbox_size=0, score_thr=0.05,
                  nms=dict(type='nms_rotated', iou_threshold=0.1),
                  max_per_img=2000),
)


def rel_err(a, b):
    return np.abs(a - b).max() / (np.abs(b).max() + 1e-12)


def load_head(ref):
    head = Point2RBoxV3Head(**CFG)
    params = {}
    for k in ref.files:
        if not k.startswith('sd::'):
            continue
        params[k[4:]] = jt.array(ref[k])  # names align (both use .gn.)
    my_keys = {n for n, _ in head.named_parameters()}
    missing = {k for k in my_keys if k not in params}
    # PSCCoder coef buffers etc. are constructed, not loaded
    missing = {m for m in missing if not m.startswith(('angle_coder',
                                                       'bbox_coder'))}
    assert not missing, f'missing from golden: {sorted(missing)[:8]}'
    head.load_parameters(params)
    head.eval()
    return head


def main():
    ref = np.load(REF)
    head = load_head(ref)

    # ---- forward parity -------------------------------------------------
    feats = [jt.array(ref[f'feat_{i}']) for i in range(5)]
    with jt.no_grad():
        cls_scores, bbox_preds, angle_preds = head.execute(feats)
    worst = 0.0
    for i in range(5):
        for name, mine in (('cls', cls_scores[i]), ('bbox', bbox_preds[i]),
                           ('angle', angle_preds[i])):
            e = rel_err(mine.numpy(), ref[f'{name}_{i}'])
            worst = max(worst, e)
            assert e < 1e-5, (name, i, e)
    print(f'[OK ] forward: 15 outputs worst rel_err={worst:.3e}')

    # ---- priors exact ---------------------------------------------------
    sizes = [tuple(ref[f'feat_{i}'].shape[-2:]) for i in range(5)]
    priors = head.get_points(sizes)
    for i in range(5):
        d = np.abs(priors[i].numpy() - ref[f'priors_{i}']).max()
        assert d == 0.0, (i, d)
    print('[OK ] priors: exact match with mmdet MlvlPointGenerator')

    # ---- get_targets ----------------------------------------------------
    targets = [
        dict(rboxes=jt.array(ref['gt0_bboxes']),
             labels=jt.array(ref['gt0_labels']).int32(),
             bids=jt.array(ref['gt0_bids']).int32()),
        dict(rboxes=jt.array(ref['gt1_bboxes']),
             labels=jt.array(ref['gt1_labels']).int32(),
             bids=jt.array(ref['gt1_bids']).int32()),
    ]
    labels, bbox_targets, bid_targets = head.get_targets(priors, targets)
    for i in range(5):
        lab_diff = int((labels[i].numpy().astype(np.int64) !=
                        ref[f'tgt_labels_{i}'].astype(np.int64)).sum())
        assert lab_diff == 0, (i, lab_diff)
        bid_diff = int((bid_targets[i].numpy().astype(np.int64) !=
                        ref[f'tgt_bid_{i}'].astype(np.int64)).sum())
        assert bid_diff == 0, (i, bid_diff)
        e = rel_err(bbox_targets[i].numpy(), ref[f'tgt_bbox_{i}'])
        assert e < 1e-6, (i, e)
    n_pos = int((labels[0].numpy() != 15).sum())
    print(f'[OK ] get_targets: labels/bids bitexact ×5 lvls, bbox rtol<1e-6 '
          f'(lvl0 pos={n_pos})')

    # ---- predict smoke + sort-before-truncate ---------------------------
    head.pseudo_generator = False
    with jt.no_grad():
        res = head.predict(feats, targets=[dict(), dict()])
    assert len(res) == 2 and set(res[0]) == {'bboxes', 'scores', 'labels'}
    s = res[0]['scores'].numpy()
    assert (np.diff(s) <= 1e-6).all(), 'scores not sorted descending'
    print(f'[OK ] predict smoke: {len(s)} dets, scores sorted desc')

    # Real test images may have no class score above threshold on all levels.
    # This must return an empty result rather than broadcasting (0,1) in where.
    strict_cfg = dict(head.test_cfg)
    strict_cfg['score_thr'] = 2.0
    empty = head.predict_by_feat(
        cls_scores, bbox_preds, angle_preds, targets=[dict(), dict()],
        cfg=strict_cfg)
    assert all(r['bboxes'].shape == (0, 5) for r in empty)
    assert all(r['scores'].shape == (0,) for r in empty)
    print('[OK ] empty-candidate inference path')

    # dedicated sort assertion on _bbox_post_process with shuffled input
    n = 50
    rng = np.random.RandomState(0)
    bx = np.stack([rng.uniform(100, 900, n), rng.uniform(100, 900, n),
                   rng.uniform(20, 60, n), rng.uniform(20, 60, n),
                   rng.uniform(-1.5, 1.5, n)], 1).astype(np.float32)
    sc = rng.rand(n).astype(np.float32)
    lb = rng.randint(0, 15, n)
    out = head._bbox_post_process(
        jt.array(bx), jt.array(sc), jt.array(lb.astype(np.int32)),
        dict(nms=dict(iou_threshold=0.1), max_per_img=10, min_bbox_size=0),
        dict(), rescale=False, with_nms=True)
    kept = out['scores'].numpy()
    assert len(kept) <= 10 and (np.diff(kept) <= 1e-6).all()
    # the survivor set must contain the global top score (never truncated away)
    assert abs(kept[0] - sc.max()) < 1e-6
    print(f'[OK ] sort-before-truncate: top-score retained, {len(kept)} kept')

    # pseudo path smoke (training-mode branch selection)
    head.train()
    res = head.predict(feats, targets=targets)
    head.eval()
    assert res[0]['bboxes'].shape[1] == 5
    print(f'[OK ] pseudo path smoke: {res[0]["bboxes"].shape[0]} pseudo boxes')

    print('PASS: v3 head stage-1 (forward/priors/targets/predict)')


if __name__ == '__main__':
    main()
