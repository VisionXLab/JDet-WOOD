"""L2 parity: Point2RBoxV3 dual-stream assembly + generate_pseudo_targets
vs the PyTorch golden (tools/dump_v3_detector_stream_reference.py).

Run in p2r-jittor:  CUDA_VISIBLE_DEVICES=1 python tests/test_v3_detector_stream.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, ROOT)

import jittor as jt  # noqa: E402
from jdet.models.networks.point2rbox_v3 import Point2RBoxV3  # noqa: E402

REF = os.path.join(ROOT, 'tests', 'data', 'v3_stream_reference.npz')

BRANCH_DRAWS = {
    'rot': (0.10, 0.5),
    'flp': (0.70, 0.0),
    'sca': (0.90, 0.5),
}


def rel_err(a, b):
    return np.abs(a - b).max() / (np.abs(b).max() + 1e-12)


def make_self():
    ns = SimpleNamespace(
        ss_prob=[0.68, 0.07, 0.25],
        rotate_range=(0.25, 0.75),
        scale_range=(0.5, 0.9),
    )
    ns.rotate_crop = lambda *a, **k: Point2RBoxV3.rotate_crop(ns, *a, **k)
    ns.vflip = lambda img: Point2RBoxV3.vflip(ns, img)
    ns.prepare_dual_stream = \
        lambda *a, **k: Point2RBoxV3.prepare_dual_stream(ns, *a, **k)
    return ns


def make_targets(ref):
    targets = []
    offset = 1
    for i in range(2):
        rb = ref[f'gt_rb_{i}']
        lb = ref[f'gt_lb_{i}']
        blen = len(lb)
        bids = np.zeros((blen, 4), dtype=np.int32)
        bids[:, 0] = i
        bids[:, 3] = np.arange(blen) + offset
        offset += blen
        targets.append(dict(rboxes=jt.array(rb.copy()),
                            labels=jt.array(lb.astype(np.int32)),
                            bids=jt.array(bids)))
    return targets


def main():
    ref = np.load(REF)
    worst_img, worst_rb = 0.0, 0.0
    for branch, (sel_p, aug_val) in BRANCH_DRAWS.items():
        ns = make_self()
        images = jt.array(ref['images'].copy())
        targets = make_targets(ref)
        imgs_all, tgts_all = ns.prepare_dual_stream(
            images, targets, rng=dict(sel_p=sel_p, aug_val=aug_val))
        e_img = rel_err(imgs_all.numpy(), ref[f'{branch}_inputs'])
        worst_img = max(worst_img, e_img)
        # rot goes through grid_sample(bilinear, reflection): jt vs torch
        # kernels differ at interpolation-arithmetic level (measured max abs
        # 1.5e-4 on ~5.3-scale data, scattered over 0.7% of pixels, interior
        # == border, gt boxes bit-exact) -> relaxed gate for that branch only.
        img_tol = 1e-4 if branch == 'rot' else 1e-5
        assert len(tgts_all) == 4
        for ti, t in enumerate(tgts_all):
            e_rb = rel_err(t['rboxes'].numpy(), ref[f'{branch}_rb_{ti}'])
            worst_rb = max(worst_rb, e_rb)
            assert (t['labels'].numpy() == ref[f'{branch}_lb_{ti}']).all()
            assert (t['bids'].numpy() == ref[f'{branch}_bid_{ti}']).all(), \
                (branch, ti)
            assert e_rb < 1e-5, (branch, ti, e_rb)
        status = 'OK ' if e_img < img_tol else 'FAIL'
        print(f'[{status}] {branch}: inputs rel_err={e_img:.3e} '
              f'(tol {img_tol:.0e}) rb worst={worst_rb:.3e} labels/bids bitexact')
        assert e_img < img_tol, (branch, e_img)

    # generate_pseudo_targets (standard branch)
    head_ns = SimpleNamespace(
        voronoi_type='standard',
        voronoi_thres=dict(
            default=[0.994, 0.005],
            override=(([2, 11], [0.999, 0.6]),
                      ([7, 8, 10, 14], [0.95, 0.005]))),
        num_classes=15,
        images_no_copypaste=jt.array(ref['pat_image'].copy())[None],
        loss_voronoi=SimpleNamespace(use_class_specific_watershed=False),
    )
    det_ns = SimpleNamespace(bbox_head=head_ns)
    det_ns.generate_pseudo_targets = \
        lambda *a, **k: Point2RBoxV3.generate_pseudo_targets(det_ns, *a, **k)

    n = len(ref['pat_lb'])
    bids = np.zeros((n, 4), dtype=np.int32)
    bids[:, 3] = np.arange(n) + 1
    target = dict(rboxes=jt.array(ref['pat_rb'].copy()),
                  labels=jt.array(ref['pat_lb'].astype(np.int32)),
                  bids=jt.array(bids))
    res = det_ns.generate_pseudo_targets([target])[0]
    diff = np.abs(res['bboxes'].numpy() - ref['pat_pseudo']).max()
    assert (res['labels'].numpy() == ref['pat_pseudo_lb']).all()
    status = 'OK ' if diff == 0.0 else 'FAIL'
    print(f'[{status}] generate_pseudo_targets[standard]: '
          f'max_abs_diff={diff:.3e} bitexact={diff == 0.0}')
    assert diff == 0.0, diff

    print('PASS: v3 detector dual-stream + pseudo-target parity')


if __name__ == '__main__':
    main()
