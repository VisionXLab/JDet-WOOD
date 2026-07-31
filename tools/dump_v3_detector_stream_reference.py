"""Golden dump for Point2RBoxV3 dual-stream assembly + generate_pseudo_targets.

Run in p2r-torch:
    PYTHONPATH=/root/ref/Point2RBox-v3 python tools/dump_v3_detector_stream_reference.py

Uses SimpleNamespace pseudo-self + unbound ref methods so no full detector
build is needed. torch.rand is monkeypatched to inject a fixed draw sequence;
one dump per aug branch (rot / flp / sca), plus a standard-branch
generate_pseudo_targets golden on a real DOTA patch.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, '/root/ref/Point2RBox-v3')

from mmengine.structures import InstanceData  # noqa: E402
from mmrotate.structures.bbox import RotatedBoxes  # noqa: E402
from mmrotate.models.detectors.point2rbox_v3 import Point2RBoxV3  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), '..', 'tests', 'data',
                   'v3_stream_reference.npz')

# fixed synthetic batch
RS = np.random.RandomState(7)
IMAGES = RS.randn(2, 3, 256, 256).astype(np.float32)
GTS = []
for i in range(2):
    n = 4
    rb = np.stack([RS.uniform(60, 200, n), RS.uniform(60, 200, n),
                   RS.uniform(20, 50, n), RS.uniform(20, 50, n),
                   RS.uniform(-1.2, 1.2, n)], 1).astype(np.float32)
    GTS.append((rb, RS.randint(0, 15, n).astype(np.int64)))

BRANCH_DRAWS = {   # (sel_p, aug_val) given ss_prob=[0.68, 0.07, 0.25]
    'rot': (0.10, 0.5),
    'flp': (0.70, 0.0),
    'sca': (0.90, 0.5),
}


def make_self():
    return SimpleNamespace(
        ss_prob=[0.68, 0.07, 0.25],
        rotate_range=(0.25, 0.75),
        scale_range=(0.5, 0.9),
        rotate_crop=lambda *a, **k: Point2RBoxV3.rotate_crop(ns, *a, **k),
    )


def make_targets():
    out = []
    offset = 1
    for i, (rb, lb) in enumerate(GTS):
        inst = InstanceData()
        inst.bboxes = RotatedBoxes(torch.from_numpy(rb.copy()))
        inst.labels = torch.from_numpy(lb.copy())
        blen = len(lb)
        bids = torch.zeros(blen, 4, dtype=torch.long)
        bids[:, 0] = i
        bids[:, 3] = torch.arange(0, blen) + offset
        inst.bids = bids
        offset += blen
        out.append(inst)
    return out


arrays = {'images': IMAGES}
for bi, (rb, lb) in enumerate(GTS):
    arrays[f'gt_rb_{bi}'] = rb
    arrays[f'gt_lb_{bi}'] = lb

orig_rand = torch.rand
for branch, (sel_p, aug_val) in BRANCH_DRAWS.items():
    ns = make_self()
    draws = iter([sel_p, aug_val, aug_val])

    def fake_rand(*a, **k):
        try:
            return torch.tensor([next(draws)])
        except StopIteration:
            return orig_rand(*a, **k)

    torch.rand = fake_rand
    try:
        inputs = torch.from_numpy(IMAGES.copy())
        targets = make_targets()
        metas = [dict(idx=0), dict(idx=1)]
        dual_inputs, dual_targets = \
            Point2RBoxV3.prepare_dual_stream_inputs(ns, inputs, targets, metas)
    finally:
        torch.rand = orig_rand

    arrays[f'{branch}_inputs'] = dual_inputs.numpy()
    arrays[f'{branch}_ss_val'] = np.float32(
        metas[0]['ss'][1] if isinstance(metas[0]['ss'][1], float)
        else float(metas[0]['ss'][1]))
    for ti, t in enumerate(dual_targets):
        arrays[f'{branch}_rb_{ti}'] = t.bboxes.tensor.numpy()
        arrays[f'{branch}_lb_{ti}'] = t.labels.numpy()
        arrays[f'{branch}_bid_{ti}'] = t.bids.numpy()

# --- generate_pseudo_targets golden (standard branch, real DOTA patch) ---
import cv2  # noqa: E402

img_path = '/root/data/split_ss_dota/trainval/images/P0000__1024__0___0.png'
img = cv2.imread(img_path)
img_t = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1))
arrays['pat_image'] = img_t.numpy()

n = 5
RS2 = np.random.RandomState(11)
prb = np.stack([RS2.uniform(200, 800, n), RS2.uniform(200, 800, n),
                np.full(n, 30.0), np.full(n, 30.0),
                np.zeros(n)], 1).astype(np.float32)
plb = RS2.randint(0, 15, n).astype(np.int64)
arrays['pat_rb'] = prb
arrays['pat_lb'] = plb

inst = InstanceData()
inst.bboxes = RotatedBoxes(torch.from_numpy(prb.copy()))
inst.labels = torch.from_numpy(plb.copy())
bids = torch.zeros(n, 4, dtype=torch.long)
bids[:, 3] = torch.arange(n) + 1
inst.bids = bids

from mmengine.structures import InstanceData as ID  # noqa
from mmdet.structures import DetDataSample  # noqa: E402
ds = DetDataSample(metainfo=dict(idx=0))
ds.gt_instances = inst

head_ns = SimpleNamespace(
    voronoi_type='standard',
    voronoi_thres=dict(
        default=[0.994, 0.005],
        override=(([2, 11], [0.999, 0.6]), ([7, 8, 10, 14], [0.95, 0.005]))),
    num_classes=15,
    images_no_copypaste=img_t[None],
    loss_voronoi=SimpleNamespace(use_class_specific_watershed=False),
)
det_ns = SimpleNamespace(
    bbox_head=head_ns,
    voronoi_diagram_watershed=lambda *a, **k:
        Point2RBoxV3.voronoi_diagram_watershed(det_ns, *a, **k),
)
res = Point2RBoxV3.generate_pseudo_targets(det_ns, [ds])
arrays['pat_pseudo'] = res[0].bboxes.tensor.numpy()
arrays['pat_pseudo_lb'] = res[0].labels.numpy()

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
np.savez_compressed(OUT, **arrays)
print(f'saved {len(arrays)} arrays -> {OUT}')
