"""Merge per-patch detections into a DOTA Task1 submission zip.

Zero-drift strategy: instead of re-implementing mmrotate's merge semantics
(offset by patch origin -> per-class nms_rotated(iou=0.1) -> Task1_<cls>.txt
-> zip), this tool instantiates the reference DOTAMetric and calls its own
merge_results(). Run inside the p2r-torch env:

    PYTHONPATH=/root/ref/Point2RBox-v3 python tools/merge_dota_submission.py \
        --results work_dirs/v3_test_results.pkl \
        --out work_dirs/dota/Task1

Input pickle: list of dicts, one per patch image —
    {'img_id': 'P0000__1024__0___0',      # patch filename stem
     'labels': (N,) int,                  # 0..14 DOTA class ids
     'bboxes': (N, 5) float32,            # le90 rbox (cx, cy, w, h, rad)
     'scores': (N,) float32}
The JDet test script exports this format (adapter on the Jittor side).

Output: <out>/Task1_<cls>.txt x15 + <out>/<basename>.zip for
https://captain-whu.github.io/DOTA/evaluation.html
"""
import argparse
import os
import pickle

import numpy as np


DOTA10_CLASSES = (
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True, help='pickle with patch dets')
    p.add_argument('--out', required=True,
                   help='outfile_prefix, must not exist yet (mmrotate rule)')
    args = p.parse_args()

    from mmrotate.evaluation.metrics.dota_metric import DOTAMetric

    with open(args.results, 'rb') as f:
        results = pickle.load(f)

    # sanity: required keys and shapes
    for r in results[:5]:
        assert {'img_id', 'labels', 'bboxes', 'scores'} <= set(r), r.keys()
        assert np.asarray(r['bboxes']).ndim == 2 and \
            (len(r['bboxes']) == 0 or np.asarray(r['bboxes']).shape[1] == 5)

    metric = DOTAMetric(format_only=True, merge_patches=True,
                        outfile_prefix=args.out)
    metric.dataset_meta = dict(classes=DOTA10_CLASSES)
    zip_path = metric.merge_results(results, args.out)
    print(f'submission zip: {zip_path}')


if __name__ == '__main__':
    main()
