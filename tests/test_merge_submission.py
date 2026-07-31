"""M4' toolchain test (runs in p2r-torch env):

    PYTHONPATH=/root/ref/Point2RBox-v3 python tests/test_merge_submission.py

Builds synthetic per-patch detections for one original image split into two
overlapping patches, runs tools/merge_dota_submission.py, and verifies:
- 15 Task1_<cls>.txt + zip produced
- patch-origin offsets applied to cx,cy
- the duplicate detection in the overlap region is NMS-merged to one line
- line format is `imgname score x1 y1 x2 y2 x3 y3 x4 y4`
"""
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')


def main():
    tmp = tempfile.mkdtemp(prefix='merge_test_')
    try:
        # One original image P9999 split like DOTA ss: patches at x=0 and x=824.
        # A "plane" at original cx=900 appears in both patches (dup in overlap);
        # a "ship" appears only in patch 2.
        results = [
            dict(img_id='P9999__1024__0___0',
                 labels=np.array([0]),
                 bboxes=np.array([[900., 500., 80., 60., 0.3]], np.float32),
                 scores=np.array([0.9], np.float32)),
            dict(img_id='P9999__1024__824___0',
                 labels=np.array([0, 6]),
                 bboxes=np.array([[76., 500., 80., 60., 0.3],
                                  [200., 300., 120., 40., -0.5]], np.float32),
                 scores=np.array([0.8, 0.7], np.float32)),
        ]
        pkl = os.path.join(tmp, 'results.pkl')
        with open(pkl, 'wb') as f:
            pickle.dump(results, f)

        out = os.path.join(tmp, 'Task1')
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, 'tools/merge_dota_submission.py'),
             '--results', pkl, '--out', out],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr

        files = sorted(f for f in os.listdir(out) if f.startswith('Task1_'))
        assert len(files) == 15, files
        assert any(f.endswith('.zip') for f in os.listdir(out))

        plane_lines = open(os.path.join(out, 'Task1_plane.txt')).read().split('\n')
        plane_lines = [l for l in plane_lines if l]
        # dup (iou≈1 after offset: 900 vs 824+76=900) must collapse to one line
        assert len(plane_lines) == 1, plane_lines
        parts = plane_lines[0].split()
        assert parts[0] == 'P9999' and len(parts) == 10, parts
        assert float(parts[1]) == 0.9  # higher-score det wins NMS
        xs = np.array(list(map(float, parts[2::2])))
        assert 850 < xs.mean() < 950, xs  # offset applied, centered near 900

        ship_lines = [l for l in open(os.path.join(out, 'Task1_ship.txt')).read().split('\n') if l]
        assert len(ship_lines) == 1
        assert 990 < np.array(list(map(float, ship_lines[0].split()[2::2]))).mean() < 1060

        print('PASS: merge/offset/NMS/format all correct')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
