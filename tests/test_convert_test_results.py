"""E2E: JDet test.pkl -> convert_test_results -> merge_dota_submission -> Task1.

Run in p2r-torch:
    PYTHONPATH=/root/ref/Point2RBox-v3 python tests/test_convert_test_results.py

Known-value chain: an axis-aligned poly on patch P9999__1024__824___0 must come
out of Task1_plane.txt offset by +824 in x with the original score.
"""
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'tools'))


def main():
    tmp = tempfile.mkdtemp(prefix='cvt_test_')
    try:
        # patch 1: plane at (100,50) w=40 h=20 angle=0 -> poly corners known
        poly = np.float32([80, 40, 120, 40, 120, 60, 80, 60])
        res1 = (poly[None], np.float32([0.9]), np.int64([0]))
        tgt1 = dict(filename='P9999__1024__824___0.png', img_size=(1024, 1024))
        # patch 2: empty detections
        res2 = (np.zeros((0, 8), np.float32), np.zeros((0,), np.float32),
                np.zeros((0,), np.int64))
        tgt2 = dict(filename='P9999__1024__0___0.png', img_size=(1024, 1024))
        test_pkl = os.path.join(tmp, 'test_12.pkl')
        with open(test_pkl, 'wb') as f:
            pickle.dump([(res1, tgt1), (res2, tgt2)], f)

        merge_in = os.path.join(tmp, 'merge_input.pkl')
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, 'tools/convert_test_results.py'),
             '--test-pkl', test_pkl, '--out', merge_in],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        with open(merge_in, 'rb') as f:
            conv = pickle.load(f)
        assert len(conv) == 2
        rb = conv[0]['bboxes'][0]
        assert abs(rb[0] - 100) < 1e-3 and abs(rb[1] - 50) < 1e-3, rb
        assert {abs(rb[2] - 40) < 1e-3, abs(rb[3] - 20) < 1e-3} == {True} or \
               {abs(rb[2] - 20) < 1e-3, abs(rb[3] - 40) < 1e-3} == {True}, rb

        out = os.path.join(tmp, 'Task1')
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, 'tools/merge_dota_submission.py'),
             '--results', merge_in, '--out', out],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        lines = [l for l in open(os.path.join(out, 'Task1_plane.txt')) if l.strip()]
        assert len(lines) == 1, lines
        parts = lines[0].split()
        assert parts[0] == 'P9999' and float(parts[1]) == 0.9, parts
        xs = np.array(list(map(float, parts[2::2])))
        ys = np.array(list(map(float, parts[3::2])))
        assert abs(xs.mean() - (100 + 824)) < 1.0, xs   # offset applied
        assert abs(ys.mean() - 50) < 1.0, ys
        print('PASS: test.pkl -> convert -> merge -> Task1 known-value chain')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
