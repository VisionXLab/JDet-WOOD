"""Convert JDet --task=test output to the merge_dota_submission input format.

Run inside the **p2r-torch** env (uses mmrotate.qbox2rbox for the poly->rbox
conversion — the exact same code the reference merge pipeline uses, zero drift):

    PYTHONPATH=/root/ref/Point2RBox-v3 python tools/convert_test_results.py \
        --test-pkl work_dirs/point2rbox_v3_1x_dota/test/test_12.pkl \
        --out work_dirs/point2rbox_v3_1x_dota/test/merge_input.pkl

Input (JDet runner.test(), verified against runner.py L201-229 +
DOTADataset.evaluate consumption):
    list of (result, target) where
      result = (polys (N,8) np, scores (N,) np, labels (N,) np **0-based**)
      target = dict with 'filename' = patch image name (e.g. P0001__1024__0___0.png)

Output: list of {img_id: patch stem, labels: (N,) int 0-based,
                 bboxes: (N,5) float32 le90 rbox, scores: (N,) float32}
"""
import argparse
import os
import pickle

import numpy as np


def convert(results):
    import torch
    from mmrotate.structures.bbox import qbox2rbox

    out = []
    for result, target in results:
        polys, scores, labels = result
        polys = np.asarray(polys, dtype=np.float32).reshape(-1, 8)
        fname = target.get('filename') or os.path.basename(
            target.get('img_file', ''))
        img_id = os.path.splitext(os.path.basename(fname))[0]
        if 'flip_mode' in target:
            raise ValueError(
                f'{img_id}: flip_test results present; the official v3 eval '
                'has no flip TTA — rerun test without flip_test.')
        if polys.shape[0] == 0:
            rboxes = np.zeros((0, 5), dtype=np.float32)
        else:
            rboxes = qbox2rbox(torch.from_numpy(polys)).numpy().astype(
                np.float32)
        out.append(dict(
            img_id=img_id,
            labels=np.asarray(labels).astype(np.int64).reshape(-1),
            bboxes=rboxes,
            scores=np.asarray(scores, dtype=np.float32).reshape(-1),
        ))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test-pkl', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    with open(args.test_pkl, 'rb') as f:
        results = pickle.load(f)
    out = convert(results)
    n_det = sum(len(r['scores']) for r in out)
    with open(args.out, 'wb') as f:
        pickle.dump(out, f)
    print(f'{len(out)} patches, {n_det} dets -> {args.out}')


if __name__ == '__main__':
    main()
