"""L2 parity test: ported point2rbox_v3_utils vs PyTorch reference.

Prerequisite (p2r-torch env):
    python tools/dump_filter_masks_reference.py \
        --out tests/data/filter_masks_reference.pkl

Then run in p2r-jittor env:
    python tests/test_filter_masks.py

Acceptance: pure cv2/numpy computation -> rtol 1e-6 on every metric value and
combined score, exact match on best_mask_idx. Covers every per-class entry of
_base_sam-dotav1-0.py (incl. circularity=-3 + penalty_circularity=100 for
classes 3/8/10), the 'default' entry, the built-in fallback config, and the
jt.Var tensor-input branch of calculate_shape_metrics.
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import jittor as jt  # noqa: E402
from jdet.models.losses.point2rbox_v3_utils import (  # noqa: E402
    calculate_shape_metrics, filter_masks)

HERE = os.path.dirname(__file__)
REF = os.path.join(HERE, 'data', 'filter_masks_reference.pkl')
SAM_CFG = os.path.join(HERE, '..', 'configs', 'point2rbox_v3',
                       '_base_sam-dotav1-0.py')
RTOL = 1e-6

ALL_METRICS = ['circularity', 'rectangularity', 'color_consistency',
               'aspect_ratio_reasonableness', 'center_alignment']


def load_mask_filter_config():
    ns = {}
    with open(SAM_CFG) as f:
        exec(f.read(), ns)
    return ns['mask_filter_config']


def assert_metrics_close(got, want, ctx):
    assert set(got) == set(want), f'{ctx}: keys {set(got)} != {set(want)}'
    for k in want:
        g, w = float(got[k]), float(want[k])
        err = abs(g - w) / max(abs(w), 1e-12)
        assert err < RTOL or abs(g - w) < 1e-12, \
            f'{ctx}: {k} got {g} want {w} rel {err:.3e}'


def main():
    with open(REF, 'rb') as f:
        ref = pickle.load(f)

    img_hwc = ref['img_hwc']
    masks = [m for m in ref['masks'].astype(bool)]
    points = ref['points']
    img_chw_jt = jt.array(np.transpose(img_hwc, (2, 0, 1)))
    cfg = load_mask_filter_config()

    n = 0
    for case in ref['metrics_cases']:
        mi = case['mask_idx']
        out_np = calculate_shape_metrics(
            img_chw_jt, masks[mi], ALL_METRICS, original_image=img_hwc,
            aspect_ratio_range=(1, 5), prompt_point=points[mi])
        assert_metrics_close(out_np, case['numpy_branch'],
                             f'mask{mi}/numpy_branch')
        out_t = calculate_shape_metrics(
            img_chw_jt, masks[mi], ALL_METRICS, original_image=img_chw_jt,
            aspect_ratio_range=(1, 5), prompt_point=points[mi])
        assert_metrics_close(out_t, case['tensor_branch'],
                             f'mask{mi}/tensor_branch')
        n += 2
    print(f'[OK ] calculate_shape_metrics: {n} cases '
          f'(numpy + jt.Var image branches) within rtol {RTOL}')

    scores = np.linspace(0.5, 0.95, len(masks)).astype(np.float32)
    for case in ref['filter_cases']:
        fc = None if case['use_fallback'] else cfg
        best, vals, mets = filter_masks(
            img_chw_jt, masks, scores, case['class_id'],
            img_hwc, points[0], filter_config=fc)
        ctx = f"filter[{case['case']}]"
        assert int(best) == case['best_mask_idx'], \
            f"{ctx}: best {int(best)} != {case['best_mask_idx']}"
        want_vals = case['metrics_values']
        got_vals = np.float64(vals)
        err = np.max(np.abs(got_vals - want_vals) /
                     np.maximum(np.abs(want_vals), 1e-12))
        assert err < RTOL, f'{ctx}: scores rel err {err:.3e}'
        for i, (gm, wm) in enumerate(zip(mets, case['shape_metrics'])):
            assert_metrics_close(gm, wm, f'{ctx}/mask{i}')
        print(f"[OK ] {ctx}: best_mask_idx={int(best)} "
              f"scores max rel err={err:.1e}")

    print(f"PASS: {n} metric cases + {len(ref['filter_cases'])} "
          f"filter_masks cases all match (rtol {RTOL})")


if __name__ == '__main__':
    main()
