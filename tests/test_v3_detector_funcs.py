"""L2 parity: v3 detector module functions vs PyTorch golden.

Prereq (p2r-torch): tools/dump_v3_detector_funcs_reference.py
Run (p2r-jittor):   python tests/test_v3_detector_funcs.py

Tolerances: gaussian_2d rtol 1e-5; voronoi pseudo_info bit-equal in the
watershed-derived fields (cv2 is deterministic given identical uint8 input);
get_single_pattern chip rtol 1e-4 (two chained grid_sample resamplings).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import jittor as jt  # noqa: E402
from jdet.models.networks.point2rbox_v3_functions import (  # noqa: E402
    gaussian_2d, get_single_pattern, voronoi_diagram_watershed)

REF = os.path.join(os.path.dirname(__file__), 'data',
                   'v3_detector_funcs_reference.npz')


def rel_err(a, b):
    return np.abs(a - b).max() / (np.abs(b).max() + 1e-12)


def main():
    ref = np.load(REF)
    n_fail = 0

    # ---- gaussian_2d ----
    g = gaussian_2d(jt.array(ref['g2d_xy']), jt.array(ref['g2d_mu']),
                    jt.array(ref['g2d_sigma']))
    gn = gaussian_2d(jt.array(ref['g2d_xy']), jt.array(ref['g2d_mu']),
                     jt.array(ref['g2d_sigma']), normalize=True)
    for name, ours, gold in (('g2d', g, ref['g2d_out']),
                             ('g2d_norm', gn, ref['g2d_out_norm'])):
        e = rel_err(ours.numpy(), gold)
        ok = e < 1e-5
        n_fail += 0 if ok else 1
        print(f'[{"OK " if ok else "FAIL"}] {name}: rel_err={e:.3e}')

    # ---- voronoi_diagram_watershed ----
    image = jt.array(ref['image_chw'])
    voronoi_thres = dict(
        default=[0.994, 0.005],
        override=(([2, 11], [0.999, 0.6]), ([7, 8, 10, 14], [0.95, 0.005])))
    for vtype in ('standard', 'gaussian-orientation'):
        sig = jt.array(ref['vor_sigma']) if vtype != 'standard' else None
        out = voronoi_diagram_watershed(
            jt.array(ref['vor_mu']), sig,
            jt.array(ref['vor_labels'].astype(np.int32)), image,
            voronoi_type=vtype, voronoi_thres=voronoi_thres, num_classes=15)
        ours = np.float32(out)
        gold = ref[f'vor_out_{vtype}']
        same_shape = ours.shape == gold.shape
        e = np.abs(ours - gold).max() if same_shape else np.inf
        ok = same_shape and e < 1e-3
        exact = same_shape and np.array_equal(ours, gold)
        n_fail += 0 if ok else 1
        print(f'[{"OK " if ok else "FAIL"}] voronoi[{vtype}]: '
              f'max_abs_diff={e:.3e} bitexact={exact}')
        if not ok and same_shape:
            bad = np.nonzero(np.abs(ours - gold) > 1e-3)[0]
            print(f'      mismatched idx: {bad[:10]} '
                  f'ours={ours[bad[:5]]} gold={gold[bad[:5]]}')

    # ---- get_single_pattern (injected randomness) ----
    rng = dict(randn2=ref['gsp_randn2'], rand2=ref['gsp_rand2'],
               rand3=ref['gsp_rand3'].astype(np.float64))
    chip, obbox, _ = get_single_pattern(
        image, ref['gsp_bbox'], 5, [1, 9, 11], rng=rng)
    gold_chip = ref['gsp_chip']
    shape_ok = tuple(chip.shape) == gold_chip.shape
    e_bbox = np.abs(np.float32(obbox) - ref['gsp_obbox']).max()
    if shape_ok:
        e_chip = rel_err(chip.numpy(), gold_chip)
        ok = e_chip < 1e-4 and e_bbox < 1e-4
    else:
        e_chip = np.inf
        ok = False
    n_fail += 0 if ok else 1
    print(f'[{"OK " if ok else "FAIL"}] get_single_pattern: '
          f'chip_shape={tuple(chip.shape)} vs {gold_chip.shape} '
          f'chip_rel_err={e_chip:.3e} bbox_max_diff={e_bbox:.3e}')

    assert n_fail == 0, f'{n_fail} case(s) failed'
    print('PASS: v3 detector module functions match golden')


if __name__ == '__main__':
    main()
