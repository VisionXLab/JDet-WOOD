"""Dump golden reference outputs for the SAM mask-filtering utils.

Run inside the p2r-torch env:

    python tools/dump_filter_masks_reference.py \
        --ref /root/ref/Point2RBox-v3 \
        --out tests/data/filter_masks_reference.pkl

Loads the reference mmrotate/models/losses/utils.py directly by path (no
registry side effects), builds deterministic synthetic masks (circle /
axis-aligned rect / rotated rect / elongated bar / empty / off-center prompt),
and records calculate_shape_metrics + filter_masks outputs for:
- every per-class entry of configs/point2rbox_v3/_base_sam-dotav1-0.py
  (classes 0,1,2,3,7,8,9,10,11,14 + 'default'), i.e. every metric combination
  actually used in training, incl. circularity=-3 + penalty_circularity=100;
- the built-in fallback filter_config (filter_config=None);
- direct calculate_shape_metrics calls with a torch.Tensor original_image
  (CHW) to pin the tensor-input branch.
"""
import argparse
import importlib.util
import os
import pickle

import cv2
import numpy as np
import torch


def load_ref_utils(ref_root):
    path = os.path.join(ref_root, 'mmrotate/models/losses/utils.py')
    spec = importlib.util.spec_from_file_location('ref_p2rv3_utils', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_mask_filter_config(ref_root):
    path = os.path.join(ref_root,
                        'configs/point2rbox_v3/_base_sam-dotav1-0.py')
    ns = {}
    with open(path) as f:
        exec(f.read(), ns)
    return ns['mask_filter_config']


def build_inputs():
    rng = np.random.RandomState(0)
    H = W = 256
    img_hwc = (rng.rand(H, W, 3) * 255).astype(np.float32)

    masks, points = [], []

    m = np.zeros((H, W), np.uint8)
    cv2.circle(m, (128, 128), 50, 1, -1)
    masks.append(m.astype(bool)); points.append(np.float32([128, 128]))

    m = np.zeros((H, W), np.uint8)
    cv2.rectangle(m, (78, 98), (178, 158), 1, -1)
    masks.append(m.astype(bool)); points.append(np.float32([128, 128]))

    m = np.zeros((H, W), np.uint8)
    box = cv2.boxPoints(((128, 128), (120, 48), 30.0))
    cv2.fillPoly(m, [np.int64(box)], 1)
    masks.append(m.astype(bool)); points.append(np.float32([128, 128]))

    m = np.zeros((H, W), np.uint8)
    cv2.rectangle(m, (8, 118), (248, 138), 1, -1)  # elongated bar, ar=12
    masks.append(m.astype(bool)); points.append(np.float32([128, 128]))

    masks.append(np.zeros((H, W), bool)); points.append(np.float32([128, 128]))

    # prompt far outside the circle -> center_alignment == -100 branch
    m = np.zeros((H, W), np.uint8)
    cv2.circle(m, (60, 60), 30, 1, -1)
    masks.append(m.astype(bool)); points.append(np.float32([220, 220]))

    return img_hwc, masks, points


ALL_METRICS = ['circularity', 'rectangularity', 'color_consistency',
               'aspect_ratio_reasonableness', 'center_alignment']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ref', default='/root/ref/Point2RBox-v3')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    ref = load_ref_utils(args.ref)
    cfg = load_mask_filter_config(args.ref)
    img_hwc, masks, points = build_inputs()
    img_chw_t = torch.from_numpy(np.transpose(img_hwc, (2, 0, 1)))

    record = dict(img_hwc=img_hwc,
                  masks=np.stack(masks),
                  points=np.stack(points),
                  metrics_cases=[], filter_cases=[])

    # 1) direct calculate_shape_metrics: numpy image branch + tensor branch,
    #    all metrics requested, with aspect range + prompt point.
    for mi, (mask, pt) in enumerate(zip(masks, points)):
        out_np = ref.calculate_shape_metrics(
            img_chw_t, mask, ALL_METRICS, original_image=img_hwc,
            aspect_ratio_range=(1, 5), prompt_point=pt)
        out_t = ref.calculate_shape_metrics(
            img_chw_t, mask, ALL_METRICS, original_image=img_chw_t,
            aspect_ratio_range=(1, 5), prompt_point=pt)
        record['metrics_cases'].append(
            dict(mask_idx=mi, numpy_branch=out_np, tensor_branch=out_t))

    # 2) filter_masks over every class entry in the real training config,
    #    plus 'default', plus the built-in fallback (filter_config=None).
    scores = np.linspace(0.5, 0.95, len(masks)).astype(np.float32)
    class_ids = [k for k in cfg if k != 'default'] + ['__default__',
                                                      '__fallback__']
    for cid in class_ids:
        if cid == '__default__':
            class_id, fc = 999, cfg          # unknown id -> default entry
        elif cid == '__fallback__':
            class_id, fc = 3, None           # builtin fallback config path
        else:
            class_id, fc = cid, cfg
        best, vals, mets = ref.filter_masks(
            img_chw_t, list(record['masks'].astype(bool)), scores, class_id,
            img_hwc, points[0], filter_config=fc)
        record['filter_cases'].append(
            dict(case=str(cid), class_id=class_id,
                 use_fallback=fc is None,
                 best_mask_idx=int(best),
                 metrics_values=np.float64(vals),
                 shape_metrics=mets))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(record, f)
    print(f'{len(record["metrics_cases"])} metric cases, '
          f'{len(record["filter_cases"])} filter cases -> {args.out}')


if __name__ == '__main__':
    main()
