"""Visualize original/augmented Point2RBox-v2 predictions at matched GT points.

The board uses the real checkpoint and real training images.  For each
augmentation it shows:
  1. predictions in the original view;
  2. predictions in the transformed view;
  3. transformed predictions mapped back to the original coordinates.

Large red/green disagreement in column 3 is exactly what consistency loss
penalizes.  The worst instances are numbered and summarized in the footer.
"""
import argparse
import copy
import math
import os

import cv2
import jittor as jt
import numpy as np

from jdet.config import init_cfg
from jdet.runner import Runner
from jdet.models.networks.point2rbox_v2 import _resized_crop_aa


def rbbox_poly(box):
    cx, cy, w, h, a = [float(x) for x in box]
    c, s = math.cos(a), math.sin(a)
    q = np.float32([[-w, -h], [w, -h], [w, h], [-w, h]]) * 0.5
    return q @ np.float32([[c, s], [-s, c]]) + np.float32([cx, cy])


def draw_boxes(image, boxes, color, ids=None, thickness=2):
    out = image.copy()
    for i, b in enumerate(boxes):
        p = np.rint(rbbox_poly(b)).astype(np.int32)
        cv2.polylines(out, [p], True, color, thickness, cv2.LINE_AA)
        if ids is not None and i in ids:
            x, y = np.rint(b[:2]).astype(int)
            cv2.putText(out, str(i), (x + 4, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                        cv2.LINE_AA)
    return out


def title(image, text):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 42), (18, 18, 18), -1)
    cv2.putText(out, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.64, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def covariance(boxes):
    wh = boxes[:, 2:4] * 0.5
    a = boxes[:, 4]
    c, s = np.cos(a), np.sin(a)
    r = np.stack([c, -s, s, c], 1).reshape(-1, 2, 2)
    d = np.zeros_like(r)
    d[:, 0, 0], d[:, 1, 1] = wh[:, 0], wh[:, 1]
    g = r @ d
    return g @ np.transpose(g, (0, 2, 1))


def discrepancy(a, b):
    ca, cb = covariance(a), covariance(b)
    scale = np.maximum(np.sqrt((ca * ca).sum((1, 2))), 1e-6)
    sigma = np.sqrt(((ca - cb) ** 2).sum((1, 2))) / scale
    angle = np.abs((b[:, 4] - a[:, 4] + np.pi / 2) % np.pi - np.pi / 2)
    return sigma, angle


def inverse_boxes(boxes, aug, value, h, w):
    b = boxes.copy()
    if aug == 'rot':
        c, s = math.cos(value), math.sin(value)
        ctr = np.float32([w / 2, h / 2])
        # forward row-vector matrix is [[c,s],[-s,c]]
        b[:, :2] = (b[:, :2] - ctr) @ np.float32([[c, -s], [s, c]]) + ctr
        b[:, 4] -= value
    elif aug == 'flp':
        b[:, 1] = h - b[:, 1]
        b[:, 4] = -b[:, 4]
    else:
        b[:, :4] /= value
    return b


def transform(model, image, target, aug, value):
    h, w = image.shape[-2:]
    t = copy.deepcopy(target)
    if aug == 'rot':
        y, ts = model.rotate_crop(image, value, (h, w), [t], 'reflection')
        return y, ts[0]
    if aug == 'flp':
        y = model.vflip(image)
        b = t['rboxes']
        t['rboxes'] = jt.concat(
            [b[:, :1], h - b[:, 1:2], b[:, 2:4], -b[:, 4:5]], 1)
        return y, t
    y = _resized_crop_aa(image, int(h / value), int(w / value), h, w)
    b = t['rboxes']
    t['rboxes'] = jt.concat([b[:, :4] * value, b[:, 4:5]], 1)
    return y, t


def to_uint8(image, target):
    x = image.copy()
    mean = np.asarray(target['mean'], np.float32)
    std = np.asarray(target['std'], np.float32)
    x = x * std + mean
    return np.clip(x.transpose(1, 2, 0), 0, 255).astype(np.uint8)[:, :, ::-1]


def pick_sample(dataset, min_objects=12):
    for idx, info in enumerate(dataset.img_infos):
        n = len(info['ann']['bboxes'])
        if n >= min_objects:
            return idx
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',
                   default='configs/point2rbox_v2/point2rbox_v2_1x_dota.py')
    p.add_argument('--out',
                   default='work_dirs/ss_debug/prediction_consistency.png')
    p.add_argument('--checkpoint', default=None,
                   help='Optional checkpoint override (loaded model-only).')
    p.add_argument('--dump', default=None,
                   help='Optional npz with per-view boxes and errors.')
    p.add_argument('--index', type=int, default=-1)
    args = p.parse_args()

    jt.flags.use_cuda = 1
    init_cfg(args.config)
    runner = Runner()  # automatically loads latest ckpt_12
    if args.checkpoint:
        runner.load(args.checkpoint, model_only=True)
    model = runner.model
    model.eval()
    model.bbox_head.train()  # select GT-point pseudo prediction path

    ds = runner.val_dataset
    idx = args.index if args.index >= 0 else pick_sample(ds)
    image_np, target_np = ds[idx]
    image = jt.array(image_np[None].astype(np.float32))
    target = {}
    for k, v in target_np.items():
        target[k] = jt.array(v) if isinstance(v, np.ndarray) else v
    base_vis = to_uint8(image_np, target_np)
    h, w = image_np.shape[-2:]

    specs = [('rot', math.radians(67.5)), ('flp', 0.0), ('sca', 0.70)]
    rows, summaries, dumped = [], [], {}
    for aug, value in specs:
        aug_im, aug_t = transform(model, image, target, aug, value)
        both = jt.concat([image, aug_im], 0)
        feat = model.backbone(both)
        if model.neck:
            feat = model.neck(feat)
        pred = model.bbox_head.predict(feat, [target, aug_t])
        ori = pred[0]['bboxes'].numpy()
        trs = pred[1]['bboxes'].numpy()
        inv = inverse_boxes(trs, aug, value, h, w)
        sig, ang = discrepancy(ori, inv)
        dumped[f'{aug}_ori'] = ori
        dumped[f'{aug}_trs'] = trs
        dumped[f'{aug}_inv'] = inv
        dumped[f'{aug}_sigma'] = sig
        dumped[f'{aug}_angle'] = ang
        score = sig + ang
        worst = np.argsort(-score)[:min(12, len(score))].tolist()

        aug_np = aug_im.numpy()[0]
        aug_target_np = dict(target_np)
        aug_target_np['mean'], aug_target_np['std'] = \
            target_np['mean'], target_np['std']
        aug_vis = to_uint8(aug_np, aug_target_np)
        left = draw_boxes(base_vis, ori, (0, 220, 0), worst)
        middle = draw_boxes(aug_vis, trs, (0, 165, 255), worst)
        right = draw_boxes(base_vis, ori, (0, 220, 0), worst)
        right = draw_boxes(right, inv, (0, 0, 255), worst)
        name = aug.upper() + (f' {math.degrees(value):.1f}deg'
                              if aug == 'rot' else
                              (f' {value:.2f}' if aug == 'sca' else ''))
        left = title(left, f'{name} | original prediction (GREEN)')
        middle = title(middle, f'{name} | transformed prediction (ORANGE)')
        right = title(right, f'{name} | inverse overlay GREEN vs RED')
        rows.append(np.concatenate([left, middle, right], 1))
        summaries.append(
            f'{name}: median sigma={np.median(sig):.3f}, '
            f'median angle={np.degrees(np.median(ang)):.1f}deg, '
            f'p95 sigma={np.percentile(sig,95):.3f}, '
            f'p95 angle={np.degrees(np.percentile(ang,95)):.1f}deg')

    board = np.concatenate(rows, 0)
    footer = np.full((120, board.shape[1], 3), 24, np.uint8)
    cv2.putText(footer, 'GREEN = original-view prediction; RED = transformed '
                'prediction mapped back. Perfect consistency means overlap.',
                (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.57,
                (240, 240, 240), 1, cv2.LINE_AA)
    for i, text in enumerate(summaries):
        cv2.putText(footer, text, (14, 52 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.53,
                    (220, 220, 220), 1, cv2.LINE_AA)
    board = np.concatenate([board, footer], 0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, board)
    if args.dump:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump)), exist_ok=True)
        np.savez(args.dump, **dumped)
    print(f"wrote {args.out}; sample={target_np['filename']}")
    for s in summaries:
        print(s)


if __name__ == '__main__':
    main()
