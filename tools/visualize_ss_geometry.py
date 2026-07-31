"""Render a human-readable Point2RBox-v2 self-supervision geometry audit.

This intentionally checks the part that is easiest to review visually:
whether an annotation stays on the same object after rot/flp/sca.  The image
transform uses the same Jittor operators as the detector.  Each transformed
point is drawn together with an ID and its inverse-mapped location.

Example:
    CUDA_VISIBLE_DEVICES=3 python tools/visualize_ss_geometry.py \
        --out work_dirs/ss_debug/geometry.png
"""
import argparse
import math
import os

import cv2
import jittor as jt
import numpy as np
from jittor.nn import grid_sample

from jdet.models.networks.point2rbox_v2 import _resized_crop_aa


CLASSES = (
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter')


def load_ann(path):
    points, labels = [], []
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 9 or p[8] not in CLASSES:
                continue
            q = np.asarray(p[:8], np.float32).reshape(4, 2)
            points.append(q.mean(0))
            labels.append(CLASSES.index(p[8]))
    return np.asarray(points, np.float32), np.asarray(labels, np.int32)


def rotate_exact(image, points, angle):
    h, w = image.shape[:2]
    x = jt.array(image.transpose(2, 0, 1)[None].astype(np.float32))
    c, s = math.cos(angle), math.sin(angle)
    tf = jt.array(np.float32([[c, -s], [s, c]]))
    xr, yr = jt.linspace(-1, 1, w), jt.linspace(-1, 1, h)
    yy, xx = jt.meshgrid(yr, xr)
    grid = jt.stack([xx, yy], -1).unsqueeze(0)
    grid = grid.reshape(-1, 2).matmul(tf).view(1, h, w, 2)
    y = grid_sample(x, grid, 'bilinear', 'reflection',
                    align_corners=True).numpy()[0].transpose(1, 2, 0)
    ctr = np.float32([w / 2, h / 2])
    q = (points - ctr) @ np.float32([[c, s], [-s, c]]) + ctr
    return np.clip(y, 0, 255).astype(np.uint8), q


def scale_exact(image, points, scale):
    h, w = image.shape[:2]
    x = jt.array(image.transpose(2, 0, 1)[None].astype(np.float32))
    y = _resized_crop_aa(x, int(h / scale), int(w / scale), h, w)
    y = y.numpy()[0].transpose(1, 2, 0)
    return np.clip(y, 0, 255).astype(np.uint8), points * scale


def draw(im, points, labels, title, color=(40, 40, 255)):
    out = im.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 44), (20, 20, 20), -1)
    cv2.putText(out, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 2, cv2.LINE_AA)
    for i, (p, lab) in enumerate(zip(points, labels)):
        x, y = np.rint(p).astype(int)
        if not (0 <= x < out.shape[1] and 0 <= y < out.shape[0]):
            continue
        cv2.drawMarker(out, (x, y), color, cv2.MARKER_CROSS, 18, 2)
        text = f'{i}:{CLASSES[int(lab)][:6]}'
        cv2.putText(out, text, (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, text, (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)
    return out


def choose_sample(images_dir, anns_dir):
    best = None
    for name in sorted(os.listdir(anns_dir)):
        if not name.endswith('.txt'):
            continue
        pts, labels = load_ann(os.path.join(anns_dir, name))
        image_path = os.path.join(images_dir, name[:-4] + '.png')
        if not os.path.exists(image_path):
            continue
        # Prefer a sufficiently crowded patch: correspondence errors are
        # easier to spot than on a one-object image.
        score = min(len(pts), 40)
        if best is None or score > best[0]:
            best = (score, image_path, pts, labels)
    if best is None:
        raise RuntimeError('no image/annotation pair found')
    return best[1:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images-dir',
                   default='/root/data/split_ss_dota/trainval/images')
    p.add_argument('--anns-dir',
                   default='/root/data/split_ss_dota/trainval/annfiles')
    p.add_argument('--out', default='work_dirs/ss_debug/geometry.png')
    p.add_argument('--rotation-deg', type=float, default=67.5)
    p.add_argument('--scale', type=float, default=0.70)
    args = p.parse_args()

    jt.flags.use_cuda = 1
    path, points, labels = choose_sample(args.images_dir, args.anns_dir)
    image = cv2.imread(path)
    if image is None:
        raise RuntimeError(f'failed to read {path}')
    h, w = image.shape[:2]

    angle = math.radians(args.rotation_deg)
    rot_im, rot_pts = rotate_exact(image, points, angle)
    flp_im, flp_pts = image[::-1].copy(), points.copy()
    flp_pts[:, 1] = h - flp_pts[:, 1]
    sca_im, sca_pts = scale_exact(image, points, args.scale)

    panels = [
        draw(image, points, labels, f'ORIGINAL | {os.path.basename(path)}'),
        draw(rot_im, rot_pts, labels,
             f'ROT +{args.rotation_deg:.1f} deg | exact grid_sample'),
        draw(flp_im, flp_pts, labels, 'VERTICAL FLIP | y -> H-y'),
        draw(sca_im, sca_pts, labels,
             f'SCALE {args.scale:.2f} | zero-pad + antialias resize'),
    ]
    board = np.concatenate([
        np.concatenate(panels[:2], axis=1),
        np.concatenate(panels[2:], axis=1)], axis=0)
    footer = np.full((90, board.shape[1], 3), 28, np.uint8)
    lines = [
        'Review rule: every ID must remain centered on the SAME physical object in all four views.',
        'Red cross = transformed weak-supervision point. IDs are stable correspondences.',
        'Scale view intentionally contains black right/bottom padding (official torchvision resized_crop semantics).',
    ]
    for i, text in enumerate(lines):
        cv2.putText(footer, text, (16, 25 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235),
                    1, cv2.LINE_AA)
    board = np.concatenate([board, footer], axis=0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, board)
    print(f'wrote {args.out} ({board.shape[1]}x{board.shape[0]})')


if __name__ == '__main__':
    main()
