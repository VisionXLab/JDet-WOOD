"""Dump golden outputs for the v3 detector module-level functions.

Run in p2r-torch:
    cd /root/ref/Point2RBox-v3 && PYTHONPATH=/root/ref/Point2RBox-v3 \
    python /root/work/B/Point2RBox-v3-jittor/tools/dump_v3_detector_funcs_reference.py \
        --image /root/data/split_ss_dota/trainval/images/P0000__1024__0___0.png \
        --out /root/work/B/Point2RBox-v3-jittor/tests/data/v3_detector_funcs_reference.npz

Covers: gaussian_2d (generic SPD sigma), voronoi_diagram_watershed
('standard' and 'gaussian-orientation' branches, with voronoi_thres
override), get_single_pattern (fixed injected randomness via monkeypatch).
"""
import argparse
import os
import types

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    import cv2
    from mmrotate.models.detectors import point2rbox_v3 as ref

    arrays = {}
    rs = np.random.RandomState(0)

    # ---- gaussian_2d ----
    N = 64
    xy = rs.rand(N, 2).astype(np.float32) * 512
    mu = np.float32([[256, 256]])
    a, b, c = 3000.0, 800.0, 1500.0          # SPD: a*c > b^2
    sigma = np.float32([[[a, b], [b, c]]])
    g = ref.gaussian_2d(torch.from_numpy(xy), torch.from_numpy(mu),
                        torch.from_numpy(sigma))
    gn = ref.gaussian_2d(torch.from_numpy(xy), torch.from_numpy(mu),
                         torch.from_numpy(sigma), normalize=True)
    arrays.update(g2d_xy=xy, g2d_mu=mu, g2d_sigma=sigma,
                  g2d_out=g.numpy(), g2d_out_norm=gn.numpy())

    # ---- image tensor (normalized like data_preprocessor would) ----
    img = cv2.imread(args.image)
    assert img is not None, args.image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean = np.float32([123.675, 116.28, 103.53])
    std = np.float32([58.395, 57.12, 57.375])
    img_chw = ((img - mean) / std).transpose(2, 0, 1)
    image_t = torch.from_numpy(img_chw)
    arrays['image_chw'] = img_chw

    # ---- voronoi_diagram_watershed via SimpleNamespace self ----
    J = 6
    mu_pts = np.float32(rs.rand(J, 2) * 900 + 60)
    labels = np.int64([0, 2, 7, 9, 11, 14])
    # SPD sigmas from R diag R^T
    thetas = rs.rand(J) * np.pi
    w2 = rs.rand(J) * 3000 + 1000
    h2 = rs.rand(J) * 1500 + 500
    sigmas = []
    for t, ww, hh in zip(thetas, w2, h2):
        R = np.float32([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        sigmas.append(R @ np.diag([ww, hh]).astype(np.float32) @ R.T)
    sigmas = np.stack(sigmas).astype(np.float32)
    arrays.update(vor_mu=mu_pts, vor_labels=labels.astype(np.int32),
                  vor_sigma=sigmas)

    voronoi_thres = dict(
        default=[0.994, 0.005],
        override=(([2, 11], [0.999, 0.6]), ([7, 8, 10, 14], [0.95, 0.005])))

    for vtype in ('standard', 'gaussian-orientation'):
        fake_head = types.SimpleNamespace(
            voronoi_thres=voronoi_thres, num_classes=15, voronoi_type=vtype)
        fake_self = types.SimpleNamespace(bbox_head=fake_head)
        sig = torch.from_numpy(sigmas) if vtype != 'standard' else None
        out = ref.Point2RBoxV3.voronoi_diagram_watershed(
            fake_self, torch.from_numpy(mu_pts), sig,
            torch.from_numpy(labels), image_t)
        out = [float(v) for v in out]
        arrays[f'vor_out_{vtype}'] = np.float32(out)

    # ---- get_single_pattern with injected fixed randomness ----
    fixed = dict(randn2=np.float32([0.3, -0.7]),
                 rand2=np.float32([0.25, 0.6]),
                 rand3=np.float64([0.1, 0.4, 0.8]))
    arrays.update(gsp_randn2=fixed['randn2'], gsp_rand2=fixed['rand2'],
                  gsp_rand3=np.float32(fixed['rand3']))

    randn_q = [torch.from_numpy(fixed['randn2'])]
    rand_q = [torch.from_numpy(fixed['rand2'])]
    nprand_q = list(fixed['rand3'])
    orig = (torch.randn, torch.rand, np.random.rand)
    torch.randn = lambda *a, **k: randn_q.pop(0)
    torch.rand = lambda *a, **k: rand_q.pop(0)
    np.random.rand = lambda *a, **k: nprand_q.pop(0)
    try:
        bbox = np.float32([300, 400, 80, 48, 0.4])
        chip, obbox, olabel = ref.get_single_pattern(image_t, bbox, 5, [1, 9, 11])
    finally:
        torch.randn, torch.rand, np.random.rand = orig
    arrays.update(gsp_bbox=bbox, gsp_chip=chip.numpy(),
                  gsp_obbox=obbox, gsp_label=np.int32(olabel))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f'saved {sorted(arrays)} -> {args.out}')


if __name__ == '__main__':
    main()
