"""Dump golden TED reference outputs with the PyTorch implementation.

Run inside the p2r-torch env, from the Point2RBox-v3 reference repo root
(so that third_parties/ted is importable):

    python tools/dump_ted_reference.py \
        --ted-dir /root/ref/Point2RBox-v3/third_parties/ted \
        --image /root/data/split_ss_dota/trainval/images/<some>.png \
        --out /root/work/B/staging/tests/data/ted_reference.npz

Saves: input tensors (fixed-seed random + optional real image) and the 4
output maps for each, for tests/test_ted.py to compare against.
"""
import argparse
import os
import sys

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ted-dir', required=True)
    p.add_argument('--image', default=None, help='optional real image (png/jpg)')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.ted_dir)))
    from ted.ted import TED  # noqa: E402

    model = TED()
    model.load_state_dict(torch.load(os.path.join(args.ted_dir, 'ted.pth'),
                                     map_location='cpu'))
    model.eval()

    arrays = {}
    torch.manual_seed(0)
    x_rand = torch.rand(2, 3, 352, 352)
    arrays['input_rand'] = x_rand.numpy()
    with torch.no_grad():
        outs = model(x_rand)
    for i, o in enumerate(outs):
        arrays[f'out_rand_{i}'] = o.numpy()

    if args.image:
        import cv2
        img = cv2.imread(args.image)
        assert img is not None, args.image
        img = cv2.resize(img, (1024, 1024)).astype(np.float32)
        x_img = torch.from_numpy(img.transpose(2, 0, 1))[None] / 255.0
        arrays['input_img'] = x_img.numpy()
        with torch.no_grad():
            outs = model(x_img)
        for i, o in enumerate(outs):
            arrays[f'out_img_{i}'] = o.numpy()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f'saved {list(arrays)} -> {args.out}')


if __name__ == '__main__':
    main()
