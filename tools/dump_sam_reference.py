"""Dump golden MobileSAM reference outputs with the PyTorch implementation.

Run inside the p2r-torch env (mobile_sam pip package installed):

    python tools/dump_sam_reference.py \
        --checkpoint /root/ref/Point2RBox-v3/mobile_sam.pt \
        --image /root/data/split_ss_dota/trainval/images/<some>.png \
        --out tests/data/sam_reference.npz

Dumps, for a fixed image and a fixed set of point prompts:
- the preprocessed input (after ResizeLongestSide + normalize + pad)
- image_embedding from the TinyViT encoder
- low-res mask logits + upscaled masks for each prompt case
Prompt cases mirror Point2RBox-v3 usage: 1 positive point + N negative points.
"""
import argparse
import os

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    from mobile_sam import sam_model_registry, SamPredictor
    import cv2

    # Dump on CPU in strict fp32: on A100, torch defaults cudnn TF32 on,
    # which bakes ~1e-3 noise into the golden reference itself.
    sam = sam_model_registry['vit_t'](checkpoint=args.checkpoint)
    sam.eval()
    device = 'cpu'
    sam.to(device)
    predictor = SamPredictor(sam)

    img = cv2.imread(args.image)
    assert img is not None, args.image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    predictor.set_image(img)

    arrays = {'image_rgb': img}
    arrays['image_embedding'] = predictor.features.detach().cpu().numpy()

    h, w = img.shape[:2]
    rng = np.random.RandomState(0)
    cases = {
        'single_pos': (np.float32([[w * 0.5, h * 0.5]]), np.int32([1])),
        'pos_neg': (
            np.float32([[w * 0.5, h * 0.5], [w * 0.25, h * 0.25],
                        [w * 0.75, h * 0.75]]),
            np.int32([1, 0, 0])),
        'rand_pts': (
            np.float32(rng.rand(5, 2) * [w, h]),
            np.int32([1, 0, 0, 0, 0])),
    }
    for name, (pts, lbls) in cases.items():
        masks, scores, logits = predictor.predict(
            point_coords=pts, point_labels=lbls, multimask_output=True)
        arrays[f'{name}_points'] = pts
        arrays[f'{name}_labels'] = lbls
        arrays[f'{name}_masks'] = masks.astype(np.uint8)
        arrays[f'{name}_scores'] = scores
        arrays[f'{name}_logits'] = logits

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f'saved {list(arrays)} -> {args.out}')


if __name__ == '__main__':
    main()
