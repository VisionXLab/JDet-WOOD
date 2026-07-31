"""Golden reference for Point2RBoxV3Head.loss_by_feat (M5 stage-2 parity).

Run in p2r-torch:
    PYTHONPATH=/root/ref/Point2RBox-v3 python tools/dump_v3_head_loss_reference.py \
        --out tests/data/v3_head_loss_reference.npz \
        [--sam --image-dir /root/data/split_ss_dota/trainval/images]

Setup: dual-stream-like batch of 4 "images" (2 originals + 2 aug views),
paired instances across views (bmsk pairs), one syn (copy-paste) instance,
square_cls / edge_loss_cls / voronoi-override classes covered.
Real DOTA patch crops (256x256) as images_no_copypaste (cv2.watershed needs
real structure); fixed positive tensor as TED edges.

The loss path is fully deterministic given inputs (the only np.random in the
loss file is inside a debug plot helper), so no rand monkeypatching is
needed. Dumps loss_dict + d(total)/d(feats) for epoch=5 and epoch=6.

--sam: second golden with the official _base_sam-dotav1-0.py filter config and
sam_instance_thr=4 (groups here have J<=4 -> SAM branch taken), MobileSAM on
CPU for strict-fp32 reproducibility (TF32 lesson from test_sam.py).
"""
import argparse
import os

import numpy as np
# NOTE: torch is imported inside functions so that the p2r-jittor test can
# `from tools.dump_v3_head_loss_reference import GT, SS_INFO, SIZES` without
# needing torch installed.


BBOX_HEAD_CFG = dict(
    num_classes=15,
    in_channels=256,
    feat_channels=256,
    strides=[8, 16, 32, 64, 128],
    use_adaptive_scale=False,
    edge_loss_start_epoch=6,
    joint_angle_start_epoch=1,
    voronoi_type='standard',
    voronoi_thres=dict(
        default=[0.994, 0.005],
        override=(([2, 11], [0.999, 0.6]), ([7, 8, 10, 14], [0.95, 0.005]))),
    square_cls=[1, 9, 11],
    edge_loss_cls=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13],
    post_process={11: 1.2},
    angle_coder=dict(
        type='PSCCoder', angle_version='le90', dual_freq=False, num_step=3,
        thr_mod=0),
    loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0,
                  alpha=0.25, loss_weight=1.0),
    loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0),
    loss_overlap=dict(type='GaussianOverlapLoss', loss_weight=10.0, lamb=0),
    loss_voronoi=dict(type='VoronoiWatershedLoss', loss_weight=5.0),
    loss_bbox_edg=dict(type='EdgeLoss', loss_weight=0.3),
    loss_ss=dict(type='Point2RBoxV2ConsistencyLoss', loss_weight=1.0),
)

# 4 targets: [img0 orig, img1 orig, img2 = aug of img0, img3 = aug of img1]
# bids = (batch, syn, view, instance); aug batch ids offset by B=2, view=1.
# instance 3 is syn (copy-paste style, ins=0) -> excluded from vor/ovl groups.
# classes: 9,11 square; 2 voronoi-override + edge cls; 4,5 edge cls; 0 plain.
GT = [
    dict(bboxes=[[100.0, 120.0, 40.0, 40.0, 0.0],
                 [60.0, 60.0, 80.0, 30.0, 0.6],
                 [190.0, 200.0, 60.0, 24.0, -0.8],
                 [200.0, 60.0, 32.0, 16.0, 1.2]],
         labels=[9, 4, 2, 5],
         bids=[[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 1, 0, 0]]),
    dict(bboxes=[[80.0, 190.0, 50.0, 20.0, -1.0],
                 [180.0, 80.0, 60.0, 60.0, 0.3]],
         labels=[0, 11],
         bids=[[1, 0, 0, 4], [1, 0, 0, 5]]),
    dict(bboxes=[[120.0, 100.0, 40.0, 40.0, 0.35],
                 [70.0, 80.0, 80.0, 30.0, 0.95],
                 [180.0, 190.0, 60.0, 24.0, -0.45]],
         labels=[9, 4, 2],
         bids=[[2, 0, 1, 1], [2, 0, 1, 2], [2, 0, 1, 3]]),
    dict(bboxes=[[95.0, 175.0, 50.0, 20.0, -0.65],
                 [170.0, 95.0, 60.0, 60.0, 0.65]],
         labels=[0, 11],
         bids=[[3, 0, 1, 4], [3, 0, 1, 5]]),
]
SS_INFO = ('rot', 0.35)
IMG_HW = 256
SIZES = [(32, 32), (16, 16), (8, 8), (4, 4), (2, 2)]


def load_images(image_dir):
    """4 deterministic 256x256 crops from a fixed DOTA patch, as CHW float
    tensors in data_preprocessor-style (normalized) space."""
    import cv2
    path = os.path.join(image_dir, 'P0000__1024__0___0.png')
    img = cv2.imread(path)
    assert img is not None, path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53], np.float32)
    std = np.array([58.395, 57.12, 57.375], np.float32)
    crops = []
    for (y, x) in [(0, 0), (256, 256), (512, 128), (128, 512)]:
        c = (img[y:y + IMG_HW, x:x + IMG_HW] - mean) / std
        crops.append(c.transpose(2, 0, 1))
    return np.stack(crops)  # (4, 3, 256, 256)


def build_head(sam, arrays):
    import torch
    from mmengine.registry import init_default_scope
    init_default_scope('mmrotate')
    from mmrotate.models.dense_heads.point2rbox_v3_head import Point2RBoxV3Head

    cfg = {k: (v.copy() if isinstance(v, dict) else v)
           for k, v in BBOX_HEAD_CFG.items()}
    if sam:
        scope = {}
        exec(open('configs/point2rbox_v3/_base_sam-dotav1-0.py').read(), scope)
        cfg['loss_voronoi'] = dict(
            type='VoronoiWatershedLoss', loss_weight=5.0,
            mask_filter_config=scope['mask_filter_config'],
            sam_instance_thr=scope['sam_instance_thr'],
            sam_sample_rules=scope['sam_sample_rules'])
    torch.manual_seed(0)
    head = Point2RBoxV3Head(**cfg)
    head.train()
    for k, v in head.state_dict().items():
        arrays[f'sd::{k}'] = v.detach().cpu().numpy()
    return head


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--sam', action='store_true')
    p.add_argument('--image-dir',
                   default='/root/data/split_ss_dota/trainval/images')
    args = p.parse_args()

    import torch
    from mmengine.structures import InstanceData
    from mmrotate.structures.bbox import RotatedBoxes

    arrays = {}
    head = build_head(args.sam, arrays)
    if args.sam:
        # deterministic strict-fp32 SAM golden (avoid GPU TF32 noise).
        #
        # UPSTREAM BUG (documented, deliberately NOT reproduced): MobileSAM's
        # build_sam calls .eval() BEFORE load_state_dict; TinyViT.Attention's
        # eval hook caches `ab = attention_biases[:, idxs]` as a
        # non-persistent buffer, so the cache keeps RANDOM-INIT biases and
        # the loaded weights never reach the eval forward. Upstream v3
        # training therefore ran SAM with unseeded random attention biases —
        # unreproducible by construction. Golden uses the corrected
        # semantics (re-eval AFTER loading), which is what the jt port
        # (loads real weights, no stale cache) computes.
        import mmrotate.models.losses.point2rbox_v2_loss as pl
        from mobile_sam import sam_model_registry
        _orig = pl.segment_anything
        sam = sam_model_registry['vit_t'](
            checkpoint='/root/work/B/ref-mobilesam/weights/mobile_sam.pt')
        sam.eval()  # refresh the ab cache from the LOADED biases
        sam.to('cpu')
        def _cpu_sam(*a, **kw):
            kw['device'] = 'cpu'
            kw['sam_checkpoint'] = \
                '/root/work/B/ref-mobilesam/weights/mobile_sam.pt'
            kw['model_type'] = 'vit_t'
            return _orig(*a, **kw)
        # ref 函数体里 `segment_anything.sam_model` 按名解析到重绑后的全局
        # （即本 wrapper），缓存必须预植到 wrapper 上
        _cpu_sam.sam_model = sam
        _cpu_sam.model_type = 'vit_t'
        pl.segment_anything = _cpu_sam

    rng = np.random.RandomState(0)
    feats_np = [rng.randn(4, 256, h, w).astype(np.float32) for h, w in SIZES]
    for i, f in enumerate(feats_np):
        arrays[f'feat_{i}'] = f
    images = load_images(args.image_dir)
    arrays['images'] = images
    edges = np.abs(rng.randn(4, 1, IMG_HW, IMG_HW)).astype(np.float32)
    arrays['edges'] = edges

    gts, metas = [], []
    for t in GT:
        gi = InstanceData()
        gi.bboxes = RotatedBoxes(torch.tensor(t['bboxes'], dtype=torch.float32))
        gi.labels = torch.tensor(t['labels'], dtype=torch.int64)
        gi.bids = torch.tensor(t['bids'], dtype=torch.int64)
        gts.append(gi)
        metas.append(dict(ss=SS_INFO))

    for epoch in (5, 6):
        feats = [torch.from_numpy(f.copy()).requires_grad_(True)
                 for f in feats_np]
        cls_scores, bbox_preds, angle_preds = head(feats)
        head.epoch = epoch
        head.images_no_copypaste = torch.from_numpy(images.copy())
        head.edges = torch.from_numpy(edges.copy())
        loss_dict = head.loss_by_feat(cls_scores, bbox_preds, angle_preds,
                                      gts, metas)
        total = sum(v for v in loss_dict.values())
        total.backward()
        for k, v in loss_dict.items():
            arrays[f'ep{epoch}::{k}'] = np.float64(v.item())
            print(f'  ep{epoch} {k} = {v.item():.6f}')
        for i, f in enumerate(feats):
            arrays[f'ep{epoch}::grad_feat_{i}'] = f.grad.numpy()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f'saved {len(arrays)} arrays -> {args.out} (sam={args.sam})')


if __name__ == '__main__':
    main()
