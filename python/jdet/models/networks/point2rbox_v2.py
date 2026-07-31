"""Point2RBoxV2 detector（Jittor 移植）。

源：Point2RBox-v3 的 `point2rbox_v2.py`。
targets 约定：
    targets[i]['rboxes'] : jt.float32 (N, 5) xywhr
    targets[i]['labels'] : jt.int     (N,)
    targets[i]['bids']   : jt.int32   (N, 4) = (batch, syn, view, instance)
    targets[i]['ss']     : tuple (aug_type, aug_val)，aug_type ∈ {'rot','flp','sca'}
dual-stream 拼接后 targets 长度翻倍（原批在前、aug 批在后）。
"""
import copy
import math

import cv2
import numpy as np
import jittor as jt
from jittor import nn
from jittor.nn import grid_sample

from jdet.utils.registry import MODELS, BACKBONES, HEADS, NECKS, build_from_cfg

try:
    from third_parties.ted.ted import TED
except ImportError:  # tools/run_net.py 只把 tools/ 放进 sys.path，补仓库根
    import os as _os
    import sys as _sys
    _repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__),
                                                '../../../..'))
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)
    from third_parties.ted.ted import TED


def _aa_bilinear_weights(in_size, out_size):
    """Weights of torchvision's antialiased bilinear tensor resize."""
    scale = in_size / out_size
    support = scale if scale >= 1.0 else 1.0
    weights = np.zeros((out_size, in_size), dtype=np.float32)
    for i in range(out_size):
        center = scale * (i + 0.5)
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size)
        js = np.arange(xmin, xmax)
        row = 1.0 - np.abs((js + 0.5 - center) / scale)
        row = np.clip(row, 0.0, None)
        total = row.sum()
        if total > 0:
            weights[i, xmin:xmax] = row / total
    return weights


def _resized_crop_aa(images, crop_h, crop_w, out_h, out_w):
    """Match torchvision resized_crop with an out-of-bounds top-left crop."""
    batch, channels, height, width = images.shape
    padded = images
    if crop_h > height or crop_w > width:
        canvas = jt.zeros(
            (batch, channels, max(crop_h, height), max(crop_w, width)),
            dtype=images.dtype)
        canvas[:, :, :height, :width] = images
        padded = canvas
    padded = padded[:, :, :crop_h, :crop_w]
    weight_h = jt.array(_aa_bilinear_weights(crop_h, out_h))
    weight_w = jt.array(_aa_bilinear_weights(crop_w, out_w))
    flat = padded.reshape(batch * channels, crop_h, crop_w)
    weight_h = weight_h.unsqueeze(0).expand(
        (batch * channels, out_h, crop_h))
    weight_w = weight_w.transpose(1, 0).unsqueeze(0).expand(
        (batch * channels, crop_w, out_w))
    flat = jt.matmul(weight_h, flat)
    flat = jt.matmul(flat, weight_w)
    return flat.reshape(batch, channels, out_h, out_w)


def get_single_pattern(image, bbox, label, square_cls):
    if bbox[2] < 16 or bbox[3] < 16 or bbox[2] > 512 or bbox[3] > 512:
        raise ValueError('pattern size out of range')

    def obb2poly(obb):
        cx, cy, w, h, t = obb
        dw, dh = (w - 1) / 2, (h - 1) / 2
        cost = np.cos(t)
        sint = np.sin(t)
        mrot = np.float32([[cost, -sint], [sint, cost]])
        poly = np.float32([[-dw, -dh], [dw, -dh], [dw, dh], [-dw, dh]])
        return np.matmul(poly, mrot.T) + np.float32([cx, cy])

    def get_pattern_gaussian(w, h):
        w, h = int(w), int(h)
        yy = jt.arange(h).float()
        xx = jt.arange(w).float()
        y = (yy[:, None].expand((h, w)) - h / 2) / (h / 2)
        x = (xx[None, :].expand((h, w)) - w / 2) / (w / 2)
        ox, oy = (jt.randn(2).clamp(-3, 3) * 0.15)
        sx, sy = (jt.rand(2) * 0.5 + 1)
        z = jt.exp(-((x - ox) * sx) ** 2 - ((y - oy) * sy) ** 2) * 0.5 + 0.5
        return z

    cx, cy, w, h, t = bbox
    w, h = int(w), int(h)
    poly = obb2poly([cx, cy, w, h, t])

    pts1 = poly[0:3]
    pts2 = np.float32([[-1, -1], [1, -1], [1, 1]])
    M = cv2.getAffineTransform(pts1, pts2)
    M = np.concatenate((M, ((0, 0, 1),)), 0)

    H, W = image.shape[1:3]
    T = np.array([[2 / W, 0, -1],
                  [0, 2 / H, -1],
                  [0, 0, 1]])
    theta = T @ np.linalg.inv(M)
    theta = jt.array(theta[:2, :].astype(np.float32))[None]
    grid = nn.affine_grid(theta, [1, 3, h, w], align_corners=True)
    chip = nn.grid_sample(image[None], grid, mode='bilinear', align_corners=True)[0]

    alpha = get_pattern_gaussian(chip.shape[-1], chip.shape[-2])[None]
    chip = jt.concat([chip, alpha])

    w_ = float(bbox[2] * (0.7 + 0.5 * np.random.rand()))
    h_ = float(bbox[3] * (0.7 + 0.5 * np.random.rand()))
    t_ = float(np.pi * np.random.rand())
    if label in square_cls:
        t_ = 0.0
    cosa = abs(math.cos(t_))
    sina = abs(math.sin(t_))
    sx, sy = int(math.ceil(cosa * w_ + sina * h_)), int(math.ceil(sina * w_ + cosa * h_))
    theta = np.float32(
        [[1 / w_ * math.cos(t_), 1 / w_ * math.sin(t_), 0],
         [1 / h_ * math.sin(-t_), 1 / h_ * math.cos(t_), 0]])
    theta[:, :2] = theta[:, :2] @ np.float32([[sx, 0], [0, sy]])
    grid = nn.affine_grid(jt.array(theta)[None], (1, 1, sy, sx), align_corners=True)
    chip = nn.grid_sample(chip[None], grid, mode='nearest', align_corners=True)[0]
    bbox = np.float32([sx / 2, sy / 2, w_, h_, t_])
    return (chip, bbox, label)


def get_copy_paste_cache(images, bboxes, labels, square_cls, num_copies):
    bboxes = bboxes.detach().numpy()
    labels = labels.detach().numpy()
    patterns = []
    for b, l in zip(bboxes, labels):
        try:
            p = get_single_pattern(images, b, l, square_cls)
            patterns.append(p)
            if len(patterns) > num_copies:
                break
        except Exception:
            pass
    return patterns


@MODELS.register_module()
class Point2RBoxV2(nn.Module):
    """Implementation of Point2RBox-v2."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 rotate_range=(0.25, 0.75),
                 scale_range=(0.5, 0.9),
                 ss_prob=[0.6, 0.15, 0.25],
                 copy_paste_start_epoch=6,
                 num_copies=10,
                 debug=False,
                 data_preprocessor=None,
                 train_cfg=None,
                 test_cfg=None):
        super(Point2RBoxV2, self).__init__()
        self.backbone = build_from_cfg(backbone, BACKBONES)
        self.neck = build_from_cfg(neck, NECKS) if neck is not None else None
        if train_cfg is not None:
            bbox_head = dict(bbox_head, train_cfg=train_cfg)
        if test_cfg is not None:
            bbox_head = dict(bbox_head, test_cfg=test_cfg)
        self.bbox_head = build_from_cfg(bbox_head, HEADS)

        self.rotate_range = rotate_range
        self.scale_range = scale_range
        self.ss_prob = ss_prob
        self.copy_paste_start_epoch = copy_paste_start_epoch
        self.num_copies = num_copies
        self.debug = debug
        self.copy_paste_cache = None
        self.epoch = 0

        # 官方 config 的 data_preprocessor（mean/std/bgr_to_rgb 等）在 JDet 里由
        # dataset transforms 完成；此处只保留 TED 反归一化需要的 mean/std
        if data_preprocessor is not None:
            mean = np.array(data_preprocessor['mean'], dtype=np.float32).reshape(3, 1, 1)
            std = np.array(data_preprocessor['std'], dtype=np.float32).reshape(3, 1, 1)
        else:
            mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)
            std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(3, 1, 1)
        self.preprocess_mean = jt.array(mean).stop_grad()
        self.preprocess_std = jt.array(std).stop_grad()

        self.ted_model = TED()
        import pickle
        import os
        ted_pkl = os.path.join(os.path.dirname(__file__),
                               '../../../../third_parties/ted/ted.pkl')
        with open(os.path.abspath(ted_pkl), 'rb') as f:
            sd = pickle.load(f)
        self.ted_model.load_parameters({k: jt.array(v) for k, v in sd.items()})
        self.ted_model.eval()
        for p in self.ted_model.parameters():
            p.stop_grad()

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.bbox_head.epoch = epoch

    def train(self):
        """Enter train mode while preserving ResNet ``norm_eval=True``.

        Jittor's base Module.train toggles descendants by DFS and does not
        dispatch to an overridden child train method.  Explicitly re-enter
        backbone train mode so its frozen-stage and BN policy is restored.
        """
        super(Point2RBoxV2, self).train()
        self.backbone.train()
        self.ted_model.eval()
        return self

    def rotate_crop(self, batch_inputs, rot=0., size=(768, 768),
                    targets=None, padding='reflection'):
        """旋转 + 中心裁剪（gt 同步变换）。targets 就地更新 'rboxes'。"""
        n, c, h, w = batch_inputs.shape
        size_h, size_w = size
        crop_h = (h - size_h) // 2
        crop_w = (w - size_w) // 2
        if rot != 0:
            cosa, sina = math.cos(rot), math.sin(rot)
            tf = jt.array(np.float32([[cosa, -sina], [sina, cosa]]))
            x_range = jt.linspace(-1, 1, w)
            y_range = jt.linspace(-1, 1, h)
            y, x = jt.meshgrid(y_range, x_range)
            grid = jt.stack([x, y], -1).unsqueeze(0).expand((n, h, w, 2))
            grid = grid.reshape(-1, 2).matmul(tf).view(n, h, w, 2)
            batch_inputs = grid_sample(batch_inputs, grid, 'bilinear', padding,
                                       align_corners=True)
            if targets is not None:
                for target in targets:
                    gt_bboxes = target['rboxes']
                    xy, wh, a = gt_bboxes[..., :2], gt_bboxes[..., 2:4], gt_bboxes[..., 4:5]
                    ctr = jt.array(np.float32([[w / 2, h / 2]]))
                    xy = (xy - ctr).matmul(tf.transpose(1, 0)) + ctr
                    a = a + rot
                    target['rboxes'] = jt.concat([xy, wh, a], dim=-1)
        batch_inputs = batch_inputs[..., crop_h:crop_h + size_h,
                                    crop_w:crop_w + size_w]
        if targets is None:
            return batch_inputs
        for target in targets:
            gt_bboxes = target['rboxes']
            xy, wh, a = gt_bboxes[..., :2], gt_bboxes[..., 2:4], gt_bboxes[..., 4:5]
            xy = xy - jt.array(np.float32([[crop_w, crop_h]]))
            target['rboxes'] = jt.concat([xy, wh, a], dim=-1)
        return batch_inputs, targets

    def vflip(self, img):
        return img[:, :, ::-1, :]

    def forward_train(self, images, targets):
        H, W = images.shape[2:4]

        # Set bids: (N, 4) = (batch, syn, view, instance)
        offset = 1
        for i, target in enumerate(targets):
            blen = target['rboxes'].shape[0]
            bids = jt.zeros((blen, 4), dtype='int32')
            bids[:, 0] = i
            bids[:, 3] = jt.arange(0, blen, 1) + offset
            target['bids'] = bids
            offset += blen

        sel_p = float(jt.rand(1).item())
        if sel_p < self.ss_prob[0]:
            # Generate rotated images and gts
            rot = math.pi * (
                float(jt.rand(1).item()) *
                (self.rotate_range[1] - self.rotate_range[0]) + self.rotate_range[0])
            ss = ('rot', rot)
            targets_aug = copy.deepcopy(targets)
            images_aug, targets_aug = self.rotate_crop(
                images, rot, [H, W], targets_aug, 'reflection')
            for target in targets_aug:
                target['bids'][:, 0] += len(targets)
                target['bids'][:, 2] = 1
        elif sel_p < self.ss_prob[0] + self.ss_prob[1]:
            # Generate flipped images and gts
            ss = ('flp', 0)
            images_aug = self.vflip(images)
            targets_aug = copy.deepcopy(targets)
            for target in targets_aug:
                b = target['rboxes']
                # RotatedBoxes.flip_('vertical')：y=H-y, a=-a
                target['rboxes'] = jt.concat(
                    [b[:, 0:1], H - b[:, 1:2], b[:, 2:4], -b[:, 4:5]], dim=-1)
                target['bids'][:, 0] += len(targets)
                target['bids'][:, 2] = 1
        else:
            # Generate scaled images and gts
            sca = (float(jt.rand(1).item()) *
                   (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0])
            ss = ('sca', sca)
            # torchvision resized_crop：越界区补零，再 antialias bilinear。
            ch, cw = int(H / sca), int(W / sca)
            images_aug = _resized_crop_aa(images, ch, cw, H, W)
            targets_aug = copy.deepcopy(targets)
            for target in targets_aug:
                b = target['rboxes']
                # RotatedBoxes.rescale_([sca, sca])：xy/wh 乘 sca，角度不变
                target['rboxes'] = jt.concat(
                    [b[:, :4] * sca, b[:, 4:5]], dim=-1)
                target['bids'][:, 0] += len(targets)
                target['bids'][:, 2] = 1

        # Official ordering is intentional: Voronoi supervision images and TED
        # edges are captured before copy-paste.  Copy-paste only changes the
        # augmented image subsequently sent through the backbone.
        supervision_images = jt.concat([images, images_aug], 0)
        self.bbox_head.images = supervision_images
        if self.epoch >= self.bbox_head.edge_loss_start_epoch:
            with jt.no_grad():
                batch_edges = self.ted_model(
                    supervision_images * self.preprocess_std + self.preprocess_mean)
                self.bbox_head.edges = batch_edges[3].clamp(0)

        # copy-paste（epoch >= copy_paste_start_epoch 后由上一轮 cache 提供）
        if self.copy_paste_cache and len(targets_aug) == len(self.copy_paste_cache):
            for i in range(len(targets_aug)):
                target, patterns = targets_aug[i], self.copy_paste_cache[i]
                if not patterns:
                    continue
                bboxes_paste = []
                labels_paste = []
                for p, b, l in patterns:
                    ph, pw = p.shape[1:3]
                    ox = np.random.randint(0, images_aug.shape[-1] - pw)
                    oy = np.random.randint(0, images_aug.shape[-2] - ph)
                    region = images_aug[i, :, oy:oy + ph, ox:ox + pw]
                    images_aug[i, :, oy:oy + ph, ox:ox + pw] = \
                        region * (1 - p[3:4]) + p[:3] * p[3:4]
                    bboxes_paste.append(b + np.float32((ox, oy, 0, 0, 0)))
                    labels_paste.append(l)
                target['rboxes'] = jt.concat(
                    [target['rboxes'], jt.array(np.float32(bboxes_paste))], 0)
                target['labels'] = jt.concat(
                    [target['labels'], jt.array(np.int32(labels_paste)).cast(target['labels'].dtype)], 0)
                bids_paste = jt.array(np.int32([i, 1, 0, 0])).expand(
                    (len(labels_paste), 4))
                target['bids'] = jt.concat([target['bids'], bids_paste], 0)

        images_all = jt.concat([images, images_aug], 0)

        targets_all = []
        for target in targets + targets_aug:
            t = dict(target)
            t['ss'] = ss
            targets_all.append(t)

        feat = self.backbone(images_all)
        if self.neck:
            feat = self.neck(feat)

        results_list = self.bbox_head.predict(feat, targets_all)

        # Update point annotations with predicted rbox（bids syn 位==0 的实例）
        for target, results in zip(targets_all, results_list):
            mask = (target['bids'][:, 1] == 0).unsqueeze(-1)
            target['rboxes'] = jt.where(mask.expand(target['rboxes'].shape),
                                        results['bboxes'], target['rboxes'])
            lmask = target['bids'][:, 1] == 0
            target['labels'] = jt.where(lmask, results['labels'].cast(target['labels'].dtype),
                                        target['labels'])

        losses = self.bbox_head.loss(feat, targets_all)

        if self.epoch >= self.copy_paste_start_epoch:
            self.copy_paste_cache = []
            for img, results in zip(images, results_list):
                self.copy_paste_cache.append(get_copy_paste_cache(
                    img, results['bboxes'], results['labels'],
                    self.bbox_head.square_cls, self.num_copies))

        return losses

    def forward_test(self, images, targets):
        feat = self.backbone(images)
        if self.neck:
            feat = self.neck(feat)
        return self.bbox_head.get_bboxes(feat, targets)

    def execute(self, images, targets):
        # 不能只看 'rboxes'（val/伪标签生成的 targets 也带 GT）——按训练态分发
        if self.is_training() and 'rboxes' in targets[0]:
            return self.forward_train(images, targets)
        return self.forward_test(images, targets)
