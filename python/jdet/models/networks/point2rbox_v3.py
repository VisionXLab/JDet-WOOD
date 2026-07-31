"""Point2RBoxV3 detector（Jittor 移植）。

源：Point2RBox-v3 的 `point2rbox_v3.py`。
模板：本目录 `point2rbox_v2.py`；模块级纯函数位于
`point2rbox_v3_functions.py`。

targets 约定：
    targets[i]['rboxes'] : jt.float32 (N, 5) xywhr
    targets[i]['labels'] : jt.int     (N,)
    targets[i]['bids']   : jt.int32   (N, 4) = (batch, syn, view, instance)
    targets[i]['ss']     : tuple (aug_type, aug_val)

上游怪癖（照抄，禁止"修正"）：
1. copy-paste cache 由 ``zip(dual_stream_inputs, results_list)`` 生成（2B 条），
   而应用条件是 ``len(targets) == len(self.copy_paste_cache)``（原批 B 条）——
   恒不成立，即 **v3 上游的 copy-paste step2 实际是死代码**（v2 是活的）。
   逐字对应保留：cache 建 2B 条、条件比 B，行为与上游一致。
2. ``images_no_copypaste`` 快照在 copy-paste 之前深拷贝；TED 边缘与
   voronoi/SAM 用快照，head.images 用（可能被贴图修改过的）dual 流。
3. ``label_assign_pseudo_label_switch_eopch`` 拼写照抄。
"""
import copy
import math

import numpy as np
import jittor as jt
from jittor import nn

from jdet.utils.registry import MODELS, BACKBONES, HEADS, NECKS, build_from_cfg
from jdet.ops.linalg2x2 import diag_embed_2x2

from .point2rbox_v3_functions import (get_copy_paste_cache,
                                      voronoi_diagram_watershed)

from third_parties.ted.ted import TED


def _aa_bilinear_weights(in_size, out_size):
    """torch upsample_bilinear2d_aa 的权重矩阵（align_corners=False）。

    torchvision 0.17 的 resized_crop(tensor) 实际语义 = 越界裁剪区**补零**
    + bilinear **antialias=True** 缩放（已实验逐位确认）。jt.nn.interpolate
    无 antialias，此处以 (out, in) 稀疏权重矩阵精确复刻可分离三角滤波。
    """
    scale = in_size / out_size
    support = scale if scale >= 1.0 else 1.0
    W = np.zeros((out_size, in_size), dtype=np.float32)
    for i in range(out_size):
        center = scale * (i + 0.5)
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size)
        js = np.arange(xmin, xmax)
        w = 1.0 - np.abs((js + 0.5 - center) / scale)
        w = np.clip(w, 0.0, None)
        s = w.sum()
        if s > 0:
            W[i, xmin:xmax] = w / s
    return W


def _resized_crop_aa(images, ch, cw, out_h, out_w):
    """torchvision resized_crop(img, 0, 0, ch, cw, [out_h, out_w]) 等价实现。

    裁剪区 (ch, cw) 超出图像的部分补零，再做 antialias 双线性缩放。
    """
    B, C, H, W = images.shape
    padded = images
    if ch > H or cw > W:
        canvas = jt.zeros((B, C, max(ch, H), max(cw, W)),
                          dtype=images.dtype)
        canvas[:, :, :H, :W] = images
        padded = canvas
    padded = padded[:, :, :ch, :cw]
    Wh = jt.array(_aa_bilinear_weights(ch, out_h))          # (out_h, ch)
    Ww = jt.array(_aa_bilinear_weights(cw, out_w))          # (out_w, cw)
    x = padded.reshape(B * C, ch, cw)
    # GPU cublas_batched_matmul 不广播 batch 维（CPU 会广播，测试测不出）——显式 expand
    Wh_b = Wh.unsqueeze(0).expand((B * C, out_h, ch))
    Ww_b = Ww.transpose(1, 0).unsqueeze(0).expand((B * C, cw, out_w))
    x = jt.matmul(Wh_b, x)                                   # (B*C, out_h, cw)
    x = jt.matmul(x, Ww_b)                                   # (B*C, out_h, out_w)
    return x.reshape(B, C, out_h, out_w)


def _empty_results():
    return dict(bboxes=jt.zeros((0, 5), dtype='float32'),
                scores=jt.zeros((0,), dtype='float32'),
                labels=jt.zeros((0,), dtype='int32'))


@MODELS.register_module()
class Point2RBoxV3(nn.Module):
    """Implementation of Point2RBox-v3 (weakly supervised, point → rbox)."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 rotate_range=(0.25, 0.75),
                 scale_range=(0.5, 0.9),
                 ss_prob=[0.6, 0.15, 0.25],
                 copy_paste_start_epoch=6,
                 label_assign_pseudo_label_switch_eopch=6,
                 num_copies=10,
                 debug=False,
                 data_preprocessor=None,
                 train_cfg=None,
                 test_cfg=None):
        super(Point2RBoxV3, self).__init__()
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
        self.label_assign_pseudo_label_switch_eopch = \
            label_assign_pseudo_label_switch_eopch
        self.num_copies = num_copies
        self.debug = debug
        self.copy_paste_cache = None
        self.epoch = 0

        if data_preprocessor is not None:
            mean = np.array(data_preprocessor['mean'],
                            dtype=np.float32).reshape(3, 1, 1)
            std = np.array(data_preprocessor['std'],
                           dtype=np.float32).reshape(3, 1, 1)
        else:
            mean = np.array([123.675, 116.28, 103.53],
                            dtype=np.float32).reshape(3, 1, 1)
            std = np.array([58.395, 57.12, 57.375],
                           dtype=np.float32).reshape(3, 1, 1)
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

        Jittor's base ``Module.train`` toggles descendants by DFS and does not
        dispatch to an overridden child ``train`` method. Re-enter backbone
        train mode explicitly so its frozen-stage and BN policy is restored.
        """
        super(Point2RBoxV3, self).train()
        self.backbone.train()
        self.ted_model.eval()
        return self

    def state_dict(self, to=None, recurse=True):
        """Exclude per-batch image caches from checkpoints.

        Jittor treats every ``Var`` assigned to a Module as persistent.  The
        head stores current images only so EdgeLoss can consume them, so saving
        these tensors bloats checkpoints and makes a fresh model (where both
        attributes are ``None``) impossible to resume with ``load_parameters``.
        """
        state = super().state_dict(to=to, recurse=recurse)
        state.pop('bbox_head.images', None)
        state.pop('bbox_head.images_no_copypaste', None)
        state.pop('bbox_head.edges', None)
        return state

    def load_parameters(self, params):
        # Backward compatibility for ckpt_7..11 produced before the caches
        # were marked non-persistent.
        params = dict(params)
        params.pop('bbox_head.images', None)
        params.pop('bbox_head.images_no_copypaste', None)
        params.pop('bbox_head.edges', None)
        return super().load_parameters(params)

    # ------------------------------------------------------------------ #
    # view augmentation（照 v2 模板；ref rotate_crop L172-239）
    # ------------------------------------------------------------------ #

    def rotate_crop(self, batch_inputs, rot=0., size=(768, 768),
                    targets=None, padding='reflection'):
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
            batch_inputs = nn.grid_sample(batch_inputs, grid, 'bilinear',
                                          padding, align_corners=True)
            if targets is not None:
                for target in targets:
                    gt_bboxes = target['rboxes']
                    xy, wh, a = (gt_bboxes[..., :2], gt_bboxes[..., 2:4],
                                 gt_bboxes[..., 4:5])
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
            xy, wh, a = (gt_bboxes[..., :2], gt_bboxes[..., 2:4],
                         gt_bboxes[..., 4:5])
            xy = xy - jt.array(np.float32([[crop_w, crop_h]]))
            target['rboxes'] = jt.concat([xy, wh, a], dim=-1)
        return batch_inputs, targets

    def vflip(self, img):
        return img[:, :, ::-1, :]

    def prepare_dual_stream(self, images, targets, rng=None):
        """ref prepare_dual_stream_inputs（L240-302），照 v2 模板写法。

        rng: 可选 dict {'sel_p': float, 'aug_val': float}，测试注入用。
        """
        H, W = images.shape[2:4]
        sel_p = float(jt.rand(1).item()) if rng is None else rng['sel_p']
        if sel_p < self.ss_prob[0]:
            rval = float(jt.rand(1).item()) if rng is None else rng['aug_val']
            rot = math.pi * (
                rval * (self.rotate_range[1] - self.rotate_range[0])
                + self.rotate_range[0])
            ss = ('rot', rot)
            targets_aug = copy.deepcopy(targets)
            images_aug, targets_aug = self.rotate_crop(
                images, rot, [H, W], targets_aug, 'reflection')
            for target in targets_aug:
                target['bids'][:, 0] += len(targets)
                target['bids'][:, 2] = 1
        elif sel_p < self.ss_prob[0] + self.ss_prob[1]:
            ss = ('flp', 0)
            images_aug = self.vflip(images)
            targets_aug = copy.deepcopy(targets)
            for target in targets_aug:
                b = target['rboxes']
                target['rboxes'] = jt.concat(
                    [b[:, 0:1], H - b[:, 1:2], b[:, 2:4], -b[:, 4:5]], dim=-1)
                target['bids'][:, 0] += len(targets)
                target['bids'][:, 2] = 1
        else:
            sval = float(jt.rand(1).item()) if rng is None else rng['aug_val']
            sca = (sval * (self.scale_range[1] - self.scale_range[0])
                   + self.scale_range[0])
            ss = ('sca', sca)
            # ref: torchvision resized_crop(0, 0, H/sca, W/sca -> H, W)
            # = 越界补零 + antialias bilinear（见 _resized_crop_aa 注释）
            ch, cw = int(H / sca), int(W / sca)
            images_aug = _resized_crop_aa(images, ch, cw, H, W)
            targets_aug = copy.deepcopy(targets)
            for target in targets_aug:
                b = target['rboxes']
                target['rboxes'] = jt.concat(
                    [b[:, :4] * sca, b[:, 4:5]], dim=-1)
                target['bids'][:, 0] += len(targets)
                target['bids'][:, 2] = 1

        images_all = jt.concat([images, images_aug], 0)
        targets_all = []
        for target in targets + targets_aug:
            t = dict(target)
            t['ss'] = ss
            targets_all.append(t)
        return images_all, targets_all

    # ------------------------------------------------------------------ #
    # v3 specifics
    # ------------------------------------------------------------------ #

    def prepare_edges(self, images_no_copypaste):
        """ref L304-311：TED 在 no-copypaste 快照上出边缘图。"""
        with jt.no_grad():
            batch_edges = self.ted_model(
                images_no_copypaste * self.preprocess_std
                + self.preprocess_mean)
            self.bbox_head.edges = batch_edges[3].clamp(0)

    def prepare_copy_paste_step2(self, images_all, targets_all):
        """ref L316-346：把上一轮 cache 的 pattern 贴进 aug 半批。

        上游此分支因 cache 长度恒为 2B 而永不触发（见文件头怪癖 1），
        逻辑仍照抄以保持字面对应。
        """
        B = images_all.shape[0]
        aug_begin_id = B // 2
        aug_samples_len = B // 2
        for i in range(aug_samples_len):
            target = targets_all[aug_begin_id + i]
            patterns = self.copy_paste_cache[i]
            if not patterns:
                continue
            bboxes_paste = []
            labels_paste = []
            for p, b, l in patterns:
                ph, pw = p.shape[1:3]
                ox = np.random.randint(0, images_all.shape[-1] - pw)
                oy = np.random.randint(0, images_all.shape[-2] - ph)
                region = images_all[aug_begin_id + i, :, oy:oy + ph,
                                    ox:ox + pw]
                images_all[aug_begin_id + i, :, oy:oy + ph, ox:ox + pw] = \
                    region * (1 - p[3:4]) + p[:3] * p[3:4]
                bboxes_paste.append(b + np.float32((ox, oy, 0, 0, 0)))
                labels_paste.append(l)
            target['rboxes'] = jt.concat(
                [target['rboxes'], jt.array(np.float32(bboxes_paste))], 0)
            target['labels'] = jt.concat(
                [target['labels'],
                 jt.array(np.int32(labels_paste)).cast(
                     target['labels'].dtype)], 0)
            bids_paste = jt.array(np.int32([i, 1, 0, 0])).expand(
                (len(labels_paste), 4))
            target['bids'] = jt.concat([target['bids'], bids_paste], 0)

    def generate_pseudo_targets(self, targets_all, results_list_assist=None):
        """ref L646-722：PLA 阶段用 voronoi+watershed 从点标注产伪 rbox。"""
        results_list = []
        for i, target in enumerate(targets_all):
            bboxes = target['rboxes']
            mask = target['bids'][:, 1] == 0
            mask_np = mask.numpy()
            mu = bboxes[:, :2][mask]
            J = int(mask_np.sum())
            if J == 0:
                results_list.append(_empty_results())
                continue
            label = target['labels'][mask]
            image = self.bbox_head.images_no_copypaste[i]

            if self.bbox_head.voronoi_type in ('gaussian-orientation',
                                               'gaussian-full'):
                result_assist = results_list_assist[i]
                rbox_preds = result_assist['bboxes']  # (J, 5)
                assert rbox_preds.shape[0] == J and rbox_preds.shape[1] == 5
                cos_r = jt.cos(rbox_preds[:, -1])
                sin_r = jt.sin(rbox_preds[:, -1])
                R = jt.stack((cos_r, -sin_r, sin_r, cos_r),
                             dim=-1).reshape(-1, 2, 2)
                sigma = jt.matmul(
                    jt.matmul(R, diag_embed_2x2(rbox_preds[:, 2:4] / 2.0)),
                    R.permute(0, 2, 1)).view(-1, 2, 2)
            else:
                sigma = None

            vt = self.bbox_head.voronoi_type
            vthres = self.bbox_head.voronoi_thres
            ncls = self.bbox_head.num_classes
            if self.bbox_head.loss_voronoi.use_class_specific_watershed:
                pseudo_info = np.ones((J, 5), dtype=np.float32)
                label_np = label.numpy()
                for cur_class_id in range(ncls):
                    cur_class_mask = label_np == cur_class_id
                    if not cur_class_mask.any():
                        continue
                    cur_mu = mu[jt.array(cur_class_mask)]
                    cur_label = label[jt.array(cur_class_mask)]
                    cur_sigma = (sigma[jt.array(cur_class_mask)]
                                 if sigma is not None else None)
                    cur_info = voronoi_diagram_watershed(
                        cur_mu, cur_sigma, cur_label, image,
                        voronoi_type=vt, voronoi_thres=vthres,
                        num_classes=ncls)
                    pseudo_info[cur_class_mask] = np.float32(
                        cur_info).reshape(-1, 5)
            else:
                pseudo_info = np.float32(voronoi_diagram_watershed(
                    mu, sigma, label, image,
                    voronoi_type=vt, voronoi_thres=vthres,
                    num_classes=ncls)).reshape(-1, 5)

            results_list.append(dict(
                bboxes=jt.array(pseudo_info),
                scores=jt.ones((J,), dtype='float32'),
                labels=label))
        return results_list

    # ------------------------------------------------------------------ #
    # train / test entry
    # ------------------------------------------------------------------ #

    def forward_train(self, images, targets):
        # Set bids: (N, 4) = (batch, syn, view, instance)，obj id 从 1 起
        offset = 1
        for i, target in enumerate(targets):
            blen = target['rboxes'].shape[0]
            bids = jt.zeros((blen, 4), dtype='int32')
            bids[:, 0] = i
            bids[:, 3] = jt.arange(0, blen, 1) + offset
            target['bids'] = bids
            offset += blen

        images_all, targets_all = self.prepare_dual_stream(images, targets)

        # snapshot BEFORE copy-paste（ref L380）；voronoi/SAM 与 TED 都用它
        images_no_copypaste = images_all.clone().detach()
        self.bbox_head.images_no_copypaste = images_no_copypaste

        if self.epoch >= self.bbox_head.edge_loss_start_epoch:
            self.prepare_edges(images_no_copypaste)

        # ref L386：条件比较原批长度 vs cache 长度（2B），上游死代码，照抄
        if self.copy_paste_cache and \
                len(targets) == len(self.copy_paste_cache):
            self.prepare_copy_paste_step2(images_all, targets_all)

        self.bbox_head.images = images_all

        feat = self.backbone(images_all)
        if self.neck:
            feat = self.neck(feat)

        # Step1: pseudo labels（ref L399-411）
        if self.epoch >= self.label_assign_pseudo_label_switch_eopch:
            results_list = self.bbox_head.predict(feat, targets_all)
        else:
            if self.bbox_head.voronoi_type == 'standard':
                results_list = self.generate_pseudo_targets(targets_all)
            elif self.bbox_head.voronoi_type in ('gaussian-orientation',
                                                 'gaussian-full'):
                results_list_assist = self.bbox_head.predict(feat, targets_all)
                results_list = self.generate_pseudo_targets(
                    targets_all, results_list_assist)

        # Step2: 用伪 rbox 回填 bids[:,1]==0 的实例（ref L414-418）
        for target, results in zip(targets_all, results_list):
            mask = target['bids'][:, 1] == 0
            mask_np = mask.numpy()
            if mask_np.sum() == 0:
                continue
            rb = target['rboxes'].numpy()
            rb[mask_np] = results['bboxes'].detach().numpy()
            target['rboxes'] = jt.array(rb)
            lb = target['labels'].numpy()
            lb[mask_np] = results['labels'].numpy().astype(lb.dtype)
            target['labels'] = jt.array(lb)

        self._last_targets_all = targets_all  # test introspection only
        outs = self.bbox_head.execute(feat)
        losses = self.bbox_head.loss(*outs, targets_all)

        # copy-paste cache 重建（ref L421-428：遍历 dual 流全部 2B 项，照抄）
        if self.epoch >= self.copy_paste_start_epoch:
            self.copy_paste_cache = []
            for img, results in zip(images_all, results_list):
                self.copy_paste_cache.append(get_copy_paste_cache(
                    img, results['bboxes'].detach(), results['labels'],
                    self.bbox_head.square_cls, self.num_copies))

        return losses

    def forward_test(self, images, targets):
        feat = self.backbone(images)
        if self.neck:
            feat = self.neck(feat)
        results = self.bbox_head.predict(feat, targets)
        # JDet evaluate contract: one (polys, scores, 0-based labels) tuple
        # per image. The evaluator performs its own label-base conversion.
        # pseudo_generator 模式保留 dict（伪标签导出器消费）。
        if self.bbox_head.pseudo_generator:
            return results
        from jdet.models.boxes.box_ops import rotated_box_to_poly
        out = []
        for r in results:
            if r['bboxes'].shape[0] == 0:
                out.append((jt.zeros((0, 8)), jt.zeros((0,)),
                            jt.zeros((0,), dtype='int32')))
                continue
            out.append((rotated_box_to_poly(r['bboxes']),
                        r['scores'], r['labels']))
        return out

    def execute(self, images, targets):
        # val/伪标签生成的数据也可能携带 GT rboxes；仅凭字段判断会在
        # model.eval() 下误入 forward_train（正式 12ep run 曾在最终 val 命中）。
        # 与 Point2RBoxV2/JDet 其他 detector 一致，训练态和字段必须同时满足。
        if self.is_training() and 'rboxes' in targets[0]:
            return self.forward_train(images, targets)
        return self.forward_test(images, targets)
