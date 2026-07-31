"""RotatedFCOSHead（mmrotate rotated_fcos_head.py 的 Jittor 移植，stage-2 用）。

对照官方 rotated-fcos-1x-dota-using-pseudo.py（PLAN §6.3）：
    strides=[8,16,32,64,128], center_sampling=True, center_sample_radius=1.5,
    norm_on_bbox=True, centerness_on_reg=True, use_hbbox_loss=False,
    scale_angle=True, bbox_coder=DistanceAnglePointCoder(le90),
    loss: FocalLoss / RotatedIoULoss / centerness CE(sigmoid), loss_angle=None
脚手架/约定复用 point2rbox_v2_head（0-based 标签、MMDetFocalLoss、mmdet 式 predict）。
"""
import copy

import numpy as np
import jittor as jt
from jittor import nn

from jdet.utils.registry import HEADS, LOSSES, BOXES, build_from_cfg
from jdet.utils.general import multi_apply
from jdet.models.utils.weight_init import normal_init, bias_init_with_prob
from jdet.models.utils.modules import ConvModule
from jdet.ops.nms_rotated import nms_rotated

INF = 1e8


class Scale(nn.Module):

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = jt.array(np.float32(scale))

    def execute(self, x):
        return x * self.scale


@BOXES.register_module()
class PseudoAngleCoder:
    """恒等角度编码（mmrotate PseudoAngleCoder）。"""

    encode_size = 1

    def encode(self, angle_targets):
        return angle_targets

    def decode(self, angle_preds, keepdim=False):
        return angle_preds if keepdim else angle_preds.squeeze(-1)


def regularize_boxes_le90_np(boxes):
    """mmrotate RotatedBoxes.regularize_boxes('le90')：w>=h、角度 ∈ [-π/2, π/2)。"""
    b = boxes.copy()
    w, h, t = b[:, 2].copy(), b[:, 3].copy(), b[:, 4].copy()
    swap = w < h
    b[:, 2] = np.where(swap, h, w)
    b[:, 3] = np.where(swap, w, h)
    t = np.where(swap, t + np.pi / 2, t)
    b[:, 4] = ((t + np.pi / 2) % np.pi) - np.pi / 2
    return b


@HEADS.register_module()
class RotatedFCOSHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 angle_version='le90',
                 use_hbbox_loss=False,
                 scale_angle=True,
                 angle_coder=dict(type='PseudoAngleCoder'),
                 bbox_coder=dict(type='DistanceAnglePointCoder'),
                 loss_cls=dict(
                     type='MMDetFocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='MMDetCrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_angle=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 conv_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super(RotatedFCOSHead, self).__init__()
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.angle_version = angle_version
        self.use_hbbox_loss = use_hbbox_loss
        self.is_scale_angle = scale_angle
        if norm_cfg is not None:
            norm_cfg = {k: v for k, v in norm_cfg.items() if k != 'requires_grad'}
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.angle_coder = build_from_cfg(angle_coder, BOXES)
        self.bbox_coder = build_from_cfg(bbox_coder, BOXES)
        self.loss_cls = build_from_cfg(loss_cls, LOSSES)
        self.loss_bbox = build_from_cfg(loss_bbox, LOSSES)
        self.loss_centerness = build_from_cfg(loss_centerness, LOSSES)
        self.loss_angle = build_from_cfg(loss_angle, LOSSES) \
            if loss_angle is not None else None

        self._init_layers()
        self.init_weights()

    # ------------------------------------------------------------------ layers
    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            for convs in (self.cls_convs, self.reg_convs):
                convs.append(
                    ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                               conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
                               bias=self.norm_cfg is None))
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels,
                                  3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.conv_angle = nn.Conv2d(
            self.feat_channels, self.angle_coder.encode_size, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        if self.is_scale_angle:
            self.scale_angle = Scale(1.0)

    def init_weights(self):
        for m in list(self.cls_convs) + list(self.reg_convs):
            if hasattr(m, 'conv'):
                normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01)
        normal_init(self.conv_centerness, std=0.01)
        normal_init(self.conv_angle, std=0.01)

    # ----------------------------------------------------------------- forward
    def forward_single(self, x, scale, stride):
        cls_feat = x
        reg_feat = x
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = bbox_pred.clamp(0)
            if not self.is_training():
                bbox_pred = bbox_pred * stride
        else:
            bbox_pred = bbox_pred.exp()
        angle_pred = self.conv_angle(reg_feat)
        if self.is_scale_angle:
            angle_pred = self.scale_angle(angle_pred).float()
        return cls_score, bbox_pred, angle_pred, centerness

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)

    def execute(self, feats, targets=None):
        # JDet SingleStageDetector 约定：head(feat, targets) 自行分发
        if targets is not None and self.is_training() and 'rboxes' in targets[0]:
            return self.loss(feats, targets)
        if targets is not None:
            return self.get_bboxes(feats, targets)
        return self.forward(feats)

    def get_points(self, featmap_sizes):
        mlvl_points = []
        for i, (h, w) in enumerate(featmap_sizes):
            stride = self.strides[i]
            x_range = (jt.arange(w).float() + 0.5) * stride
            y_range = (jt.arange(h).float() + 0.5) * stride
            y = y_range[:, None].expand((h, w))
            x = x_range[None, :].expand((h, w))
            mlvl_points.append(
                jt.stack([x.reshape(-1), y.reshape(-1)], -1))
        return mlvl_points

    # ------------------------------------------------------------------- loss
    def loss(self, x, targets):
        outs = self.forward(x)
        return self.loss_by_feat(*outs, targets)

    def loss_by_feat(self, cls_scores, bbox_preds, angle_preds, centernesses,
                     targets):
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds) \
            == len(centernesses)
        featmap_sizes = [featmap.shape[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes)
        labels, bbox_targets, angle_targets = self.get_targets(
            all_level_points, targets)

        num_imgs = cls_scores[0].shape[0]
        flatten_cls_scores = jt.concat([
            s.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for s in cls_scores])
        flatten_bbox_preds = jt.concat([
            b.permute(0, 2, 3, 1).reshape(-1, 4) for b in bbox_preds])
        angle_dim = self.angle_coder.encode_size
        flatten_angle_preds = jt.concat([
            a.permute(0, 2, 3, 1).reshape(-1, angle_dim) for a in angle_preds])
        flatten_centerness = jt.concat([
            c.permute(0, 2, 3, 1).reshape(-1) for c in centernesses])
        flatten_labels = jt.concat(labels)
        flatten_bbox_targets = jt.concat(bbox_targets)
        flatten_angle_targets = jt.concat(angle_targets)
        flatten_points = jt.concat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        bg_class_ind = self.num_classes
        pos_mask = (flatten_labels >= 0) & (flatten_labels < bg_class_ind)
        pos_inds = jt.nonzero(pos_mask).reshape(-1)
        num_pos = max(float(pos_inds.shape[0]), 1.0)  # reduce_mean：单卡恒等
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        centerness_denorm = max(
            float(pos_centerness_targets.sum().detach().item()), 1e-6)

        if pos_inds.shape[0] > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_angle_preds = self.angle_coder.decode(
                pos_angle_preds, keepdim=True)
            pos_bbox_preds5 = jt.concat(
                [pos_bbox_preds, pos_decoded_angle_preds], dim=-1)
            pos_bbox_targets5 = jt.concat(
                [pos_bbox_targets, pos_angle_targets], dim=-1)
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds5)
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets5)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)

    @staticmethod
    def centerness_target(pos_bbox_targets):
        """FCOS centerness：sqrt((min lr/max lr)·(min tb/max tb))。"""
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if pos_bbox_targets.shape[0] == 0:
            return jt.zeros((0,))
        centerness_targets = (
            left_right.min(-1) / left_right.max(-1).clamp(1e-12)) * (
            top_bottom.min(-1) / top_bottom.max(-1).clamp(1e-12))
        return jt.sqrt(centerness_targets)

    # ----------------------------------------------------------------- targets
    def get_targets(self, points, targets):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        expanded_regress_ranges = [
            jt.array(np.float32(self.regress_ranges[i]))[None].expand(
                (points[i].shape[0], 2)) for i in range(num_levels)
        ]
        concat_regress_ranges = jt.concat(expanded_regress_ranges, dim=0)
        concat_points = jt.concat(points, dim=0)
        num_points = [center.shape[0] for center in points]

        labels_list, bbox_targets_list, angle_targets_list = multi_apply(
            self._get_targets_single,
            targets,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        splits = np.cumsum(num_points)[:-1].tolist()

        def split_lvl(t):
            out, prev = [], 0
            for s in splits + [t.shape[0]]:
                out.append(t[prev:s])
                prev = s
            return out

        labels_list = [split_lvl(x) for x in labels_list]
        bbox_targets_list = [split_lvl(x) for x in bbox_targets_list]
        angle_targets_list = [split_lvl(x) for x in angle_targets_list]

        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_angle_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(jt.concat([x[i] for x in labels_list]))
            bbox_targets = jt.concat([x[i] for x in bbox_targets_list])
            angle_targets = jt.concat([x[i] for x in angle_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_angle_targets.append(angle_targets)
        return (concat_lvl_labels, concat_lvl_bbox_targets,
                concat_lvl_angle_targets)

    def _get_targets_single(self, target, points, regress_ranges,
                            num_points_per_lvl):
        num_points = points.shape[0]
        gt_labels = target['labels']
        gt_bboxes_np = target['rboxes'].detach().numpy() \
            if isinstance(target['rboxes'], jt.Var) else target['rboxes']
        num_gts = gt_bboxes_np.shape[0]

        if num_gts == 0:
            return jt.full((num_points,), self.num_classes, dtype=gt_labels.dtype), \
                jt.zeros((num_points, 4)), \
                jt.zeros((num_points, 1))

        # areas 在 regularize 之前计算（w*h 不变，与上游一致）
        areas_np = gt_bboxes_np[:, 2] * gt_bboxes_np[:, 3]
        gt_bboxes = jt.array(regularize_boxes_le90_np(gt_bboxes_np))

        areas = jt.array(areas_np)[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            (num_points, num_gts, 2))
        points_e = points[:, None, :].expand((num_points, num_gts, 2))
        gt_bboxes_e = gt_bboxes[None].expand((num_points, num_gts, 5))
        gt_ctr = gt_bboxes_e[..., :2]
        gt_wh = gt_bboxes_e[..., 2:4]
        gt_angle = gt_bboxes_e[..., 4:5]

        cos_a = jt.cos(gt_angle)
        sin_a = jt.sin(gt_angle)
        rot_matrix = jt.concat([cos_a, sin_a, -sin_a, cos_a], dim=-1) \
            .reshape(num_points, num_gts, 2, 2)
        offset = points_e - gt_ctr
        offset = jt.matmul(rot_matrix, offset[..., None]).squeeze(-1)

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = jt.stack((left, top, right, bottom), -1)

        inside_gt_bbox_mask = bbox_targets.min(-1) > 0
        if self.center_sampling:
            radius = self.center_sample_radius
            stride = jt.zeros_like(offset)
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end
            inside_center_bbox_mask = (jt.abs(offset) < stride).all(-1)
            inside_gt_bbox_mask = jt.logical_and(inside_center_bbox_mask,
                                                 inside_gt_bbox_mask)

        max_regress_distance = bbox_targets.max(-1)
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        areas = jt.where(inside_gt_bbox_mask, areas, jt.full_like(areas, INF))
        areas = jt.where(inside_regress_range, areas, jt.full_like(areas, INF))
        min_area_inds, min_area = jt.argmin(areas, dim=1)

        labels = gt_labels[min_area_inds]
        labels = jt.where(min_area == INF,
                          jt.full_like(labels, self.num_classes), labels)
        rr = jt.arange(num_points)
        bbox_targets = bbox_targets[rr, min_area_inds]
        angle_targets = gt_angle[rr, min_area_inds]

        return labels, bbox_targets, angle_targets

    # ----------------------------------------------------------------- predict
    def get_bboxes(self, x, targets):
        from jdet.models.boxes.box_ops import rotated_box_to_poly
        outs = self.forward(x)
        results = []
        featmap_sizes = [s.shape[-2:] for s in outs[0]]
        mlvl_priors = self.get_points(featmap_sizes)
        for img_id, target in enumerate(targets):
            r = self._predict_single(
                [s[img_id].detach() for s in outs[0]],
                [b[img_id].detach() for b in outs[1]],
                [a[img_id].detach() for a in outs[2]],
                [c[img_id].detach() for c in outs[3]],
                mlvl_priors, target)
            if r['bboxes'].shape[0] == 0:
                results.append((jt.zeros((0, 8)), jt.zeros((0,)),
                                jt.zeros((0,), dtype='int32')))
                continue
            results.append((rotated_box_to_poly(r['bboxes']),
                            r['scores'], r['labels']))
        return results

    def _predict_single(self, cls_score_list, bbox_pred_list, angle_pred_list,
                        centerness_list, mlvl_priors, target):
        cfg = copy.deepcopy(self.test_cfg) if self.test_cfg else {}
        nms_pre = cfg.get('nms_pre', -1)
        score_thr = cfg.get('score_thr', 0)

        mlvl_bboxes, mlvl_scores, mlvl_labels, mlvl_factors = [], [], [], []
        for cls_score, bbox_pred, angle_pred, centerness, priors in zip(
                cls_score_list, bbox_pred_list, angle_pred_list,
                centerness_list, mlvl_priors):
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(
                -1, self.angle_coder.encode_size)
            cf = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()

            flat_scores = scores.reshape(-1)
            valid_idx = jt.nonzero(flat_scores > score_thr).reshape(-1)
            valid_scores = flat_scores[valid_idx]
            if nms_pre > 0 and valid_scores.shape[0] > nms_pre:
                order, _ = jt.argsort(valid_scores, descending=True)
                valid_idx = valid_idx[order[:nms_pre]]
                valid_scores = valid_scores[order[:nms_pre]]
            keep_idxs = (valid_idx // self.cls_out_channels).int32()
            labels = (valid_idx % self.cls_out_channels).int32()

            bbox_pred = bbox_pred[keep_idxs]
            angle_pred = angle_pred[keep_idxs]
            priors_sel = priors[keep_idxs]
            cf_sel = cf[keep_idxs]
            decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
            pred5 = jt.concat([bbox_pred, decoded_angle], dim=-1)
            bboxes = self.bbox_coder.decode(priors_sel, pred5)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(valid_scores)
            mlvl_labels.append(labels)
            mlvl_factors.append(cf_sel)

        scores = jt.concat(mlvl_scores)
        labels = jt.concat(mlvl_labels)
        bboxes = jt.concat(mlvl_bboxes)
        factors = jt.concat(mlvl_factors)
        # mmdet：score_factors（centerness）乘进最终分数再 NMS
        scores = scores * factors

        min_bbox_size = cfg.get('min_bbox_size', 0)
        if min_bbox_size >= 0 and bboxes.shape[0] > 0:
            m = (bboxes[:, 2] > min_bbox_size) & (bboxes[:, 3] > min_bbox_size)
            keep = jt.nonzero(m).reshape(-1)
            bboxes, scores, labels = bboxes[keep], scores[keep], labels[keep]

        if bboxes.shape[0] > 0:
            nms_cfg = cfg.get('nms', dict(type='nms_rotated', iou_threshold=0.1))
            iou_thr = nms_cfg.get('iou_threshold', 0.1)
            order, _ = jt.argsort(scores, descending=True)
            bboxes, scores, labels = bboxes[order], scores[order], labels[order]
            max_coord = float(bboxes.max().item()) + 1
            offsets = labels.float32() * max_coord
            boxes_for_nms = jt.concat(
                [bboxes[:, :2] + offsets[:, None], bboxes[:, 2:]], -1)
            keep = nms_rotated(boxes_for_nms, scores, iou_thr)
            keep, _ = jt.sort(keep)
            bboxes, scores, labels = bboxes[keep], scores[keep], labels[keep]
            max_per_img = cfg.get('max_per_img', -1)
            if max_per_img > 0 and bboxes.shape[0] > max_per_img:
                bboxes = bboxes[:max_per_img]
                scores = scores[:max_per_img]
                labels = labels[:max_per_img]

        return dict(bboxes=bboxes, scores=scores, labels=labels)
