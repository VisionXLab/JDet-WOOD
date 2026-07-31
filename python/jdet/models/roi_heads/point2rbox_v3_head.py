# Ported from the PyTorch/mmrotate Point2RBox-v3 dense head. The structure
# follows JDet's h2rbox_v2p_head.py conventions.
#
# targets convention:
#   targets[i] is a dict with keys
#     'rboxes' : jt.float32 (N, 5) xywhr
#     'labels' : jt.int    (N,)
#     'bids'   : jt.int32  (N, 4) = (batch, syn, view, instance),
#                identical semantics to mmdet-side InstanceData.bids
#     'ss'     : python tuple (aug_type, aug_val)
#   dual-stream: original batch first, augmented second; aug bids[:,0] += B;
#   copy-paste instances carry bids=(i,1,0,0).
#
# Ref quirks copied verbatim (do NOT "fix"):
# - conv_gate is created and initialised (bias_prob=0.01) but never used in
#   single_level_forward; kept for weight/state parity.
# - use_adaptive_scale is popped from kwargs before anything else (ref does
#   `kwargs["use_adaptive_scale"]` + del) — here a plain keyword argument.
# - _predict_by_feat_single_pseudo divides BOTH xy and wh by scale_factor[1]
#   (index 1 only), ref L871.
# - selest_pseudo takes labels from level 0 (identical across levels) and
#   returns scores of ones, ref L786-787. Note the ref method name typo
#   ("selest") is kept so cross-references with upstream stay greppable.

import copy
import numpy as np
import jittor as jt
from jittor import nn

from jdet.utils.general import multi_apply
from jdet.utils.registry import HEADS, LOSSES, BOXES, build_from_cfg
from jdet.models.utils.weight_init import normal_init, bias_init_with_prob
from jdet.models.utils.modules import ConvModule
from jdet.ops.nms_rotated import nms_rotated
from jdet.ops.linalg2x2 import diag_embed_2x2
# Reuse the shared bid-grouping helpers from the v2 head.
from jdet.models.roi_heads.point2rbox_v2_head import _group_mean, _group_any

INF = 1e8


# DistanceAnglePointCoder: A merged a shared, registry-registered port into
# jdet/models/boxes/coder.py (v2 head uses it too). The self-contained copy
# that lived here was removed to avoid the duplicate-registration crash;
# stage-1 get_targets/predict parity re-verified against the shared coder.


def select_single_mlvl(mlvl_tensors, batch_id, detach=True):
    if detach:
        return [t[batch_id].detach() for t in mlvl_tensors]
    return [t[batch_id] for t in mlvl_tensors]


def filter_scores_and_topk(scores, score_thr, topk, results=None):
    """Jittor port of mmdet.models.utils.filter_scores_and_topk.

    scores: (num_points, num_classes). Returns (scores, labels, keep_idxs,
    filtered_results) matching mmdet semantics: threshold -> flatten valid
    (point, class) pairs -> topk by score.
    """
    valid_mask = scores > score_thr
    scores_flat = scores[valid_mask]
    valid_idxs = jt.nonzero(valid_mask)  # (M, 2): [point_idx, class_idx]

    num_topk = min(topk, scores_flat.shape[0]) if topk >= 0 \
        else scores_flat.shape[0]
    # mmdet: scores.sort(descending=True) then take first num_topk
    idxs = jt.argsort(scores_flat, descending=True)[0]
    idxs = idxs[:num_topk]
    scores_out = scores_flat[idxs]
    topk_idxs = valid_idxs[idxs]
    keep_idxs, labels = topk_idxs[:, 0], topk_idxs[:, 1]

    filtered_results = None
    if results is not None:
        filtered_results = {k: v[keep_idxs] for k, v in results.items()}
    return scores_out, labels, keep_idxs, filtered_results


@HEADS.register_module()
class Point2RBoxV3Head(nn.Module):
    """Point2RBox-v3 head (Jittor). See file header for scope and conventions."""

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=[8, 16, 32, 64, 128],
                 regress_ranges=[(-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)],
                 center_sampling=True,
                 center_sample_radius=0.75,
                 angle_version='le90',
                 edge_loss_start_epoch=6,
                 joint_angle_start_epoch=1,
                 pseudo_generator=False,
                 voronoi_type='gaussian-orientation',
                 voronoi_thres=dict(default=[0.994, 0.005]),
                 square_cls=[],
                 edge_loss_cls=[],
                 post_process={},
                 bbox_coder=dict(type='DistanceAnglePointCoder'),
                 angle_coder=dict(
                     type='PSCCoder',
                     angle_version='le90',
                     dual_freq=False,
                     num_step=3,
                     thr_mod=0),
                 # MMDetFocalLoss = mmdet.FocalLoss 语义（0-based，bg=15）。
                 # 底座旧 FocalLoss 是 1-based，禁止使用（A commit f45109b）。
                 loss_cls=dict(
                     type='MMDetFocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0),
                 loss_overlap=dict(
                     type='GaussianOverlapLoss', loss_weight=10.0),
                 loss_voronoi=dict(
                     type='VoronoiWatershedLoss', loss_weight=5.0),
                 loss_bbox_edg=dict(type='EdgeLoss', loss_weight=0.3),
                 loss_ss=dict(
                     type='Point2RBoxV2ConsistencyLoss', loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, is_train=True),
                 use_adaptive_scale=False,
                 conv_bias='auto',
                 conv_cfg=None,
                 test_cfg=None):
        super(Point2RBoxV3Head, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.angle_version = angle_version
        self.edge_loss_start_epoch = edge_loss_start_epoch
        self.joint_angle_start_epoch = joint_angle_start_epoch
        self.pseudo_generator = pseudo_generator
        self.voronoi_type = voronoi_type
        self.voronoi_thres = voronoi_thres
        self.square_cls = square_cls
        self.edge_loss_cls = edge_loss_cls
        self.post_process = post_process
        self.use_adaptive_scale = use_adaptive_scale
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.test_cfg = test_cfg

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.cls_out_channels = num_classes if self.use_sigmoid_cls \
            else num_classes + 1

        self.angle_coder = build_from_cfg(angle_coder, BOXES)
        self.bbox_coder = build_from_cfg(bbox_coder, BOXES)
        self.loss_cls = build_from_cfg(loss_cls, LOSSES)
        self.loss_bbox = build_from_cfg(loss_bbox, LOSSES)
        self.loss_overlap = build_from_cfg(loss_overlap, LOSSES)
        self.loss_voronoi = build_from_cfg(loss_voronoi, LOSSES)
        self.loss_bbox_edg = build_from_cfg(loss_bbox_edg, LOSSES)
        self.loss_ss = build_from_cfg(loss_ss, LOSSES)

        if self.use_adaptive_scale:
            self.adaptive_scale = jt.ones((num_classes, num_classes))

        self.epoch = 0
        self.images = None
        self.images_no_copypaste = None
        self.edges = None
        self.vis = None

        self._init_layers()

    # ------------------------------------------------------------------ #
    # layers / forward
    # ------------------------------------------------------------------ #

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                           conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
                           bias=self.conv_bias))
            self.reg_convs.append(
                ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                           conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
                           bias=self.conv_bias))
        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.conv_angle = nn.Conv2d(
            self.feat_channels, self.angle_coder.encode_size, 3, padding=1)
        # kept but unused in forward — parity with ref (see header)
        self.conv_gate = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.init_weights()

    def init_weights(self):
        # ref init_cfg: Normal std=0.01 on all Conv2d, with bias_prob=0.01
        # overrides on conv_cls and conv_gate
        for convs in (self.cls_convs, self.reg_convs):
            for m in convs:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01)
        normal_init(self.conv_angle, std=0.01)
        normal_init(self.conv_gate, std=0.01, bias=bias_cls)

    def execute(self, feats):
        return multi_apply(self.single_level_forward, feats, self.strides)

    def single_level_forward(self, feat, stride):
        cls_feat = feat
        reg_feat = feat
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        angle_pred = self.conv_angle(reg_feat)

        # Gaussian sig_x, sig_y, p — verbatim transform (ref L192-197)
        sig_x = bbox_pred[:, 0].exp()
        sig_y = bbox_pred[:, 1].exp()
        dx = bbox_pred[:, 2].sigmoid() * 2 - 1  # (-1, 1)
        dy = bbox_pred[:, 3].sigmoid() * 2 - 1  # (-1, 1)
        bbox_pred = jt.stack((sig_x, sig_y, dx, dy), dim=1) * stride

        return cls_score, bbox_pred, angle_pred

    # ------------------------------------------------------------------ #
    # priors
    # ------------------------------------------------------------------ #

    def get_points(self, featmap_sizes):
        """FCOS-style priors, identical to mmdet MlvlPointGenerator with
        offset=0.5 (all strides are even, so stride//2 == 0.5*stride)."""
        mlvl_points = []
        for i, (h, w) in enumerate(featmap_sizes):
            stride = self.strides[i]
            x_range = jt.arange(w).float()
            y_range = jt.arange(h).float()
            y, x = jt.meshgrid(y_range, x_range)
            points = jt.stack(
                (x.flatten() * stride, y.flatten() * stride),
                dim=-1) + stride // 2
            mlvl_points.append(points)
        return mlvl_points

    # ------------------------------------------------------------------ #
    # target assignment (ref L489-643)
    # ------------------------------------------------------------------ #

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

        labels_list, bbox_targets_list, bid_targets_list = multi_apply(
            self._get_targets_single,
            targets,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split per img -> per level, then concat over imgs per level
        splits = np.cumsum(num_points)[:-1].tolist()

        def split_lvls(t):
            return jt.split(t, num_points, dim=0)

        labels_list = [split_lvls(l) for l in labels_list]
        bbox_targets_list = [split_lvls(b) for b in bbox_targets_list]
        bid_targets_list = [split_lvls(b) for b in bid_targets_list]

        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_bid_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                jt.concat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                jt.concat([b[i] for b in bbox_targets_list]))
            concat_lvl_bid_targets.append(
                jt.concat([b[i] for b in bid_targets_list]))
        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_bid_targets

    def _get_targets_single(self, target, points, regress_ranges,
                            num_points_per_lvl):
        gt_bboxes = target['rboxes']          # (num_gts, 5)
        gt_labels = target['labels']
        gt_bids = target['bids']

        num_points = points.shape[0]
        num_gts = gt_bboxes.shape[0]
        if num_gts == 0:
            return (jt.full((num_points,), self.num_classes,
                            dtype=gt_labels.dtype),
                    jt.zeros((num_points, 4)),
                    jt.zeros((num_points, 4), dtype=gt_bids.dtype))

        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        # ref uses .repeat here (with a TODO wondering why); expand is
        # numerically identical for read-only use — kept as repeat for parity
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            (num_points, num_gts, 2))
        points_e = points[:, None, :].expand((num_points, num_gts, 2))
        gt_bboxes_e = gt_bboxes[None].expand((num_points, num_gts, 5))

        gt_ctr = gt_bboxes_e[..., 0:2]
        gt_wh = gt_bboxes_e[..., 2:4]
        gt_angle = gt_bboxes_e[..., 4:5]

        cos_angle, sin_angle = jt.cos(gt_angle), jt.sin(gt_angle)
        rot_matrix = jt.concat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points_e - gt_ctr
        offset = jt.matmul(rot_matrix, offset[..., None]).squeeze(-1)

        gt_w = gt_wh[..., 0]
        gt_h = gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = gt_w / 2 + offset_x
        right = gt_w / 2 - offset_x
        top = gt_h / 2 + offset_y
        bottom = gt_h / 2 - offset_y
        bbox_targets = jt.stack((left, top, right, bottom), dim=-1)

        inside_gt_bbox_mask = bbox_targets.min(dim=-1) > 0
        if self.center_sampling:
            radius = self.center_sample_radius
            stride_map = jt.zeros(offset.shape)
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride_map[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end
            inside_center_bbox_mask = (jt.abs(offset) < stride_map).all(dim=-1)
            inside_gt_bbox_mask = jt.logical_and(inside_center_bbox_mask,
                                                 inside_gt_bbox_mask)

        max_regress_distance = bbox_targets.max(dim=-1)
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # out-of-place INF masking (ref does in-place; targets are constants
        # so this is purely a safety idiom)
        areas = jt.where(inside_gt_bbox_mask, areas,
                         jt.full_like(areas, INF) if hasattr(jt, 'full_like')
                         else jt.ones_like(areas) * INF)
        areas = jt.where(inside_regress_range, areas, jt.ones_like(areas) * INF)
        min_area_inds, min_area = jt.argmin(areas, dim=1)

        labels = gt_labels[min_area_inds]
        labels = jt.where(min_area == INF,
                          jt.full((num_points,), self.num_classes,
                                  dtype=labels.dtype),
                          labels)
        rng = jt.arange(num_points)
        bbox_targets = bbox_targets[rng, min_area_inds]
        angle_targets = gt_angle[rng, min_area_inds]
        bid_targets = gt_bids[min_area_inds]
        bbox_targets = jt.concat((bbox_targets, angle_targets), dim=-1)
        return labels, bbox_targets, bid_targets

    # ------------------------------------------------------------------ #
    # loss (ref L201-488)
    # ------------------------------------------------------------------ #

    def loss(self, cls_scores, bbox_preds, angle_preds, targets):
        return self.loss_by_feat(cls_scores, bbox_preds, angle_preds, targets)

    def loss_by_feat(self, cls_scores, bbox_preds, angle_preds, targets):
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds)
        featmap_sizes = [featmap.shape[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes)
        labels, bbox_targets, bid_targets = self.get_targets(
            all_level_points, targets)

        num_imgs = cls_scores[0].shape[0]
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(
                -1, self.angle_coder.encode_size)
            for angle_pred in angle_preds
        ]
        flatten_cls_scores = jt.concat(flatten_cls_scores)
        flatten_bbox_preds = jt.concat(flatten_bbox_preds)
        flatten_angle_preds = jt.concat(flatten_angle_preds)
        flatten_labels = jt.concat(labels)
        flatten_bbox_targets = jt.concat(bbox_targets)
        flatten_bid_targets = jt.concat(bid_targets)
        flatten_points = jt.concat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        bg_class_ind = self.num_classes
        pos_mask = (flatten_labels >= 0) & (flatten_labels < bg_class_ind)
        pos_inds = jt.nonzero(pos_mask).reshape(-1)
        num_pos = max(float(pos_inds.shape[0]), 1.0)  # reduce_mean: 单卡恒等
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_bid_targets = flatten_bid_targets[pos_inds]

        self.vis = [None] * len(targets)
        if pos_inds.shape[0] > 0:
            pos_points = flatten_points[pos_inds]
            pos_labels = flatten_labels[pos_inds]
            pos_cls_scores = flatten_cls_scores[pos_inds].sigmoid()
            pos_cls_scores = jt.gather(
                pos_cls_scores, 1, pos_labels[:, None])[:, 0]

            pos_decoded_angle_preds = self.angle_coder.decode(
                pos_angle_preds, keepdim=True)
            if self.epoch < self.joint_angle_start_epoch:
                pos_decoded_angle_preds = pos_decoded_angle_preds.detach()
            square_mask = jt.zeros_like(pos_labels).bool()
            for c in self.square_cls:
                square_mask = jt.logical_or(square_mask, pos_labels == c)
            # ref L303 in-place `[square_mask] = 0` -> out-of-place (R5)
            pos_decoded_angle_preds = jt.where(
                square_mask.unsqueeze(-1),
                jt.zeros_like(pos_decoded_angle_preds),
                pos_decoded_angle_preds)

            pos_rbox_targets = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)
            pos_rbox_preds = jt.concat((pos_points + pos_bbox_preds[:, 2:],
                                        pos_bbox_preds[:, :2] * 2,
                                        pos_decoded_angle_preds), -1)

            cos_r = jt.cos(pos_decoded_angle_preds)
            sin_r = jt.sin(pos_decoded_angle_preds)
            R = jt.stack((cos_r, -sin_r, sin_r, cos_r),
                         dim=-1).reshape(-1, 2, 2)
            pos_gaus_preds = jt.matmul(
                jt.matmul(R, diag_embed_2x2(pos_bbox_preds[:, :2])),
                R.permute(0, 2, 1))

            # Regress copy-paste objects and point-annotated centers
            # ref L317 in-place masked assign -> out-of-place (R5)
            pos_syn_mask = pos_bid_targets[:, 1] == 1
            keep = pos_syn_mask.unsqueeze(-1)
            pos_rbox_targets = jt.concat([
                pos_rbox_targets[:, :2],
                jt.where(keep.expand((keep.shape[0], 3)),
                         pos_rbox_targets[:, 2:],
                         pos_rbox_preds[:, 2:].detach())], -1)
            loss_bbox = self.loss_bbox(
                pos_rbox_preds, pos_rbox_targets, avg_factor=num_pos)

            # Use gt point to replace predicted center for other losses
            pos_rbox_preds = jt.concat((pos_rbox_targets[:, :2],
                                        pos_bbox_preds[:, :2] * 2,
                                        pos_decoded_angle_preds), -1)

            # Aggregate targets of the same instance based on identical bid
            # (torch index_reduce_ -> np.unique + one-hot mean, v2 范式)
            bid_np = pos_bid_targets.detach().numpy().astype(np.float64)
            bid_with_view = bid_np[:, 3] + 0.5 * bid_np[:, 2]
            bid, idx = np.unique(bid_with_view, return_inverse=True)
            G = len(bid)

            # 组内 bid_with_view 恒同 → ins_bid_with_view == bid
            _, bidx, bcnt = np.unique(bid.astype(np.int64),
                                      return_inverse=True, return_counts=True)
            bmsk_np = bcnt[bidx] == 2

            ins_bids = _group_any(bid_np[:, 3], idx, G)
            ins_batch = _group_any(bid_np[:, 0], idx, G)
            ins_labels_np = _group_any(pos_labels.detach().numpy(), idx, G)
            ins_labels = jt.array(ins_labels_np)

            ins_gaus_preds = _group_mean(
                pos_gaus_preds.reshape(-1, 4), idx, G).reshape(-1, 2, 2)
            ins_rbox_preds = _group_mean(pos_rbox_preds, idx, G)
            ins_rbox_targets = _group_mean(pos_rbox_targets, idx, G)

            ori_mu_all = ins_rbox_targets[:, 0:2]
            loss_bbox_ovl = jt.zeros(1).sum()
            loss_bbox_vor = jt.zeros(1).sum()
            for batch_id in range(len(targets)):
                group_mask_np = (ins_batch == batch_id) & (ins_bids != 0)
                gidx = np.nonzero(group_mask_np)[0]
                if len(gidx) == 0:
                    continue
                gidx_jt = jt.array(gidx.astype(np.int32))
                mu = ori_mu_all[gidx_jt]
                sigma = ins_gaus_preds[gidx_jt]
                label = ins_labels[gidx_jt]
                if len(gidx) >= 2:
                    if self.use_adaptive_scale:
                        n = label.shape[0]
                        row_indices = label.unsqueeze(1).expand((n, n))
                        col_indices = label.unsqueeze(0).expand((n, n))
                        o_s = self.adaptive_scale[row_indices, col_indices]
                        loss_bbox_ovl += self.loss_overlap(
                            (mu, jt.matmul(sigma, sigma)), overlap_scale=o_s)
                    else:
                        loss_bbox_ovl += self.loss_overlap(
                            (mu, jt.matmul(sigma, sigma)))
                if len(gidx) >= 1:
                    pos_thres = [self.voronoi_thres['default'][0]] \
                        * self.num_classes
                    neg_thres = [self.voronoi_thres['default'][1]] \
                        * self.num_classes
                    if 'override' in self.voronoi_thres.keys():
                        for item in self.voronoi_thres['override']:
                            for cls in item[0]:
                                pos_thres[cls] = item[1][0]
                                neg_thres[cls] = item[1][1]
                    if self.loss_voronoi.use_class_specific_watershed:
                        cur_loss_bbox_vor = jt.zeros(1).sum()
                        for cur_class_id in range(self.num_classes):
                            cur_class_mask_np = \
                                ins_labels_np[gidx] == cur_class_id
                            cidx = np.nonzero(cur_class_mask_np)[0]
                            if len(cidx) == 0:
                                continue
                            cidx_jt = jt.array(cidx.astype(np.int32))
                            cur_mu = mu[cidx_jt]
                            cur_sigma = sigma[cidx_jt]
                            cur_label = label[cidx_jt]
                            cur_loss_bbox_vor += self.loss_voronoi(
                                (cur_mu, jt.matmul(cur_sigma, cur_sigma)),
                                cur_label,
                                self.images_no_copypaste[batch_id],
                                pos_thres, neg_thres,
                                voronoi=self.voronoi_type) * len(cidx)
                        loss_bbox_vor += cur_loss_bbox_vor / len(gidx)
                    else:
                        loss_bbox_vor += self.loss_voronoi(
                            (mu, jt.matmul(sigma, sigma)),
                            label, self.images_no_copypaste[batch_id],
                            pos_thres, neg_thres,
                            voronoi=self.voronoi_type)
                    self.vis[batch_id] = self.loss_voronoi.vis

            # Batched RBox for Edge Loss
            loss_bbox_edg = jt.zeros(1).sum()
            if self.epoch >= self.edge_loss_start_epoch:
                batched_rbox = []
                for batch_id in range(len(targets)):
                    group_mask_np = (ins_batch == batch_id) & (ins_bids != 0)
                    gidx = np.nonzero(group_mask_np)[0]
                    rbox = ins_rbox_preds[jt.array(gidx.astype(np.int32))] \
                        if len(gidx) else jt.zeros((0, 5))
                    label_np = ins_labels_np[gidx] if len(gidx) \
                        else np.zeros(0)
                    edge_mask = np.zeros(len(gidx), dtype=bool)
                    for c in self.edge_loss_cls:
                        edge_mask |= label_np == c
                    eidx = np.nonzero(edge_mask)[0]
                    batched_rbox.append(
                        rbox[jt.array(eidx.astype(np.int32))] if len(eidx)
                        else jt.zeros((0, 5)))
                loss_bbox_edg = self.loss_bbox_edg(batched_rbox, self.edges)

            loss_bbox_ovl = loss_bbox_ovl / len(targets)
            loss_bbox_vor = loss_bbox_vor / len(targets)
            loss_bbox_edg = loss_bbox_edg / len(targets)

            if not bmsk_np.any():
                # 空 pair（该 batch 无跨视图配对实例）：jittor 对 (0,·) 张量的
                # 切片/decode 不宽容（torch 可空跑），提前短路，保持梯度连通。
                loss_ss = 0 * pos_angle_preds.sum()
                return dict(
                    loss_cls=loss_cls,
                    loss_bbox=loss_bbox,
                    loss_bbox_vor=loss_bbox_vor,
                    loss_bbox_ovl=loss_bbox_ovl,
                    loss_bbox_edg=loss_bbox_edg,
                    loss_ss=loss_ss,
                )

            bmsk_idx = jt.array(np.nonzero(bmsk_np)[0].astype(np.int32))
            pair_gaus_preds = ins_gaus_preds[bmsk_idx].view(-1, 2, 2, 2)
            pair_labels_np = ins_labels_np[bmsk_np].reshape(-1, 2)[:, 0]
            square_mask_np = np.zeros_like(pair_labels_np, dtype=bool)
            for c in self.square_cls:
                square_mask_np |= pair_labels_np == c

            pair_cls_scores = _group_mean(
                pos_cls_scores, idx, G)[bmsk_idx].view(-1, 2)
            pair_angle_preds = _group_mean(
                pos_angle_preds, idx, G)[bmsk_idx].view(
                    -1, 2, pos_angle_preds.shape[-1])
            # Decode acts on the last dimension. Flattening to 2D avoids the
            # unsupported 3D boolean-mask path while preserving values.
            M2 = pair_angle_preds.shape[0] * 2
            pair_angle_preds = self.angle_coder.decode(
                pair_angle_preds.reshape(M2, -1), keepdim=True
            ).reshape(-1, 2, 1)

            # Self-supervision
            ss_info = targets[0]['ss']
            valid = pair_cls_scores[:, 1] > 0.1
            bbox_area = pair_gaus_preds[:, 0, 0, 0] \
                * pair_gaus_preds[:, 0, 1, 1] * 4
            sca = ss_info[1] if ss_info[0] == 'sca' else 1
            valid = jt.logical_and(valid, bbox_area > 24 ** 2)
            valid = jt.logical_and(valid, bbox_area * sca > 24 ** 2)
            valid = jt.logical_and(valid, bbox_area < 512 ** 2)
            valid = jt.logical_and(valid, bbox_area * sca < 512 ** 2)

            if bool(valid.any()):
                vidx = jt.nonzero(valid).reshape(-1)
                ori_gaus = pair_gaus_preds[vidx][:, 0]
                trs_gaus = pair_gaus_preds[vidx][:, 1]
                square_mask_v = jt.array(square_mask_np)[vidx]
                ori_angle = pair_angle_preds[vidx][:, 0]
                trs_angle = pair_angle_preds[vidx][:, 1]
                loss_ss = self.loss_ss(
                    (ori_gaus, ori_angle),
                    (trs_gaus, trs_angle),
                    square_mask_v,
                    *ss_info)
            else:
                loss_ss = 0 * pos_angle_preds.sum()
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_bbox_vor = pos_bbox_preds.sum()
            loss_bbox_ovl = pos_bbox_preds.sum()
            loss_bbox_edg = pos_bbox_preds.sum()
            loss_ss = pos_bbox_preds.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_bbox_vor=loss_bbox_vor,
            loss_bbox_ovl=loss_bbox_ovl,
            loss_bbox_edg=loss_bbox_edg,
            loss_ss=loss_ss,
        )

    # ------------------------------------------------------------------ #
    # predict paths (ref L645-1011)
    # ------------------------------------------------------------------ #

    def predict(self, feats, targets, rescale=False):
        outs = self.execute(feats)
        return self.predict_by_feat(*outs, targets=targets, rescale=rescale)

    def predict_by_feat(self, cls_scores, bbox_preds, angle_preds,
                        targets=None, cfg=None, rescale=False, with_nms=True):
        assert len(cls_scores) == len(bbox_preds)
        featmap_sizes = [t.shape[-2:] for t in cls_scores]
        mlvl_priors = self.get_points(featmap_sizes)

        result_list = []
        for img_id in range(len(targets)):
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            angle_pred_list = select_single_mlvl(angle_preds, img_id)

            if self.is_training() or self.pseudo_generator:
                fn = self._predict_by_feat_single_pseudo
            else:
                fn = self._predict_by_feat_single
            results = fn(cls_score_list, bbox_pred_list, angle_pred_list,
                         mlvl_priors, targets[img_id], cfg,
                         rescale=rescale, with_nms=with_nms)
            result_list.append(results)
        return result_list

    def selest_pseudo(self, results_list):
        """Select pseudo boxes across fpn levels by cls score (ref L752-789).
        results_list entries are dicts with 'bboxes'/'scores'/'labels'."""
        scores_list = [r['scores'].unsqueeze(1) for r in results_list]
        bboxes_list = [r['bboxes'] for r in results_list]
        labels_list = [r['labels'].unsqueeze(1) for r in results_list]

        scores_all = jt.concat(scores_list, dim=1).sigmoid()  # (N, L)
        max_ids, max_values = jt.argmax(scores_all, dim=1)

        bboxes_stacked = jt.stack(bboxes_list, dim=0)  # (L, N, 5)
        labels_stacked = jt.stack(labels_list, dim=0)  # (L, N, 1)
        L, N = bboxes_stacked.shape[0], bboxes_stacked.shape[1]
        row_indices = jt.arange(N)
        flat = max_ids * N + row_indices
        pseudo_bboxes_selected = bboxes_stacked.reshape(L * N, 5)[flat]
        labels_selected = labels_stacked.reshape(L * N, 1)[flat]  # noqa: F841
        # ref quirk: labels taken from level 0, scores set to ones
        return dict(bboxes=pseudo_bboxes_selected,
                    scores=jt.ones_like(scores_all[:, 0]),
                    labels=labels_list[0].squeeze(1))

    def _predict_by_feat_single_pseudo(self, cls_score_list, bbox_pred_list,
                                       angle_pred_list, mlvl_priors, target,
                                       cfg, rescale=False, with_nms=True):
        if self.is_training():
            scale_factor = [1, 1]
        else:
            # JDet 约定（实测 P2RV2DOTADataset）：scale_factor 是标量 float；
            # mmrotate 是 (w,h) 二元组。统一成 [w,h] 形态（ref 只用索引 1）。
            scale_factor = target.get('scale_factor', [1, 1])
            if not hasattr(scale_factor, '__len__'):
                scale_factor = [scale_factor, scale_factor]

        if self.is_training():
            mask = target['bids'][:, 1] == 0
            gt_bboxes = target['rboxes'][mask]
            gt_labels = target['labels'][mask]
        else:
            gt_bboxes = target['rboxes']
            gt_labels = target['labels']

        results_list = []
        for level_id in range(len(self.strides)):
            gt_pos = (gt_bboxes[:, 0:2] / self.strides[level_id]).long()
            cls_score = cls_score_list[level_id]
            bbox_pred = bbox_pred_list[level_id]
            angle_pred = angle_pred_list[level_id]
            H, W = cls_score.shape[1:3]
            gt_valid_mask = (gt_pos[:, 0] >= 0) & (gt_pos[:, 0] < W) & \
                            (gt_pos[:, 1] >= 0) & (gt_pos[:, 1] < H)
            gt_idx = gt_pos[:, 1] * W + gt_pos[:, 0]
            gt_idx = gt_idx.clamp(0, cls_score[0].numel() - 1)

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)[gt_idx]
            cls_score = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels)[gt_idx]
            angle_pred = angle_pred.permute(1, 2, 0).reshape(
                -1, self.angle_coder.encode_size)[gt_idx]

            decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
            bboxes = jt.concat(
                (gt_bboxes[:, 0:2], bbox_pred[:, :2] * 2, decoded_angle), -1)

            # out-of-place variants of ref's in-place edits (all detached)
            invalid = jt.logical_not(gt_valid_mask)[:, None]
            zero_wh_a = jt.concat(
                (bboxes[:, :2], jt.zeros_like(bboxes[:, 2:])), -1)
            bboxes = jt.where(invalid, zero_wh_a, bboxes)
            # ref L871: xy AND wh both divided by scale_factor[1] (quirk)
            bboxes = jt.concat(
                (bboxes[:, :4] / scale_factor[1], bboxes[:, 4:]), -1)

            # Jittor's ternary where does not broadcast an empty (0,1) mask
            # to (0,5).  Nothing remains to post-process in this case.
            if bboxes.shape[0] == 0:
                results_list.append(dict(
                    bboxes=bboxes.detach(),
                    scores=jt.zeros((0,), dtype='float32'),
                    labels=gt_labels))
                continue

            for cid in self.post_process.keys():
                m = (gt_labels == cid)[:, None]
                scaled = jt.concat(
                    (bboxes[:, :2], bboxes[:, 2:4] * self.post_process[cid],
                     bboxes[:, 4:]), -1)
                bboxes = jt.where(m, scaled, bboxes)
            for cid in self.square_cls:
                m = (gt_labels == cid)[:, None]
                zero_a = jt.concat(
                    (bboxes[:, :-1], jt.zeros_like(bboxes[:, -1:])), -1)
                bboxes = jt.where(m, zero_a, bboxes)

            row_indices = jt.arange(cls_score.shape[0])
            results_list.append(dict(
                bboxes=bboxes.detach(),
                scores=cls_score[row_indices, gt_labels].detach(),
                labels=gt_labels))
        return self.selest_pseudo(results_list)

    def _predict_by_feat_single(self, cls_score_list, bbox_pred_list,
                                angle_pred_list, mlvl_priors, target, cfg,
                                rescale=False, with_nms=True):
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg) if cfg is not None else {}
        nms_pre = cfg.get('nms_pre', -1)
        score_thr = cfg.get('score_thr', 0)

        mlvl_bbox_preds = []
        mlvl_scores = []
        mlvl_labels = []
        for cls_score, bbox_pred, angle_pred, priors in zip(
                cls_score_list, bbox_pred_list, angle_pred_list, mlvl_priors):
            assert cls_score.shape[-2:] == bbox_pred.shape[-2:]
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(
                -1, self.angle_coder.encode_size)
            cls_score = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = nn.softmax(cls_score, dim=-1)[:, :-1]

            scores, labels, keep_idxs, filtered = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, angle_pred=angle_pred,
                     priors=priors))
            bbox_pred = filtered['bbox_pred']
            angle_pred = filtered['angle_pred']
            priors = filtered['priors']

            decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
            bbox_pred = jt.concat(
                (priors + bbox_pred[:, 2:], bbox_pred[:, :2] * 2,
                 decoded_angle), -1)

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        scores = jt.concat(mlvl_scores)
        labels = jt.concat(mlvl_labels)
        bboxes = jt.concat(mlvl_bbox_preds)

        # A real DOTA test patch can have no score above threshold at every
        # FPN level.  Avoid empty (0,1) -> (0,5) broadcasting in jt.where.
        if bboxes.shape[0] == 0:
            return self._bbox_post_process(
                bboxes, scores, labels, cfg, target, rescale=rescale,
                with_nms=with_nms)

        for cid in self.post_process.keys():
            m = (labels == cid)[:, None]
            scaled = jt.concat(
                (bboxes[:, :2], bboxes[:, 2:4] * self.post_process[cid],
                 bboxes[:, 4:]), -1)
            bboxes = jt.where(m, scaled, bboxes)
        for cid in self.square_cls:
            m = (labels == cid)[:, None]
            zero_a = jt.concat(
                (bboxes[:, :-1], jt.zeros_like(bboxes[:, -1:])), -1)
            bboxes = jt.where(m, zero_a, bboxes)

        return self._bbox_post_process(bboxes, scores, labels, cfg,
                                       target, rescale=rescale,
                                       with_nms=with_nms)

    def _bbox_post_process(self, bboxes, scores, labels, cfg, target,
                           rescale=False, with_nms=True):
        """mmdet BaseDenseHead._bbox_post_process semantics with the
        class-offset batched-nms trick.

        ⚠ jdet nms_rotated keeps ORIGINAL order while mmcv sorts by score
        descending — we sort explicitly before max_per_img truncation
        (docs/porting_notes.md, A-core conclusion #2)."""
        if rescale:
            # JDet scale_factor 标量 / mmrotate 二元组 双兼容
            sf = target.get('scale_factor', [1, 1])
            if not hasattr(sf, '__len__'):
                sf = [sf, sf]
            bboxes = jt.concat(
                (bboxes[:, :2] / sf[0], bboxes[:, 2:4] / sf[0],
                 bboxes[:, 4:]), -1)

        min_bbox_size = cfg.get('min_bbox_size', -1)
        if min_bbox_size >= 0:
            valid = (bboxes[:, 2] > min_bbox_size) & \
                    (bboxes[:, 3] > min_bbox_size)
            if not bool(valid.all()):
                bboxes = bboxes[valid]
                scores = scores[valid]
                labels = labels[valid]

        if with_nms and bboxes.shape[0] > 0:
            nms_cfg = cfg.get('nms', dict(iou_threshold=0.1))
            iou_thr = nms_cfg.get('iou_threshold', 0.1)
            # class-offset trick (mmcv batched_nms): shift boxes per class so
            # cross-class overlaps never suppress each other
            max_coord = bboxes[:, :2].max() + bboxes[:, 2:4].max()
            offsets = labels.float() * (max_coord + 1)
            bboxes_for_nms = jt.concat(
                (bboxes[:, :2] + offsets[:, None], bboxes[:, 2:]), -1)
            keep = nms_rotated(bboxes_for_nms, scores, iou_thr)
            bboxes = bboxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            # explicit score-descending sort before truncation
            order = jt.argsort(scores, descending=True)[0]
            max_per_img = cfg.get('max_per_img', bboxes.shape[0])
            order = order[:max_per_img]
            bboxes = bboxes[order]
            scores = scores[order]
            labels = labels[order]

        return dict(bboxes=bboxes, scores=scores, labels=labels)
