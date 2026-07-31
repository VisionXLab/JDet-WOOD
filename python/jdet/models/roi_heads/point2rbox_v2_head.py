"""Point2RBoxV2Head（Jittor 移植）。

源：/root/ref/Point2RBox-v3/mmrotate/models/dense_heads/point2rbox_v2_head.py
脚手架风格对齐 h2rbox_v2p_head.py（JDet 无 AnchorFreeHead 基类，自建 convs）。

关键翻译（详见 docs/porting_notes.md）：
- index_reduce_('amin')：unique 分组内该值恒同（bid/batch/label 由 instance 唯一决定），
  等价于「组内任取代表」→ 用 scatter 赋值实现（无梯度量）
- index_reduce_('mean')：预测量带梯度 → one-hot 矩阵乘求组均值（梯度安全）
- torch.unique(return_inverse/counts)：作用于 detach 的 bid，走 numpy
- 全部 masked in-place 赋值 → jt.where out-of-place
- reduce_mean：单卡训练为恒等（官方 config 总 batch=2 单卡，见 PLAN §9.1）
"""
import copy

import numpy as np
import jittor as jt
from jittor import nn

from jdet.utils.registry import HEADS, LOSSES, BOXES, build_from_cfg
from jdet.utils.general import multi_apply
from jdet.models.utils.weight_init import normal_init, bias_init_with_prob
from jdet.models.utils.modules import ConvModule
from jdet.models.boxes.coder import PSCCoder  # noqa: F401（registry 注册）
from jdet.ops.linalg2x2 import diag_embed_2x2
from jdet.ops.nms_rotated import nms_rotated

INF = 1e8


def _group_mean(values, idx, num_groups):
    """按 idx 分组求均值（等价 index_reduce_('mean', include_self=False)），梯度安全。

    values: (N, C) 或 (N,)；idx: (N,) int numpy/var；返回 (G, C) 或 (G,)。
    """
    squeeze = values.ndim == 1
    if squeeze:
        values = values.unsqueeze(-1)
    idx_np = idx if isinstance(idx, np.ndarray) else idx.numpy()
    # scatter-add 段求和 O(N·C)。原 one-hot 矩阵乘是 O(G·N·C)——num_pos 大的
    # 批次（数万正样本 × 数千组）单次分配数百 MB、耗时秒级，是慢批次主因
    N, C = values.shape
    idx_jt = jt.array(idx_np.astype(np.int32))
    out = jt.zeros((num_groups, C), dtype=values.dtype).scatter(
        0, idx_jt.unsqueeze(-1).expand((N, C)), values, reduce='add')
    count_np = np.bincount(idx_np, minlength=num_groups).astype(np.float32)
    count = jt.array(np.maximum(count_np, 1.0)).unsqueeze(-1)
    out = out / count
    return out[:, 0] if squeeze else out


def _group_any(values_np, idx_np, num_groups):
    """分组内任取代表（值恒同时与 amin 等价）。numpy 实现，无梯度。"""
    out = np.zeros((num_groups,) + values_np.shape[1:], dtype=values_np.dtype)
    out[idx_np] = values_np
    return out


@HEADS.register_module()
class Point2RBoxV2Head(nn.Module):
    """Point2RBox-v2 head（单层特征 strides=[8]，铁律二 #6）。"""

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=[8],
                 regress_ranges=[(-1, 1e8)],
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
                 loss_cls=dict(
                     type='MMDetFocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0),
                 loss_overlap=dict(type='GaussianOverlapLoss', loss_weight=10.0),
                 loss_voronoi=dict(type='VoronoiWatershedLoss', loss_weight=5.0),
                 loss_bbox_edg=dict(type='EdgeLoss', loss_weight=0.3),
                 loss_ss=dict(type='Point2RBoxV2ConsistencyLoss', loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 conv_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super(Point2RBoxV2Head, self).__init__()
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
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
        # jittor GroupNorm 无 requires_grad 参数（参数默认可训练），剥离该键
        if norm_cfg is not None:
            norm_cfg = {k: v for k, v in norm_cfg.items() if k != 'requires_grad'}
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.epoch = 0
        self.images = None
        self.edges = None
        self.vis = None

        self.angle_coder = build_from_cfg(angle_coder, BOXES)
        self.bbox_coder = build_from_cfg(bbox_coder, BOXES)
        self.loss_cls = build_from_cfg(loss_cls, LOSSES)
        self.loss_bbox = build_from_cfg(loss_bbox, LOSSES)
        self.loss_ss = build_from_cfg(loss_ss, LOSSES)
        self.loss_overlap = build_from_cfg(loss_overlap, LOSSES)
        self.loss_voronoi = build_from_cfg(loss_voronoi, LOSSES)
        self.loss_bbox_edg = build_from_cfg(loss_bbox_edg, LOSSES)

        self._init_layers()
        self.init_weights()

    # ------------------------------------------------------------------ layers
    def _init_layers(self):
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()
        self.conv_angle = nn.Conv2d(
            self.feat_channels, self.angle_coder.encode_size, 3, padding=1)
        self.conv_gate = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

    def _init_cls_convs(self):
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

    def _init_reg_convs(self):
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

    def _init_predictor(self):
        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            if hasattr(m, 'conv'):
                normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            if hasattr(m, 'conv'):
                normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01)
        normal_init(self.conv_angle, std=0.01)
        normal_init(self.conv_gate, std=0.01, bias=bias_cls)

    # ----------------------------------------------------------------- forward
    def forward(self, x):
        cls_feat = x[0]
        reg_feat = x[0]

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        angle_pred = self.conv_angle(reg_feat)

        # Gaussian sig_x, sig_y, p
        sig_x = bbox_pred[:, 0].exp()
        sig_y = bbox_pred[:, 1].exp()
        dx = bbox_pred[:, 2].sigmoid() * 2 - 1  # (-1, 1)
        dy = bbox_pred[:, 3].sigmoid() * 2 - 1  # (-1, 1)
        bbox_pred = jt.stack((sig_x, sig_y, dx, dy), 1) * 8

        return (cls_score,), (bbox_pred,), (angle_pred,)

    def execute(self, x):
        return self.forward(x)

    # ------------------------------------------------------------------ points
    def get_points(self, featmap_sizes, dtype='float32'):
        """mmdet MlvlPointGenerator.grid_priors（offset=0.5 → (i+0.5)*stride）。"""
        mlvl_points = []
        for i, (h, w) in enumerate(featmap_sizes):
            stride = self.strides[i]
            x_range = (jt.arange(w).float() + 0.5) * stride
            y_range = (jt.arange(h).float() + 0.5) * stride
            y = y_range[:, None].expand((h, w))
            x = x_range[None, :].expand((h, w))
            points = jt.stack([x.reshape(-1), y.reshape(-1)], -1)
            mlvl_points.append(points)
        return mlvl_points

    # ------------------------------------------------------------------- loss
    def loss(self, x, targets):
        outs = self.forward(x)
        return self.loss_by_feat(*outs, targets)

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
            angle_pred.permute(0, 2, 3, 1).reshape(-1, self.angle_coder.encode_size)
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
        num_pos = max(float(pos_inds.shape[0]), 1.0)  # reduce_mean：单卡恒等
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
            pos_cls_scores = jt.gather(pos_cls_scores, 1, pos_labels[:, None])[:, 0]

            pos_decoded_angle_preds = self.angle_coder.decode(
                pos_angle_preds, keepdim=True)
            if self.epoch < self.joint_angle_start_epoch:
                pos_decoded_angle_preds = pos_decoded_angle_preds.detach()
            square_mask = jt.zeros_like(pos_labels).bool()
            for c in self.square_cls:
                square_mask = jt.logical_or(square_mask, pos_labels == c)
            pos_decoded_angle_preds = jt.where(
                square_mask.unsqueeze(-1), jt.zeros_like(pos_decoded_angle_preds),
                pos_decoded_angle_preds)

            pos_rbox_targets = self.bbox_coder.decode(pos_points, pos_bbox_targets)
            pos_rbox_preds = jt.concat((pos_points + pos_bbox_preds[:, 2:],
                                        pos_bbox_preds[:, :2] * 2,
                                        pos_decoded_angle_preds), -1)

            cos_r = jt.cos(pos_decoded_angle_preds)
            sin_r = jt.sin(pos_decoded_angle_preds)
            R = jt.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
            pos_gaus_preds = jt.matmul(
                jt.matmul(R, diag_embed_2x2(pos_bbox_preds[:, :2])),
                R.permute(0, 2, 1))

            # Regress copy-paste objects and point-annotated centers
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

            # Aggregate targets of the same instance based on their identical bid
            bid_np = pos_bid_targets.detach().numpy().astype(np.float64)
            bid_with_view = bid_np[:, 3] + 0.5 * bid_np[:, 2]
            bid, idx = np.unique(bid_with_view, return_inverse=True)
            G = len(bid)

            # 组内 bid_with_view 恒同 → ins_bid_with_view == bid（amin 的равно价形式）
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
                    loss_bbox_ovl += self.loss_overlap(
                        (mu, jt.matmul(sigma, sigma)))
                if len(gidx) >= 1:
                    pos_thres = [self.voronoi_thres['default'][0]] * self.num_classes
                    neg_thres = [self.voronoi_thres['default'][1]] * self.num_classes
                    if 'override' in self.voronoi_thres.keys():
                        for item in self.voronoi_thres['override']:
                            for cls in item[0]:
                                pos_thres[cls] = item[1][0]
                                neg_thres[cls] = item[1][1]
                    loss_bbox_vor += self.loss_voronoi(
                        (mu, jt.matmul(sigma, sigma)),
                        label, self.images[batch_id],
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
                    label_np = ins_labels_np[gidx] if len(gidx) else np.zeros(0)
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

            bmsk_idx = jt.array(np.nonzero(bmsk_np)[0].astype(np.int32))
            pair_gaus_preds = ins_gaus_preds[bmsk_idx].view(-1, 2, 2, 2)
            pair_labels_np = ins_labels_np[bmsk_np].reshape(-1, 2)[:, 0]
            square_mask_np = np.zeros_like(pair_labels_np, dtype=bool)
            for c in self.square_cls:
                square_mask_np |= pair_labels_np == c

            pair_cls_scores = _group_mean(pos_cls_scores, idx, G)[bmsk_idx].view(-1, 2)
            pair_angle_preds = _group_mean(pos_angle_preds, idx, G)[bmsk_idx] \
                .view(-1, 2, pos_angle_preds.shape[-1])
            pair_angle_preds = self.angle_coder.decode(pair_angle_preds, keepdim=True)

            # Self-supervision
            ss_info = targets[0]['ss']
            valid = pair_cls_scores[:, 1] > 0.1
            bbox_area = pair_gaus_preds[:, 0, 0, 0] * pair_gaus_preds[:, 0, 1, 1] * 4
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

        labels_list, bbox_targets_list, bid_targets_list = multi_apply(
            self._get_targets_single,
            targets,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level（单层时即整段）
        splits = np.cumsum(num_points)[:-1].tolist()

        def split_lvl(t):
            if not splits:
                return [t]
            out, prev = [], 0
            for s in splits + [t.shape[0]]:
                out.append(t[prev:s])
                prev = s
            return out

        labels_list = [split_lvl(labels) for labels in labels_list]
        bbox_targets_list = [split_lvl(b) for b in bbox_targets_list]
        bid_targets_list = [split_lvl(b) for b in bid_targets_list]

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
        num_points = points.shape[0]
        gt_bboxes = target['rboxes']
        gt_labels = target['labels']
        gt_bids = target['bids']
        num_gts = gt_bboxes.shape[0]

        if num_gts == 0:
            return jt.full((num_points,), self.num_classes, dtype=gt_labels.dtype), \
                jt.zeros((num_points, 5)), \
                jt.zeros((num_points, 4), dtype=gt_bids.dtype)

        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            (num_points, num_gts, 2))
        points_e = points[:, None, :].expand((num_points, num_gts, 2))
        gt_bboxes_e = gt_bboxes[None].expand((num_points, num_gts, 5))
        gt_ctr = gt_bboxes_e[..., :2]
        gt_wh = gt_bboxes_e[..., 2:4]
        gt_angle = gt_bboxes_e[..., 4:5]

        offset = points_e - gt_ctr
        w, h = gt_wh[..., 0], gt_wh[..., 1]

        center_r = jt.clamp((w * h).sqrt() / 64, 1, 5)[..., None]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = jt.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            radius = self.center_sample_radius
            stride = jt.zeros_like(offset)
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end
            inside_gt_bbox_mask = (jt.abs(offset) < stride * center_r).all(-1)

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
        bid_targets = gt_bids[min_area_inds]
        bbox_targets = jt.concat((bbox_targets, angle_targets), -1)

        return labels, bbox_targets, bid_targets

    # ----------------------------------------------------------------- predict
    def predict(self, x, targets):
        """训练期（或 pseudo_generator）用 GT 点位取预测框。返回 list[dict]。"""
        outs = self.forward(x)
        return self.predict_by_feat(*outs, targets=targets)

    def predict_by_feat(self, cls_scores, bbox_preds, angle_preds, targets,
                        cfg=None, rescale=False, with_nms=True):
        result_list = []
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(len(cls_scores))]
        mlvl_priors = self.get_points(featmap_sizes)
        for img_id, target in enumerate(targets):
            cls_score_list = [s[img_id].detach() for s in cls_scores]
            bbox_pred_list = [b[img_id].detach() for b in bbox_preds]
            angle_pred_list = [a[img_id].detach() for a in angle_preds]
            if self.is_training() or self.pseudo_generator:
                results = self._predict_by_feat_single_pseudo(
                    cls_score_list, bbox_pred_list, angle_pred_list,
                    mlvl_priors, target, cfg, rescale, with_nms)
            else:
                results = self._predict_by_feat_single(
                    cls_score_list, bbox_pred_list, angle_pred_list,
                    mlvl_priors, target, cfg, rescale, with_nms)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single_pseudo(self, cls_score_list, bbox_pred_list,
                                       angle_pred_list, mlvl_priors, target,
                                       cfg, rescale=False, with_nms=True):
        if self.is_training():
            scale_factor = [1, 1]
        else:
            scale_factor = target.get('scale_factor', [1, 1])
            if np.isscalar(scale_factor):
                scale_factor = [scale_factor, scale_factor]
        gt_bboxes = target['rboxes']
        gt_labels = target['labels']
        gt_pos = (gt_bboxes[:, 0:2] / self.strides[0] * scale_factor[1]).int32()

        cls_score, bbox_pred, angle_pred = \
            cls_score_list[0], bbox_pred_list[0], angle_pred_list[0]
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
        bboxes = jt.concat((gt_bboxes[:, 0:2], bbox_pred[:, :2] * 2,
                            decoded_angle), -1)

        inval = jt.logical_not(gt_valid_mask).unsqueeze(-1)
        bboxes = jt.concat([
            bboxes[:, :2],
            jt.where(inval.expand((inval.shape[0], 3)),
                     jt.zeros_like(bboxes[:, 2:]), bboxes[:, 2:])], -1)
        bboxes = jt.concat([bboxes[:, :2], bboxes[:, 2:4] / scale_factor[1],
                            bboxes[:, 4:5]], -1)

        for id in self.post_process.keys():
            m = (gt_labels == id).unsqueeze(-1)
            bboxes = jt.concat([
                bboxes[:, :2],
                jt.where(m.expand((m.shape[0], 2)),
                         bboxes[:, 2:4] * self.post_process[id], bboxes[:, 2:4]),
                bboxes[:, 4:5]], -1)
        for id in self.square_cls:
            m = (gt_labels == id).unsqueeze(-1)
            bboxes = jt.concat([
                bboxes[:, :4],
                jt.where(m, jt.zeros_like(bboxes[:, 4:5]), bboxes[:, 4:5])], -1)

        return dict(bboxes=bboxes.detach(),
                    scores=jt.ones_like(cls_score[:, 0]),
                    labels=gt_labels)

    def _predict_by_feat_single(self, cls_score_list, bbox_pred_list,
                                angle_pred_list, mlvl_priors, target,
                                cfg, rescale=False, with_nms=True):
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg) if cfg is not None else {}
        nms_pre = cfg.get('nms_pre', -1)
        score_thr = cfg.get('score_thr', 0)

        mlvl_bbox_preds = []
        mlvl_scores = []
        mlvl_labels = []
        for level_idx, (cls_score, bbox_pred, angle_pred, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              angle_pred_list, mlvl_priors)):
            assert cls_score.shape[-2:] == bbox_pred.shape[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(
                -1, self.angle_coder.encode_size)
            cls_score = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels)
            scores = cls_score.sigmoid()

            # filter_scores_and_topk（mmdet 语义）：多类打平 → 过 score_thr → topk
            flat_scores = scores.reshape(-1)
            valid_mask = flat_scores > score_thr
            valid_idx = jt.nonzero(valid_mask).reshape(-1)
            valid_scores = flat_scores[valid_idx]
            if nms_pre > 0 and valid_scores.shape[0] > nms_pre:
                order, _ = jt.argsort(valid_scores, descending=True)
                topk_idx = order[:nms_pre]
                valid_idx = valid_idx[topk_idx]
                valid_scores = valid_scores[topk_idx]
            keep_idxs = (valid_idx // self.cls_out_channels).int32()
            labels = (valid_idx % self.cls_out_channels).int32()

            bbox_pred = bbox_pred[keep_idxs]
            angle_pred = angle_pred[keep_idxs]
            priors_sel = priors[keep_idxs]

            decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
            bbox_pred = jt.concat((priors_sel + bbox_pred[:, 2:],
                                   bbox_pred[:, :2] * 2, decoded_angle), -1)

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_scores.append(valid_scores)
            mlvl_labels.append(labels)

        scores = jt.concat(mlvl_scores)
        labels = jt.concat(mlvl_labels)
        bboxes = jt.concat(mlvl_bbox_preds)

        for id in self.post_process.keys():
            m = (labels == id).unsqueeze(-1)
            bboxes = jt.concat([
                bboxes[:, :2],
                jt.where(m.expand((m.shape[0], 2)),
                         bboxes[:, 2:4] * self.post_process[id], bboxes[:, 2:4]),
                bboxes[:, 4:5]], -1)
        for id in self.square_cls:
            m = (labels == id).unsqueeze(-1)
            bboxes = jt.concat([
                bboxes[:, :4],
                jt.where(m, jt.zeros_like(bboxes[:, 4:5]), bboxes[:, 4:5])], -1)

        # _bbox_post_process：min_bbox_size 过滤 → 逐类 nms_rotated → max_per_img
        min_bbox_size = cfg.get('min_bbox_size', 0)
        if min_bbox_size >= 0 and bboxes.shape[0] > 0:
            m = (bboxes[:, 2] > min_bbox_size) & (bboxes[:, 3] > min_bbox_size)
            keep = jt.nonzero(m).reshape(-1)
            bboxes, scores, labels = bboxes[keep], scores[keep], labels[keep]

        if with_nms and bboxes.shape[0] > 0:
            nms_cfg = cfg.get('nms', dict(type='nms_rotated', iou_threshold=0.1))
            iou_thr = nms_cfg.get('iou_threshold', 0.1)
            # 按分数降序（mmcv nms 的 keep 顺序，jdet nms 返回原始顺序 → 先排序）
            order, _ = jt.argsort(scores, descending=True)
            bboxes, scores, labels = bboxes[order], scores[order], labels[order]
            # 逐类偏移（batched_nms class-agnostic=False 语义）
            if bboxes.shape[0] > 0:
                max_coord = float(bboxes.max().item()) + 1
                offsets = labels.float32() * max_coord
                boxes_for_nms = jt.concat(
                    [bboxes[:, :2] + offsets[:, None], bboxes[:, 2:]], -1)
                keep = nms_rotated(boxes_for_nms, scores, iou_thr)
                keep, _ = jt.sort(keep)  # jdet 返回已是升序=分数降序
                bboxes, scores, labels = bboxes[keep], scores[keep], labels[keep]
            max_per_img = cfg.get('max_per_img', -1)
            if max_per_img > 0 and bboxes.shape[0] > max_per_img:
                bboxes = bboxes[:max_per_img]
                scores = scores[:max_per_img]
                labels = labels[:max_per_img]

        return dict(bboxes=bboxes, scores=scores, labels=labels)

    # ------------------------------------------------------------ test bridge
    def get_bboxes(self, x, targets):
        """测试路径：返回 JDet 约定的 (polys, scores, labels) 元组列表。"""
        from jdet.models.boxes.box_ops import rotated_box_to_poly
        outs = self.forward(x)
        results = self.predict_by_feat(*outs, targets=targets)
        out = []
        for r in results:
            if r['bboxes'].shape[0] == 0:
                out.append((jt.zeros((0, 8)), jt.zeros((0,)), jt.zeros((0,), dtype='int32')))
                continue
            polys = rotated_box_to_poly(r['bboxes'])
            out.append((polys, r['scores'], r['labels']))
        return out
