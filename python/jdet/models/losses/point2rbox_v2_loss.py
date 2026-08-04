"""Point2RBox-v2 的四个 loss（Jittor 移植）。

源：Point2RBox-v3 的 `point2rbox_v2_loss.py`（v3 版，含对 v2 向后兼容
的扩展参数，默认值即 v2 行为）。

移植要点（详见 docs/porting_notes.md）：
- torch.linalg.eigh/solve → jdet.ops.linalg2x2 的 2×2 闭式解（可导、退化保护）
- loss[torch.eye(B)] = 0 等 in-place 写法 → out-of-place（乘 mask / jt.where），
  Jittor 的 in-place 自动微分不可靠（前向对、梯度 0）
- 上游源文件顶部 3 处未使用的 import（click/pandas/sympy）未移植
"""
import math

import cv2
import numpy as np
import jittor as jt
from jittor import nn

from jdet.utils.registry import LOSSES
from jdet.models.losses.gaussian_dist_loss import postprocess, weight_reduce_loss
from jdet.ops.linalg2x2 import eigh_2x2, solve_2x2, diag_embed_2x2


def gwd_sigma_loss(pred, target, weight=None, fun='log1p', tau=1.0, alpha=1.0,
                   normalize=True, reduction='mean', avg_factor=None):
    """GWD loss 的仅 sigma 版本（忽略 mu）。对齐上游 @weighted_loss 展开形式。"""
    Sigma_p = pred
    Sigma_t = target

    whr_distance = Sigma_p[..., 0, 0] + Sigma_p[..., 1, 1]
    whr_distance = whr_distance + Sigma_t[..., 0, 0] + Sigma_t[..., 1, 1]

    _t_tr = jt.matmul(Sigma_p, Sigma_t)
    _t_tr = _t_tr[..., 0, 0] + _t_tr[..., 1, 1]
    det_p = Sigma_p[..., 0, 0] * Sigma_p[..., 1, 1] - Sigma_p[..., 0, 1] * Sigma_p[..., 1, 0]
    det_t = Sigma_t[..., 0, 0] * Sigma_t[..., 1, 1] - Sigma_t[..., 0, 1] * Sigma_t[..., 1, 0]
    _t_det_sqrt = (det_p * det_t).clamp(1e-7).sqrt()
    whr_distance = whr_distance + (-2) * ((_t_tr + 2 * _t_det_sqrt).clamp(1e-7).sqrt())

    distance = (alpha * alpha * whr_distance).clamp(1e-7).sqrt()

    if normalize:
        scale = 2 * (_t_det_sqrt.clamp(1e-7).sqrt().clamp(1e-7).sqrt()).clamp(1e-7)
        distance = distance / scale

    loss = postprocess(distance, fun=fun, tau=tau)
    return weight_reduce_loss(loss, weight, reduction, avg_factor)


def bhattacharyya_coefficient(pred, target):
    """2-D 高斯分布间的 Bhattacharyya 系数，shape (N,)（batch 维保留）。"""
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    _shape = xy_p.shape

    xy_p = xy_p.reshape(-1, 2)
    xy_t = xy_t.reshape(-1, 2)
    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)

    Sigma_M = (Sigma_p + Sigma_t) / 2
    dxy = (xy_p - xy_t).unsqueeze(-1)
    t0 = jt.exp(-0.125 * jt.matmul(dxy.permute(0, 2, 1), solve_2x2(Sigma_M, dxy)))
    det_p = Sigma_p[:, 0, 0] * Sigma_p[:, 1, 1] - Sigma_p[:, 0, 1] * Sigma_p[:, 1, 0]
    det_t = Sigma_t[:, 0, 0] * Sigma_t[:, 1, 1] - Sigma_t[:, 0, 1] * Sigma_t[:, 1, 0]
    det_m = Sigma_M[:, 0, 0] * Sigma_M[:, 1, 1] - Sigma_M[:, 0, 1] * Sigma_M[:, 1, 0]
    t1 = (det_p * det_t).clamp(1e-7).sqrt()
    t2 = det_m

    coef = t0 * (t1 / t2).clamp(1e-7).sqrt()[..., None, None]
    coef = coef.reshape(_shape[:-1])
    return coef


def gaussian_overlap_loss(pred, target, weight=None, alpha=0.01, beta=0.6065,
                          overlap_scale=None, reduction='mean', avg_factor=None):
    """基于 Bhattacharyya 系数的高斯重叠 loss（@weighted_loss 展开形式）。"""
    mu, sigma = pred
    B = mu.shape[0]
    mu0 = mu[None].expand(B, B, 2)
    sigma0 = sigma[None].expand(B, B, 2, 2)
    mu1 = mu[:, None].expand(B, B, 2)
    sigma1 = sigma[:, None].expand(B, B, 2, 2)
    loss = bhattacharyya_coefficient((mu0, sigma0), (mu1, sigma1))
    if overlap_scale is not None:
        loss = jt.multiply(loss, overlap_scale) * overlap_scale.numel() / nn.relu(overlap_scale).sum()

    # 上游 loss[torch.eye(B, dtype=bool)] = 0 → out-of-place（对角元乘 0，梯度等价）
    loss = loss * (1 - jt.init.eye(B, dtype=loss.dtype))
    loss = nn.leaky_relu(loss - beta, scale=alpha) + beta * alpha
    loss = loss.sum(-1)
    return weight_reduce_loss(loss, weight, reduction, avg_factor)


@LOSSES.register_module()
class GaussianOverlapLoss(nn.Module):
    """Gaussian Overlap Loss（官方 config：loss_weight=10.0, lamb=0）。"""

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 lamb=1e-4):
        super(GaussianOverlapLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.lamb = lamb

    def execute(self,
                pred,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                overlap_scale=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        assert len(pred[0]) == len(pred[1])

        mu, sigma = pred

        L = eigh_2x2(sigma)[0].clamp(1e-7).sqrt()
        # F.l1_loss(L, zeros, reduction='none') == |L|
        loss_lamb = jt.abs(L)
        loss_lamb = self.lamb * jt.log(1 + loss_lamb).mean()

        overlap_loss = gaussian_overlap_loss(
            pred,
            None,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            overlap_scale=overlap_scale,
        )

        return self.loss_weight * (loss_lamb + overlap_loss)


def _smooth_l1(pred, target, beta, reduction='mean'):
    """torch.nn.functional.smooth_l1_loss 等价实现（带 beta）。"""
    diff = jt.abs(pred - target)
    loss = jt.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def gaussian_2d(xy, mu, sigma, normalize=False):
    dxy = (xy - mu).unsqueeze(-1)
    t0 = jt.exp(-0.5 * jt.matmul(dxy.permute(0, 2, 1), solve_2x2(sigma, dxy)))
    if normalize:
        det = sigma[..., 0, 0] * sigma[..., 1, 1] - sigma[..., 0, 1] * sigma[..., 1, 0]
        t0 = t0 / (2 * np.pi * det.clamp(1e-7).sqrt())
    return t0


def sigma_to_rbox_params(sigma):
    if not (tuple(sigma.shape) == (2, 2)):
        raise ValueError('输入必须是一个 (2, 2) 的张量')
    L, V = eigh_2x2(sigma)
    W_rotated = 2 * jt.sqrt(L[1])
    H_rotated = 2 * jt.sqrt(L[0])
    major_axis_vector = V[:, 1]
    angle_rad = jt.atan2(major_axis_vector[1], major_axis_vector[0])
    return W_rotated, H_rotated, angle_rad


def _get_box_prompt_from_gaussian(mu_j, sigma_j, sigma_scale=1, ellipse_scale_factor=1):
    W_base, H_base, angle_rad = sigma_to_rbox_params(sigma_j)

    scale_factor_from_sigma = math.sqrt(sigma_scale)
    final_scale_factor = scale_factor_from_sigma * ellipse_scale_factor

    semi_axis_a = (W_base / 2) * final_scale_factor
    semi_axis_b = (H_base / 2) * final_scale_factor

    cos_theta = jt.cos(angle_rad)
    sin_theta = jt.sin(angle_rad)

    half_width_bbox = jt.sqrt((semi_axis_a * cos_theta) ** 2 + (semi_axis_b * sin_theta) ** 2)
    half_height_bbox = jt.sqrt((semi_axis_a * sin_theta) ** 2 + (semi_axis_b * cos_theta) ** 2)

    mu_x, mu_y = mu_j[0], mu_j[1]
    bbox_prompt = jt.stack([mu_x - half_width_bbox, mu_y - half_height_bbox,
                            mu_x + half_width_bbox, mu_y + half_height_bbox],
                           dim=-1).detach().numpy()
    return bbox_prompt.reshape(1, 4)


def segment_anything(image, mu, sigma, device=None, sam_checkpoint=None, model_type=None,
                     label=None, debug=False, mask_filter_config=None, sam_sample_rules=None):
    """SAM 分支（v3 扩展；v2 默认 sam_instance_thr=-1 不触发）。

    predictor 依赖 `jdet.models.sam`，签名与 torch 版 MobileSAM 一致。
    """
    if debug:
        print('Entering SAM branch:')
    try:
        from jdet.models.sam import sam_model_registry, SamPredictor
    except ImportError:
        raise ImportError('jdet.models.sam is required for the SAM branch')
    from jdet.models.losses.point2rbox_v2_utils import filter_masks

    img_np = (image - image.min()) / (image.max() - image.min()) * 255.0
    img_np = img_np.permute(1, 2, 0).detach().numpy().astype(np.uint8)

    H, W = img_np.shape[:2]
    J = len(mu)

    if sam_checkpoint is None:
        import os
        for path in ['./mobile_sam.pt']:
            if os.path.exists(path):
                sam_checkpoint = path
                break
        if sam_checkpoint is None:
            raise ValueError('未找到MobileSAM检查点，请指定sam_checkpoint参数')

    if not hasattr(segment_anything, 'sam_model') or \
            not hasattr(segment_anything, 'model_type') or \
            segment_anything.model_type != model_type:
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        segment_anything.sam_model = sam
        segment_anything.model_type = model_type
    else:
        sam = segment_anything.sam_model

    predictor = SamPredictor(sam)
    predictor.set_image(img_np)

    points = mu.detach().numpy()

    markers = jt.full((H, W), J + 1, dtype='int32')

    total_loss = 0.0
    valid_instances = 0
    L, V = eigh_2x2(sigma)
    for j, point in enumerate(points):
        if debug:
            print(f'Processing point {j+1}/{J} at {point}')

        box_prompt = None
        all_points = [point]
        all_labels = [1]

        for k in range(J):
            if k != j:
                if sam_sample_rules is not None:
                    skip = False
                    j_label = int(label[j].item())
                    k_label = int(label[k].item())
                    dist = np.sqrt(((points[j] - points[k]) ** 2).sum())
                    for filter_pair in sam_sample_rules['filter_pairs']:
                        class_id1, class_id2, dist_thr = filter_pair
                        if ((j_label == class_id1 and k_label == class_id2) or
                                (j_label == class_id2 and k_label == class_id1)) \
                                and dist < dist_thr:
                            skip = True
                            break
                    if skip:
                        continue
                all_points.append(points[k])
                all_labels.append(0)

        masks, scores, _ = predictor.predict(
            point_coords=np.array(all_points),
            point_labels=np.array(all_labels),
            box=box_prompt,
            multimask_output=True)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        masks_processed = []
        for mask in masks:
            mask_opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            num_labels, labels_conn, stats, centroids = \
                cv2.connectedComponentsWithStats(mask_opened)
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                masks_processed.append(labels_conn == largest_label)
            else:
                masks_processed.append(mask_opened > 0)
        masks = masks_processed

        class_id = int(label[j].item())
        best_mask_idx, metrics_values, shape_metrics = filter_masks(
            image, masks, scores, class_id, img_np, point, mask_filter_config, debug)

        mask = masks[best_mask_idx]
        mask_np = np.asarray(mask, dtype=bool)
        # markers[mask_tensor] = j + 1 → out-of-place
        markers = jt.where(jt.array(mask_np), jt.full_like(markers, j + 1), markers)

        ys, xs = np.nonzero(mask_np)
        if len(xs) > 0:
            xy = jt.array(np.stack([xs, ys], 1).astype(np.float32))
            xy_centered = xy - mu[j]
            xy_rotated = jt.matmul(V[j].transpose(1, 0), xy_centered[:, :, None])[:, :, 0]
            max_x = jt.abs(xy_rotated[:, 0]).max()
            max_y = jt.abs(xy_rotated[:, 1]).max()
            L_target = jt.concat([max_x, max_y]) ** 2  # jt reduce 出 [1]，concat 得 (2,)
            L_diag = diag_embed_2x2(L[j])
            L_target_diag = diag_embed_2x2(L_target)
            instance_loss = gwd_sigma_loss(L_diag.unsqueeze(0),
                                           L_target_diag.unsqueeze(0).detach(),
                                           reduction='mean')
            total_loss = total_loss + instance_loss
            valid_instances += 1

    final_loss = total_loss / max(1, valid_instances)
    return final_loss, markers


def voronoi_watershed_loss(mu, sigma, label, image, pos_thres=0.994, neg_thres=0.005,
                           down_sample=2, topk=0.95, default_sigma=4096,
                           voronoi='gaussian-orientation', alpha=0.1, debug=False):
    J = len(sigma)
    if J == 0:
        return sigma.sum(), None
    D = down_sample
    H, W = image.shape[-2:]
    h, w = H // D, W // D
    x = jt.linspace(0, h, h)
    y = jt.linspace(0, w, w)
    # torch.meshgrid(x, y, indexing='xy') → 输出 shape (len(y), len(x))，
    # X[i,j]=x[j], Y[i,j]=y[i]（jittor 1.3.8.5 meshgrid 无 indexing 参数，手工构造）
    X = x[None, :].expand((w, h))
    Y = y[:, None].expand((w, h))
    xy = jt.stack([X, Y], -1)
    # Get distribution for each instance
    mm = (mu.detach() / D).round()

    # 逐实例 python 循环在 jittor 惰性模式下会积出 O(J) 巨图（密集图 J 可达数千，
    # 图融合优化超线性爆炸，单 batch 卡数十分钟）。该路径全为 detach 量 →
    # 改分块批量化 + 增量 argmax，公式与 gaussian_2d 完全一致（二次型展开）。
    if voronoi == 'standard':
        sg = jt.array(np.array([[default_sigma, 0], [0, default_sigma]],
                               dtype=np.float32))[None].expand((J, 2, 2))
        sg = sg / D ** 2
    elif voronoi == 'gaussian-orientation':
        L, V = eigh_2x2(sigma)
        L = L.detach()
        L = L / (L[:, 0:1] * L[:, 1:2]).sqrt() * default_sigma
        sg = jt.matmul(jt.matmul(V, diag_embed_2x2(L)), V.permute(0, 2, 1)).detach()
        sg = sg / D ** 2
    elif voronoi == 'gaussian-full':
        sg = sigma.detach() / D ** 2

    from jdet.ops.linalg2x2 import inv_2x2
    inv_sg = inv_2x2(sg.detach())  # (J,2,2)
    xy_flat = xy.view(-1, 2)          # (hw,2)
    hw = xy_flat.shape[0]
    best_val = jt.full((hw,), -1.0)
    best_idx = jt.zeros((hw,), dtype='int32')
    CHUNK = 256
    for start in range(0, J, CHUNK):
        end = min(start + CHUNK, J)
        mu_c = mm[start:end]                       # (C,2)
        ic = inv_sg[start:end]                     # (C,2,2)
        dx = xy_flat[None, :, 0] - mu_c[:, 0:1]    # (C,hw)
        dy = xy_flat[None, :, 1] - mu_c[:, 1:2]
        q = ic[:, 0, 0][:, None] * dx * dx \
            + (ic[:, 0, 1] + ic[:, 1, 0])[:, None] * dx * dy \
            + ic[:, 1, 1][:, None] * dy * dy
        t0 = jt.exp(-0.5 * q)                      # (C,hw)
        idx_local, vmax_c = jt.argmax(t0, 0)
        update = vmax_c > best_val                 # 平局保留更早 chunk（与 stack+argmax 一致）
        best_idx = jt.where(update, (idx_local + start).int32(), best_idx)
        best_val = jt.where(update, vmax_c, best_val)
        best_idx.sync(); best_val.sync()           # 分块落地，图规模有界
    vor = best_idx.view(h, w)
    val = best_val.view(h, w)
    if D > 1:
        vor = vor[:, None, :, None].expand((h, D, w, D)).reshape(H, W)
        val = nn.interpolate(val[None, None], size=(H, W), mode='bilinear',
                             align_corners=True)[0, 0]
    cls = label[vor]
    kernel = jt.ones((1, 1, 3, 3), dtype=val.dtype)
    kernel[0, 0, 1, 1] = -8
    ridges = nn.conv2d(vor[None].float().unsqueeze(0), kernel, padding=1)[0, 0] != 0
    vor = vor + 1
    if not isinstance(pos_thres, jt.Var):
        pos_thres = jt.array(np.asarray(pos_thres, dtype=np.float32))
    if not isinstance(neg_thres, jt.Var):
        neg_thres = jt.array(np.asarray(neg_thres, dtype=np.float32))
    # 上游三连 in-place：vor[val<pos]=0 → vor[val<neg]=J+1 → vor[ridges]=J+1
    # （顺序敏感，neg 是 pos 的子集、ridges 最后覆盖）→ out-of-place 等价
    vor = jt.where(val < pos_thres[cls], jt.zeros_like(vor), vor)
    vor = jt.where(val < neg_thres[cls], jt.full_like(vor, J + 1), vor)
    vor = jt.where(ridges, jt.full_like(vor, J + 1), vor)

    # PyTorch/Jittor 不支持 watershed，用 cv2（CPU、无梯度路径）
    img_uint8 = (image - image.min()) / (image.max() - image.min()) * 255
    img_uint8 = img_uint8.permute(1, 2, 0).detach().numpy().astype(np.uint8)
    img_uint8 = cv2.medianBlur(img_uint8, 3)
    markers = vor.detach().numpy().astype(np.int32)
    markers = jt.array(cv2.watershed(img_uint8, markers))

    L, V = eigh_2x2(sigma)
    # L_target 在上游即整体 detach（loss 用 L_target.detach()）→ 全程 numpy 计算，
    # 避免 O(J) 逐实例 jt 子图（同上，J 大时惰性图爆炸）
    markers_np = markers.detach().numpy()
    mu_np = mu.detach().numpy()
    V_np = V.detach().numpy()
    L_np = L.detach().numpy()
    L_target_np = np.empty((J, 2), dtype=np.float32)
    # 逐实例 np.nonzero 是 O(J·HW)（J 上万的密集 patch 卡数分钟）→
    # 单次 argsort 按 marker id 分桶，总体 O(HW log HW)
    Hm, Wm = markers_np.shape
    flat = markers_np.ravel()
    order = np.argsort(flat, kind='stable')
    sorted_ids = flat[order]
    # 每个 id 的像素区间 [lo, hi)
    los = np.searchsorted(sorted_ids, np.arange(1, J + 1), side='left')
    his = np.searchsorted(sorted_ids, np.arange(1, J + 1), side='right')
    ys_all, xs_all = np.divmod(order, Wm)
    for j in range(J):
        lo, hi = los[j], his[j]
        if lo == hi:
            L_target_np[j] = L_np[j]
            continue
        xy_j = np.stack([xs_all[lo:hi], ys_all[lo:hi]], 1).astype(np.float32) - mu_np[j]
        xy_j = xy_j @ V_np[j]  # (V^T x)^T = x^T V
        L_target_np[j, 0] = np.abs(xy_j[:, 0]).max() ** 2
        L_target_np[j, 1] = np.abs(xy_j[:, 1]).max() ** 2
    L_target = jt.array(L_target_np)
    L = diag_embed_2x2(L)
    L_target = diag_embed_2x2(L_target)
    loss = gwd_sigma_loss(L, L_target.detach(), reduction='none')
    # torch.topk(largest=False)[0].mean() → 升序排序取前 k
    k = int(np.ceil(loss.shape[0] * topk))
    sort_idx, sorted_loss = jt.argsort(loss)
    loss = sorted_loss[:k].mean()
    return loss, (vor, markers)


def get_loss_from_mask(mu, sigma, label, image, pos_thres, neg_thres, down_sample=2,
                       topk=0.95, default_sigma=4096, voronoi='gaussian-orientation',
                       alpha=0.1, debug=False, mask_filter_config=None,
                       sam_checkpoint='./mobile_sam.pt', model_type='vit_t',
                       sam_instance_thr=-1, device=None, sam_sample_rules=None):
    J = len(sigma)
    if J == 0:
        return sigma.sum(), None
    if J <= sam_instance_thr:
        loss, markers = segment_anything(
            image, mu, sigma,
            device=device,
            sam_checkpoint=sam_checkpoint,
            model_type=model_type,
            label=label,
            debug=debug,
            mask_filter_config=mask_filter_config,
            sam_sample_rules=sam_sample_rules)
        vor = markers.clone()
        return loss, (vor, markers)
    else:
        loss, (vor, markers) = voronoi_watershed_loss(
            mu, sigma, label, image,
            pos_thres, neg_thres, down_sample, topk,
            default_sigma, voronoi, alpha,
            debug=debug)
        return loss, (vor, markers)


@LOSSES.register_module()
class VoronoiWatershedLoss(nn.Module):
    """VoronoiWatershedLoss（官方 config：loss_weight=5.0, voronoi='standard'）。

    v3 扩展参数（mask_filter_config/sam_instance_thr/sam_sample_rules/
    use_class_specific_watershed）默认值即 v2 行为。
    """

    def __init__(self,
                 loss_weight=1.0,
                 down_sample=2,
                 topk=0.95,
                 alpha=0.1,
                 default_sigma=4096,
                 debug=False,
                 mask_filter_config=None,
                 sam_instance_thr=-1,
                 sam_sample_rules=None,
                 use_class_specific_watershed=False):
        super(VoronoiWatershedLoss, self).__init__()
        self.loss_weight = loss_weight
        self.down_sample = down_sample
        self.topk = topk
        self.alpha = alpha
        self.default_sigma = default_sigma
        self.debug = debug
        self.mask_filter_config = mask_filter_config
        self.sam_instance_thr = sam_instance_thr
        self.sam_sample_rules = sam_sample_rules
        self.use_class_specific_watershed = use_class_specific_watershed
        self.vis = None

    def execute(self, pred, label, image, pos_thres, neg_thres, voronoi='orientation'):
        loss, self.vis = get_loss_from_mask(
            *pred,
            label,
            image,
            pos_thres,
            neg_thres,
            self.down_sample,
            default_sigma=self.default_sigma,
            topk=self.topk,
            voronoi=voronoi,
            alpha=self.alpha,
            debug=self.debug,
            mask_filter_config=self.mask_filter_config,
            sam_instance_thr=self.sam_instance_thr,
            sam_sample_rules=self.sam_sample_rules)
        return self.loss_weight * loss


def rbbox2roi(bbox_list):
    """list of (N_i, 5+) rboxes → (N, 6) rois [batch_ind, cx, cy, w, h, a]。"""
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.shape[0] > 0:
            img_inds = jt.full((bboxes.shape[0], 1), img_id, dtype=bboxes.dtype)
            rois = jt.concat([img_inds, bboxes[:, :5]], dim=-1)
        else:
            rois = jt.zeros((0, 6), dtype=bboxes.dtype)
        rois_list.append(rois)
    rois = jt.concat(rois_list, 0)
    return rois


@LOSSES.register_module()
class EdgeLoss(nn.Module):
    """Edge Loss（官方 config：loss_weight=0.3）。"""

    def __init__(self,
                 resolution=24,
                 max_scale=1.6,
                 sigma=6,
                 reduction='mean',
                 loss_weight=1.0,
                 debug=False):
        super(EdgeLoss, self).__init__()
        self.resolution = resolution
        self.max_scale = max_scale
        self.sigma = sigma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.center_idx = self.resolution / self.max_scale
        self.debug = debug

        from jdet.models.roi_extractors.rotated_single_level import \
            RotatedSingleRoIExtractor
        self.roi_extractor = RotatedSingleRoIExtractor(
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=(2 * self.resolution + 1),
                sample_num=2,
                clockwise=True),
            out_channels=1,
            featmap_strides=[1],
            finest_scale=1024)

        edge_idx = jt.arange(0, self.resolution + 1).float()
        edge_distribution = jt.exp(-((edge_idx - self.center_idx) ** 2) / (2 * self.sigma ** 2))
        edge_distribution[0] = 0
        edge_distribution[-1] = 0
        # These are torch ``register_buffer`` constants, not graph detach
        # sites. Jittor exposes every Var attached to a Module through
        # ``parameters()``; detach() alone therefore lets AdamW weight decay
        # them and corrupt EdgeLoss during long runs.
        self.edge_idx = edge_idx.stop_grad()
        self.edge_distribution = edge_distribution.stop_grad()

    def execute(self, pred, edge):
        G = self.resolution
        C = self.center_idx
        roi = rbbox2roi(pred)
        # roi[:, 3:5] *= max_scale → out-of-place
        roi = jt.concat([roi[:, :3], roi[:, 3:5] * self.max_scale, roi[:, 5:6]], dim=1)
        feat = self.roi_extractor([edge], roi)
        if feat.shape[0] == 0:
            return jt.zeros(1).sum()
        featx = feat.sum(1).abs().sum(1)
        featy = feat.sum(1).abs().sum(2)
        featx2 = featx[:, :G + 1].flip(-1) + featx[:, G:]
        featy2 = featy[:, :G + 1].flip(-1) + featy[:, G:]  # (N, 25)
        ex = (nn.softmax(featx2 * self.edge_distribution, dim=1) * self.edge_idx).sum(1) / C
        ey = (nn.softmax(featy2 * self.edge_distribution, dim=1) * self.edge_idx).sum(1) / C
        exy = jt.stack([ex, ey], -1)
        rbbox_concat = jt.concat(pred, 0)

        return self.loss_weight * _smooth_l1(rbbox_concat[:, 2:4],
                                             (rbbox_concat[:, 2:4] * exy).detach(),
                                             beta=8)


@LOSSES.register_module()
class Point2RBoxV2ConsistencyLoss(nn.Module):
    """Consistency Loss（官方 config：loss_weight=1.0）。"""

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):
        super(Point2RBoxV2ConsistencyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(self, ori_pred, trs_pred, square_mask, aug_type, aug_val):
        ori_gaus, ori_angle = ori_pred
        trs_gaus, trs_angle = trs_pred

        if aug_type == 'rot':
            rot = jt.array(np.asarray(aug_val, dtype=np.float32))
            cos_r = jt.cos(rot)
            sin_r = jt.sin(rot)
            R = jt.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
            # GPU 的 cublas_batched_matmul 不广播 batch 维：R 是 (1,2,2) 需 expand
            R = R.expand((ori_gaus.shape[0], 2, 2))
            ori_gaus = jt.matmul(jt.matmul(R, ori_gaus), R.permute(0, 2, 1))
            d_ang = trs_angle - ori_angle - aug_val
        elif aug_type == 'flp':
            ori_gaus = ori_gaus * jt.array(
                np.array([1, -1, -1, 1], dtype=np.float32)).reshape(2, 2)
            d_ang = trs_angle + ori_angle
        else:
            sca = jt.array(np.asarray(aug_val, dtype=np.float32))
            ori_gaus = ori_gaus * sca
            d_ang = trs_angle - ori_angle

        loss_ssg = gwd_sigma_loss(jt.matmul(ori_gaus, ori_gaus),
                                  jt.matmul(trs_gaus, trs_gaus))
        d_ang = (d_ang + math.pi / 2) % math.pi - math.pi / 2
        loss_ssa = _smooth_l1(d_ang, jt.zeros_like(d_ang), beta=0.1, reduction='none')
        # loss_ssa[~square_mask].sum() / max(1, (~square_mask).sum())
        # 的 out-of-place 等价实现。v3 的 angle coder 会令 loss_ssa 为
        # (N, 1)，而 square_mask/keep 是 (N,)；直接相乘会广播成
        # (N, N) 外积。给 keep 补尾部 singleton 维，保持 torch 沿
        # 第 0 维布尔索引的语义。
        keep = (jt.logical_not(square_mask)).float()
        keep_shape = (keep.shape[0],) + (1,) * (len(loss_ssa.shape) - 1)
        keep_broadcast = keep.reshape(keep_shape)
        loss_ssa = ((loss_ssa * keep_broadcast).sum()
                    / jt.maximum(keep.sum(), jt.float32(1.0)))

        return self.loss_weight * (loss_ssg + loss_ssa)


@LOSSES.register_module()
class MMDetFocalLoss(nn.Module):
    """mmdet FocalLoss 的 Jittor 等价实现（0-based 标签，bg=num_classes）。

    底座 FocalLoss 是 1-based 标签约定（v1 数据集），与 mmdet 的
    py_sigmoid_focal_loss 语义不同（one-hot 列映射差一位、bg 处理不同），
    v2 head 使用本类。公式对齐 mmdet.models.losses.focal_loss.py_sigmoid_focal_loss。
    """

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(MMDetFocalLoss, self).__init__()
        assert use_sigmoid is True
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        num_classes = pred.shape[1]
        # F.one_hot(target, num_classes+1)[:, :num_classes]：bg 行全 0
        idx = jt.arange(num_classes)[None, :]
        onehot = (target[:, None] == idx).float()

        pred_sigmoid = pred.sigmoid()
        pt = (1 - pred_sigmoid) * onehot + pred_sigmoid * (1 - onehot)
        focal_weight = (self.alpha * onehot + (1 - self.alpha) * (1 - onehot)) \
            * pt.pow(self.gamma)
        # BCE with logits（数值稳定形式）
        bce = jt.maximum(pred, jt.zeros_like(pred)) - pred * onehot \
            + jt.log(1 + jt.exp(-jt.abs(pred)))
        loss = bce * focal_weight
        if weight is not None:
            if weight.ndim == 1:
                weight = weight.reshape(-1, 1)
            loss = loss * weight
        loss = weight_reduce_loss(loss, None, reduction, avg_factor)
        return self.loss_weight * loss


@LOSSES.register_module()
class MMDetCrossEntropyLoss(nn.Module):
    """mmdet CrossEntropyLoss(use_sigmoid=True) 的 Jittor 等价（centerness 用）。

    binary_cross_entropy_with_logits + mmdet weight_reduce 语义
    （底座 CrossEntropyLoss 的 avg_factor/加权语义不同，不复用）。
    """

    def __init__(self, use_sigmoid=True, reduction='mean', loss_weight=1.0):
        super(MMDetCrossEntropyLoss, self).__init__()
        assert use_sigmoid is True, '本移植只覆盖 sigmoid 分支（stage-2 centerness）'
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        target = target.float32()
        # BCE with logits（数值稳定形式）
        loss = jt.maximum(pred, jt.zeros_like(pred)) - pred * target \
            + jt.log(1 + jt.exp(-jt.abs(pred)))
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return self.loss_weight * loss
