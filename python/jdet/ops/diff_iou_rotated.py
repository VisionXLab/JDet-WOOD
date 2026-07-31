"""可微旋转 IoU（mmcv.ops.diff_iou_rotated_2d 的 Jittor 移植）。

RotatedIoULoss（stage-2）依赖。与 mmcv 实现逐式对应；唯一的 CUDA 件
SortVertices 本就不可微（mark_non_differentiable），此处用 numpy 按极角排序
实现——多边形面积（绝对值鞋带公式）对起点与方向不变，IoU 数值与 mmcv 一致。

in-place 写法已按 jittor 约定改 out-of-place（t[numerator==0]=-1 → jt.where 等）。
"""
import numpy as np
import jittor as jt

EPSILON = 1e-8


def box_intersection(corners1, corners2):
    """两组矩形边的交点。corners: (B,N,4,2) → 交点 (B,N,4,4,2) + 有效 mask (B,N,4,4)。"""
    line1 = jt.concat([corners1, corners1[:, :, [1, 2, 3, 0], :]], dim=3)
    line2 = jt.concat([corners2, corners2[:, :, [1, 2, 3, 0], :]], dim=3)
    line1_ext = line1.unsqueeze(3)
    line2_ext = line2.unsqueeze(2)
    x1 = line1_ext[..., 0:1]
    y1 = line1_ext[..., 1:2]
    x2 = line1_ext[..., 2:3]
    y2 = line1_ext[..., 3:4]
    x3 = line2_ext[..., 0:1]
    y3 = line2_ext[..., 1:2]
    x4 = line2_ext[..., 2:3]
    y4 = line2_ext[..., 3:4]
    numerator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    denumerator_t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    zero = numerator == 0.0
    t = jt.where(zero, jt.full_like(numerator, -1.0), denumerator_t / (numerator + EPSILON))
    mask_t = (t > 0) & (t < 1)
    denumerator_u = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
    u = jt.where(zero, jt.full_like(numerator, -1.0), -denumerator_u / (numerator + EPSILON))
    mask_u = (u > 0) & (u < 1)
    mask = (mask_t & mask_u)
    t_stable = denumerator_t / (numerator + EPSILON)
    intersections = jt.stack(
        [(x1 + t_stable * (x2 - x1)).squeeze(-1),
         (y1 + t_stable * (y2 - y1)).squeeze(-1)], dim=-1)
    intersections = intersections * mask.float().squeeze(-1).unsqueeze(-1)
    return intersections, mask.squeeze(-1)


def box1_in_box2(corners1, corners2):
    a = corners2[:, :, 0:1, :]
    b = corners2[:, :, 1:2, :]
    d = corners2[:, :, 3:4, :]
    ab = b - a
    am = corners1 - a
    ad = d - a
    prod_ab = (ab * am).sum(-1)
    norm_ab = (ab * ab).sum(-1)
    prod_ad = (ad * am).sum(-1)
    norm_ad = (ad * ad).sum(-1)
    cond1 = (prod_ab / norm_ab > -1e-6) & (prod_ab / norm_ab < 1 + 1e-6)
    cond2 = (prod_ad / norm_ad > -1e-6) & (prod_ad / norm_ad < 1 + 1e-6)
    return cond1 & cond2


def build_vertices(corners1, corners2, c1_in_2, c2_in_1, intersections, valid_mask):
    B, N = corners1.shape[0], corners1.shape[1]
    vertices = jt.concat(
        [corners1, corners2, intersections.view(B, N, -1, 2)], dim=2)
    mask = jt.concat([c1_in_2, c2_in_1, valid_mask.view(B, N, -1)], dim=2)
    return vertices, mask


def sort_indices_np(vertices_np, mask_np):
    """mmcv SortVertices 的 numpy 等价：有效顶点按极角升序 + 首点闭合 + 零值槽填充。

    输出 (B,N,9) int64。排序索引不可微（与 mmcv 相同），面积对起点/方向不变。
    """
    B, N = mask_np.shape[:2]
    idx = np.zeros((B, N, 9), dtype=np.int64)
    inter_invalid = ~mask_np[:, :, 8:]  # 后 16 个槽（交点区），无效处值为 0
    for b in range(B):
        for n in range(N):
            pad_c = np.nonzero(inter_invalid[b, n])[0]
            pad = int(pad_c[0]) + 8 if len(pad_c) else 0
            valid = np.nonzero(mask_np[b, n])[0]
            if len(valid) == 0:
                idx[b, n, :] = pad
                continue
            ang = np.arctan2(vertices_np[b, n, valid, 1],
                             vertices_np[b, n, valid, 0])
            order = valid[np.argsort(ang)]
            k = min(len(order), 8)
            idx[b, n, :k] = order[:k]
            idx[b, n, k] = order[0]
            idx[b, n, k + 1:] = pad
    return idx


def calculate_area(idx_sorted, vertices):
    """鞋带公式，梯度经 gather 流向顶点坐标。"""
    idx_ext = idx_sorted.unsqueeze(-1).expand(idx_sorted.shape + (2,))
    selected = jt.gather(vertices, 2, idx_ext)
    total = selected[:, :, 0:-1, 0] * selected[:, :, 1:, 1] \
        - selected[:, :, 0:-1, 1] * selected[:, :, 1:, 0]
    total = total.sum(2)
    area = jt.abs(total) / 2
    return area, selected


def oriented_box_intersection_2d(corners1, corners2):
    intersections, valid_mask = box_intersection(corners1, corners2)
    c12 = box1_in_box2(corners1, corners2)
    c21 = box1_in_box2(corners2, corners1)
    vertices, mask = build_vertices(corners1, corners2, c12, c21,
                                    intersections, valid_mask)
    # 均值归一化（排序在归一化坐标系做，mmcv 同）
    mask_f = mask.float()
    num_valid = mask_f.sum(2)  # (B,N)
    mean = (vertices * mask_f.unsqueeze(-1)).sum(2, keepdims=True) \
        / num_valid.unsqueeze(-1).unsqueeze(-1).clamp(1)
    vertices_normalized = vertices - mean
    idx_np = sort_indices_np(vertices_normalized.detach().numpy(),
                             mask.detach().numpy().astype(bool))
    idx = jt.array(idx_np.astype(np.int32))
    # ⚠️ 面积在「均值中心化 + mask 再归零」的坐标上算：
    #   - 闭合多边形的鞋带公式平移不变，中心化不改面积；
    #   - 填充槽必须仍是 (0,0) 才零贡献（裸减 mean 会让它变成 -mean），
    #     故中心化后乘 mask 把无效槽重新归零；
    #   - 不能像 mmcv 那样在原始图像坐标上算：坐标 ~1e2-1e3 时 x_i*y_j 项达 1e4-1e6，
    #     退化/极小交集的正负大项相消，jittor 融合出的 FMA 使两项舍入不再互为相反数，
    #     残差 ~1e-2 px² 成为假面积（torch eager 恰好精确抵消，golden 上实测差 16%）。
    vertices_centered = vertices_normalized * mask_f.unsqueeze(-1)
    return calculate_area(idx, vertices_centered)


def box2corners(box):
    """(B,N,5) xywha → (B,N,4,2) 角点（可微）。"""
    B, N = box.shape[0], box.shape[1]
    x = box[..., 0:1]
    y = box[..., 1:2]
    w = box[..., 2:3]
    h = box[..., 3:4]
    alpha = box[..., 4:5]
    x4 = jt.array(np.float32([0.5, -0.5, -0.5, 0.5])) * w
    y4 = jt.array(np.float32([0.5, 0.5, -0.5, -0.5])) * h
    corners = jt.stack([x4, y4], dim=-1)  # (B,N,4,2)
    sin = jt.sin(alpha)
    cos = jt.cos(alpha)
    row1 = jt.concat([cos, sin], dim=-1)
    row2 = jt.concat([-sin, cos], dim=-1)
    rot_T = jt.stack([row1, row2], dim=-2)  # (B,N,2,2)
    rotated = jt.nn.bmm(corners.view(-1, 4, 2), rot_T.view(-1, 2, 2))
    rotated = rotated.view(B, N, 4, 2)
    # rotated[...,0] += x → out-of-place
    rotated = jt.stack([rotated[..., 0] + x, rotated[..., 1] + y], dim=-1)
    return rotated


def diff_iou_rotated_2d(box1, box2):
    """(B,N,5) × (B,N,5) → (B,N) 可微 IoU。"""
    corners1 = box2corners(box1)
    corners2 = box2corners(box2)
    intersection, _ = oriented_box_intersection_2d(corners1, corners2)
    area1 = box1[:, :, 2] * box1[:, :, 3]
    area2 = box2[:, :, 2] * box2[:, :, 3]
    union = area1 + area2 - intersection
    iou = intersection / union
    return iou
