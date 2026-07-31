# Module-level pure functions of the Point2RBox-v3 detector, ported to
# Jittor translation of Point2RBox-v3 detector helper functions.
# (L28-125 module functions + L553-643 voronoi_diagram_watershed lifted to a
# standalone function; the detector class wraps these thinly).
#
# Porting rules applied (docs/porting_notes.md):
# - 2x2 linear algebra goes through jdet.ops.linalg2x2 (solve_2x2 / eigh_2x2 /
#   diag_embed_2x2) — no hand-rolled eigh here.
# - torch.meshgrid(indexing='xy'/'ij') is replaced by explicit broadcasting to
#   avoid jt.meshgrid convention ambiguity.
# - In-place masked assignments are rewritten as jt.where chains in the same
#   order (integer/marker domain, gradients already stopped there).
# - cv2.watershed / cv2.medianBlur stay on the CPU numpy path; inputs pass
#   through .detach() before .numpy() so no gradient ever crosses.
# - Randomness sites accept an optional `rng` dict so tests can inject fixed
#   values; the default (rng=None) keeps the original live-random behavior.
#
# Upstream quirks kept verbatim (do NOT "fix"):
# - torch.linspace(0, h, h): h points spanning [0, h] inclusive (step
#   h/(h-1)), NOT arange(h).
# - meshgrid 'xy' + .view(h, w): axes only line up because DOTA patches are
#   square (h == w after down_sample); copied as-is.
# - `vor += 1` then thresholds then ridges, in that exact order.
# - cls_bg is computed but unused by the return value (upstream leftover).
# - get_single_pattern raises on out-of-range sizes and
#   get_copy_paste_cache swallows every exception; the `len(patterns) >
#   num_copies` check means up to num_copies+1 patterns are returned.

import cv2
import numpy as np

import jittor as jt
from jittor import nn

from jdet.ops.linalg2x2 import solve_2x2, eigh_2x2, diag_embed_2x2, inv_2x2


def _voronoi_val_idx(xy_flat, mm, sg_all, chunk=64):
    """分块批量化的逐实例 gaussian 响应 argmax。

    等价于 stack([gaussian_2d(xy, mm[j], sg[j]) for j]) 后 jt.argmax(dim=0)，
    但把 O(J) 的 python 循环换成 O(J/chunk) —— 密集图（J~1000）下逐实例循环
    会让 jittor 惰性图融合优化超线性爆炸、单 batch 卡 15+ 分钟（A commit
    1aa8ad8 在 loss 侧同款修复）。逐元素算式与 gaussian_2d(solve_2x2/inv_2x2)
    完全一致（bit 级），平局语义与 jt.argmax 首现优先一致（后块严格 > 才胜）。

    xy_flat: (N, 2); mm: (J, 2); sg_all: (J, 2, 2) 或 (1, 2, 2)（standard 共享）。
    返回 (idx (N,) int32, val (N,) float32)。
    """
    N = xy_flat.shape[0]
    J = mm.shape[0]
    shared_sg = sg_all.shape[0] == 1
    xs = xy_flat[:, 0].unsqueeze(0)                     # (1, N)
    ys = xy_flat[:, 1].unsqueeze(0)
    val = jt.full((N,), -1e30, dtype='float32')
    idx = jt.zeros((N,), dtype='int32')
    for o in range(0, J, chunk):
        mm_c = mm[o:o + chunk]                          # (C, 2)
        sg_c = sg_all if shared_sg else sg_all[o:o + chunk]
        inv = inv_2x2(sg_c)                             # (C|1, 2, 2)
        i00 = inv[:, 0, 0].unsqueeze(-1)
        i01 = inv[:, 0, 1].unsqueeze(-1)
        i10 = inv[:, 1, 0].unsqueeze(-1)
        i11 = inv[:, 1, 1].unsqueeze(-1)
        dx = xs - mm_c[:, 0].unsqueeze(-1)              # (C, N)
        dy = ys - mm_c[:, 1].unsqueeze(-1)
        sol_x = i00 * dx + i01 * dy
        sol_y = i10 * dx + i11 * dy
        g = jt.exp(-0.5 * (dx * sol_x + dy * sol_y))    # (C, N)
        cidx, cval = jt.argmax(g, dim=0)
        better = cval > val
        val = jt.where(better, cval, val)
        idx = jt.where(better, (cidx + o).int32(), idx)
    return idx, val


def _det_2x2(sigma):
    return (sigma[..., 0, 0] * sigma[..., 1, 1]
            - sigma[..., 0, 1] * sigma[..., 1, 0])


def gaussian_2d(xy, mu, sigma, normalize=False):
    """xy: (N, 2); mu: (1, 2) or (N, 2); sigma: (1, 2, 2) or (N, 2, 2)."""
    dxy = (xy - mu).unsqueeze(-1)                       # (N, 2, 1)
    sol = solve_2x2(sigma, dxy)                         # (N, 2, 1) broadcast
    t0 = jt.exp(-0.5 * jt.matmul(dxy.permute(0, 2, 1), sol))
    if normalize:
        t0 = t0 / (2 * np.pi * _det_2x2(sigma).clamp(1e-7).sqrt())
    return t0


def get_pattern_gaussian(w, h, rng=None):
    w, h = int(w), int(h)
    y = (jt.arange(h).float32() - h / 2) / (h / 2)
    x = (jt.arange(w).float32() - w / 2) / (w / 2)
    y = y.unsqueeze(1).expand(h, w)
    x = x.unsqueeze(0).expand(h, w)
    if rng is None:
        oxy = jt.randn(2).clamp(-3, 3) * 0.15
        sxy = jt.rand(2) * 0.5 + 1
    else:
        oxy = jt.array(np.float32(rng['randn2'])).clamp(-3, 3) * 0.15
        sxy = jt.array(np.float32(rng['rand2'])) * 0.5 + 1
    ox, oy = oxy[0], oxy[1]
    sx, sy = sxy[0], sxy[1]
    z = jt.exp(-((x - ox) * sx) ** 2 - ((y - oy) * sy) ** 2) * 0.5 + 0.5
    return z


def get_single_pattern(image, bbox, label, square_cls, rng=None):
    if bbox[2] < 16 or bbox[3] < 16 or bbox[2] > 512 or bbox[3] > 512:
        raise ValueError('pattern bbox size out of [16, 512]')

    def obb2poly(obb):
        cx, cy, w, h, t = obb
        dw, dh = (w - 1) / 2, (h - 1) / 2
        cost = np.cos(t)
        sint = np.sin(t)
        mrot = np.float32([[cost, -sint], [sint, cost]])
        poly = np.float32([[-dw, -dh], [dw, -dh], [dw, dh], [-dw, dh]])
        return np.matmul(poly, mrot.T) + np.float32([cx, cy])

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
    theta = np.matmul(T, np.linalg.inv(M))
    theta = jt.array(np.float32(theta[:2, :]))[None]
    grid = nn.affine_grid(theta, [1, 3, h, w], align_corners=True)
    chip = nn.grid_sample(image[None], grid, align_corners=True)[0]

    alpha = get_pattern_gaussian(chip.shape[-1], chip.shape[-2], rng=rng)[None]
    chip = jt.concat((chip, alpha))

    if rng is None:
        r3 = np.random.rand(3)
    else:
        r3 = np.asarray(rng['rand3'], dtype=np.float64)
    w2 = float(bbox[2] * (0.7 + 0.5 * r3[0]))
    h2 = float(bbox[3] * (0.7 + 0.5 * r3[1]))
    t2 = float(np.pi * r3[2])
    if label in square_cls:
        t2 = t2 * 0
    w_t = jt.float32(w2)
    h_t = jt.float32(h2)
    t_t = jt.float32(t2)
    cosa = jt.abs(jt.cos(t_t))
    sina = jt.abs(jt.sin(t_t))
    sx = int(jt.ceil(cosa * w_t + sina * h_t).item())
    sy = int(jt.ceil(sina * w_t + cosa * h_t).item())
    theta2 = jt.concat([
        1 / w_t * jt.cos(t_t), 1 / w_t * jt.sin(t_t), jt.zeros_like(t_t),
        1 / h_t * jt.sin(-t_t), 1 / h_t * jt.cos(t_t), jt.zeros_like(t_t),
    ]).reshape(2, 3)
    scale = jt.array(np.float32([[sx, 0], [0, sy]]))
    theta2 = jt.concat([jt.matmul(theta2[:, :2], scale), theta2[:, 2:]], dim=1)
    grid = nn.affine_grid(theta2[None], (1, 1, sy, sx), align_corners=True)
    chip = nn.grid_sample(chip[None], grid, align_corners=True,
                          mode='nearest')[0]
    bbox = np.float32([sx / 2, sy / 2, w2, h2, t2])
    return (chip, bbox, label)


def get_copy_paste_cache(images, bboxes, labels, square_cls, num_copies,
                         rng_list=None):
    bboxes = bboxes.numpy()
    labels = labels.numpy()
    patterns = []
    for i, (b, l) in enumerate(zip(bboxes, labels)):
        try:
            rng = None if rng_list is None else rng_list[i]
            p = get_single_pattern(images, b, l, square_cls, rng=rng)
            patterns.append(p)
            if len(patterns) > num_copies:
                break
        except Exception:
            pass
    return patterns


def _meshgrid_xy_stack(x, y):
    """Replicates torch.meshgrid(x, y, indexing='xy') then stack(..., -1).

    Output shape (len(y), len(x), 2) with out[i, j] = (x[j], y[i]).
    """
    h = x.shape[0]
    w = y.shape[0]
    gx = x.unsqueeze(0).expand(w, h)
    gy = y.unsqueeze(1).expand(w, h)
    return jt.stack([gx, gy], dim=-1)


def voronoi_diagram_watershed(mu, sigma, label, image,
                              voronoi_type, voronoi_thres, num_classes,
                              down_sample=2, default_sigma=4096):
    """Standalone port of Point2RBoxV3.voronoi_diagram_watershed.

    mu: (J, 2) jt.Var; sigma: (J, 2, 2) jt.Var or None (standard branch);
    label: (J,) int jt.Var; image: (3, H, W) jt.Var.
    Returns the flat pseudo_info list exactly like upstream.
    """
    J = len(mu)
    D = down_sample

    pos_thres = [voronoi_thres['default'][0]] * num_classes
    neg_thres = [voronoi_thres['default'][1]] * num_classes
    if 'override' in voronoi_thres.keys():
        for item in voronoi_thres['override']:
            for cls in item[0]:
                pos_thres[cls] = item[1][0]
                neg_thres[cls] = item[1][1]

    H, W = image.shape[-2:]
    h, w = H // D, W // D
    x = jt.linspace(0, h, h).float32()
    y = jt.linspace(0, w, w).float32()
    xy = _meshgrid_xy_stack(x, y)
    mm = (mu.detach() / D).round()
    if voronoi_type == 'standard':
        sg = jt.array(np.float32([default_sigma, 0, 0, default_sigma])
                      ).reshape(1, 2, 2)
        sg = sg / D ** 2
    elif voronoi_type == 'gaussian-orientation':
        L, V = eigh_2x2(sigma)
        L = L.detach().clone()
        L = L / (L[:, 0:1] * L[:, 1:2]).sqrt() * default_sigma
        sg = jt.matmul(jt.matmul(V, diag_embed_2x2(L)),
                       V.permute(0, 2, 1)).detach()
        sg = sg / D ** 2
    elif voronoi_type == 'gaussian-full':
        sg = sigma.detach() / D ** 2
    # 分块批量化（原逐实例循环在密集图卡死，见 _voronoi_val_idx 注释）
    vor_idx, val = _voronoi_val_idx(xy.view(-1, 2), mm, sg)
    vor = vor_idx.view(h, w)
    val = val.view(h, w)
    if D > 1:
        vor = vor[:, None, :, None].repeat(1, D, 1, D).reshape(H, W)
        val = nn.interpolate(val[None, None], size=(H, W), mode='bilinear',
                             align_corners=True)[0, 0]
    cls = label[vor]
    kernel = jt.ones((1, 1, 3, 3)).float32()
    kernel[0, 0, 1, 1] = -8
    ridges = nn.conv2d(vor[None, None].float32(), kernel,
                       padding=1)[0, 0] != 0
    pos_thres = jt.array(np.float32(pos_thres))
    neg_thres = jt.array(np.float32(neg_thres))
    vor = vor + 1
    fill_bg = jt.ones_like(vor) * (J + 1)
    vor = jt.where(val < pos_thres[cls], jt.zeros_like(vor), vor)
    vor = jt.where(val < neg_thres[cls], fill_bg, vor)
    vor = jt.where(ridges, fill_bg, vor)

    # kept for parity with upstream (computed there, unused by the return)
    cls_bg = jt.where(vor == J + 1, jt.ones_like(cls) * num_classes, cls)
    cls_bg = jt.where(vor == 0, jt.ones_like(cls_bg) * -1, cls_bg)  # noqa: F841

    # cv2 watershed on CPU (gradients are already stopped above)
    img_min = image.min()
    img_max = image.max()
    img_uint8 = (image - img_min) / (img_max - img_min) * 255
    img_uint8 = img_uint8.permute(1, 2, 0).detach().numpy().astype(np.uint8)
    img_uint8 = cv2.medianBlur(img_uint8, 3)
    markers = vor.detach().numpy().astype(np.int32)
    markers = cv2.watershed(img_uint8, markers)

    # 原逐实例 `np.nonzero(markers == j+1)` 是 J 次全图扫描（J~1000 时每次
    # 调用数秒、每 iter 2B 视图 → 分钟级爬行，py-spy 实测热点）。改为单次
    # 前景扫描 + np.maximum.at 分组聚合；max 与逐实例结果逐位一致。
    pseudo_info = []
    mu_np = mu.detach().numpy()
    fg_ys, fg_xs = np.nonzero((markers >= 1) & (markers <= J))
    ids = markers[fg_ys, fg_xs] - 1                     # 0..J-1
    w_half = np.zeros(J, dtype=np.float32)
    h_half = np.zeros(J, dtype=np.float32)
    has_px = np.zeros(J, dtype=bool)
    has_px[np.unique(ids)] = True
    np.maximum.at(w_half, ids,
                  np.abs(fg_xs.astype(np.float32) - mu_np[ids, 0]))
    np.maximum.at(h_half, ids,
                  np.abs(fg_ys.astype(np.float32) - mu_np[ids, 1]))
    for j in range(J):
        pseudo_info.append(float(mu_np[j][0]))  # cx
        pseudo_info.append(float(mu_np[j][1]))  # cy
        if not has_px[j]:
            pseudo_info.append(0)  # w_half
            pseudo_info.append(0)  # h_half
        else:
            pseudo_info.append(float(w_half[j] * 2))
            pseudo_info.append(float(h_half[j] * 2))
        pseudo_info.append(0)  # angle, set 0

    return pseudo_info
