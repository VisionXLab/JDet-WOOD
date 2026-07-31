"""dump_golden_fcos_head.py — torch 侧 dump stage-2 RotatedFCOSHead 的 parity golden。

配置逐项取自官方 rotated-fcos-1x-dota-using-pseudo.py 的 bbox_head 段
（center_sampling=True/r=1.5、norm_on_bbox=True、centerness_on_reg=True、
scale_angle=True、DistanceAnglePointCoder le90、FocalLoss、RotatedIoULoss、
loss_angle=None、CE centerness），仅缩小通道/卷积层数（512→64、4→2）控制
golden 体积——通道数不改变任何分支语义。

固定权重 + 固定输入（5 层 FPN feat + 跨尺度 GT）→ loss_dict 三项 + feat 梯度。
RotatedIoULoss 依赖 mmcv CUDA 算子（diff_iou_rotated_2d），必须在 GPU 上跑。

用法：P2R_REF=/root/ref/Point2RBox-v3 python tools/dump_golden_fcos_head.py
"""
import os
import sys

import numpy as np
import torch

REPO = os.environ.get('P2R_REF', '/root/ref/Point2RBox-v3')
sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'tests', 'parity', 'golden', 'fcos_head_parity.npz')

HEAD_CFG = dict(
    num_classes=15,
    in_channels=64,
    feat_channels=64,
    stacked_convs=2,
    strides=[8, 16, 32, 64, 128],
    center_sampling=True,
    center_sample_radius=1.5,
    norm_on_bbox=True,
    centerness_on_reg=True,
    use_hbbox_loss=False,
    scale_angle=True,
    bbox_coder=dict(type='DistanceAnglePointCoder', angle_version='le90'),
    loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0,
                  alpha=0.25, loss_weight=1.0),
    loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
    loss_angle=None,
    loss_centerness=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True,
                         loss_weight=1.0),
)

IMG = 512  # 对应 feat 尺寸 64/32/16/8/4


def make_inputs():
    rng = np.random.RandomState(77)
    feats = [rng.randn(2, 64, IMG // s, IMG // s).astype(np.float32) * 0.5
             for s in HEAD_CFG['strides']]
    gts = []
    for i in range(2):
        n = 10
        # 尺寸跨 5 个 regress_range，保证每层都有正样本
        wh = np.exp(rng.uniform(np.log(14), np.log(420), (n, 2))).astype(np.float32)
        rb = np.concatenate([
            rng.uniform(60, IMG - 60, (n, 2)).astype(np.float32),
            wh,
            rng.uniform(-np.pi / 2, np.pi / 2, (n, 1)).astype(np.float32)], 1)
        rb[0, 4] = 0.0  # 一个轴对齐盒
        labels = rng.randint(0, 15, n).astype(np.int64)
        gts.append((rb, labels))
    return feats, gts


def main():
    from mmrotate.utils import register_all_modules
    register_all_modules(True)
    from mmrotate.models.dense_heads.rotated_fcos_head import RotatedFCOSHead
    from mmrotate.structures.bbox import RotatedBoxes
    from mmengine.structures import InstanceData

    assert torch.cuda.is_available(), 'RotatedIoULoss 需要 CUDA'
    dev = 'cuda'
    torch.manual_seed(7)
    torch.backends.cudnn.deterministic = True
    # golden 必须全精度：torch 2.x CUDA 默认 TF32 卷积（10 位尾数），
    # 前向就带 ~4e-4 相对误差，会整体污染 golden（jittor 侧是 FP32）
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    head = RotatedFCOSHead(**HEAD_CFG).to(dev)
    # conv_reg bias 抬正：随机初始化下 clamp(0) 会产生大量 w/h=0 的退化 pred，
    # 其旋转 IoU 处于两侧实现各自的浮点噪声底（log 模式 -1/iou 放大到不可对齐），
    # golden 会带上 torch 自身噪声。抬 bias 后 pred 均为正常小框，IoU 几何良定义。
    # bias 在 state_dict 里，jittor 侧照常加载，不破坏 parity。
    with torch.no_grad():
        head.conv_reg.bias += 1.5
    head.train()

    feats_np, gts = make_inputs()
    feats = [torch.tensor(f, device=dev, requires_grad=True) for f in feats_np]

    batch_gt = []
    for rb, labels in gts:
        gi = InstanceData()
        gi.bboxes = RotatedBoxes(torch.tensor(rb, device=dev))
        gi.labels = torch.tensor(labels, device=dev)
        batch_gt.append(gi)
    metas = [dict(img_shape=(IMG, IMG, 3)) for _ in range(2)]

    outs = head(feats)
    losses = head.loss_by_feat(*outs, batch_gt, metas)
    total = sum(v.sum() for v in losses.values())
    total.backward(retain_graph=True)
    grad_total = [f.grad.detach().cpu().numpy() for f in feats]
    # cls+ctr 支路单独的梯度：不经 RotatedIoULoss（其 shoelace 在原始
    # 图像坐标上算，梯度自带 ~1e-3 噪声底），jittor 侧可对到 1e-4
    for f in feats:
        f.grad = None
    (losses['loss_cls'].sum() + losses['loss_centerness'].sum()).backward()
    grad_clsctr = [f.grad.detach().cpu().numpy() for f in feats]

    out = {}
    for i, f in enumerate(feats_np):
        out[f'feat{i}'] = f
        out[f'feat{i}_grad'] = grad_total[i]
        out[f'feat{i}_grad_clsctr'] = grad_clsctr[i]
    for i, (rb, labels) in enumerate(gts):
        out[f'gt{i}_rb'], out[f'gt{i}_lb'] = rb, labels
    for k, v in losses.items():
        out[f'loss_{k}'] = float(v.sum().item())
    # 前向输出也入 golden（比 loss 更细粒度的定位手段）
    names = ('cls', 'bbox', 'angle', 'ctr')
    for name, group in zip(names, outs):
        for lvl, t in enumerate(group):
            out[f'out_{name}{lvl}'] = t.detach().cpu().numpy()
    sd = {k: v.detach().cpu().numpy() for k, v in head.state_dict().items()}
    np.savez(OUT, **out, **{f'w__{k}': v for k, v in sd.items()})
    print('saved ->', OUT)
    for k, v in losses.items():
        print(' ', k, '=', float(v.sum().item()))
    print(' state_dict keys:', len(sd))


if __name__ == '__main__':
    main()
