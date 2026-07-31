"""RotatedIoULoss（mmrotate rotated_iou_loss.py 的 Jittor 移植，stage-2 用）。"""
import jittor as jt
from jittor import nn

from jdet.utils.registry import LOSSES
from jdet.models.losses.gaussian_dist_loss import weight_reduce_loss
from jdet.ops.diff_iou_rotated import diff_iou_rotated_2d


def rotated_iou_loss(pred, target, weight=None, linear=False, mode='log',
                     eps=1e-6, reduction='mean', avg_factor=None):
    """@weighted_loss 展开形式。pred/target: (N, 5) xywha。"""
    assert mode in ['linear', 'square', 'log']
    if linear:
        mode = 'linear'
    ious = diff_iou_rotated_2d(pred.unsqueeze(0), target.unsqueeze(0))
    ious = ious.squeeze(0).clamp(eps)
    if mode == 'linear':
        loss = 1 - ious
    elif mode == 'square':
        loss = 1 - ious ** 2
    else:
        loss = -jt.log(ious)
    return weight_reduce_loss(loss, weight, reduction, avg_factor)


@LOSSES.register_module()
class RotatedIoULoss(nn.Module):
    """官方 stage-2 config：RotatedIoULoss(loss_weight=1.0)，默认 mode='log'。"""

    def __init__(self, linear=False, eps=1e-6, reduction='mean',
                 loss_weight=1.0, mode='log'):
        super(RotatedIoULoss, self).__init__()
        assert mode in ['linear', 'square', 'log']
        if linear:
            mode = 'linear'
        self.mode = mode
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None, **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and not jt.any(weight > 0) and reduction != 'none':
            if pred.ndim == weight.ndim + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        if weight is not None and weight.ndim > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * rotated_iou_loss(
            pred, target, weight, mode=self.mode, eps=self.eps,
            reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss
