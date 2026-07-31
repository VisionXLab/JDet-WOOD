"""mmrotate RotatedSingleRoIExtractor 的 Jittor 移植（EdgeLoss 依赖）。

roi_layer 语义对齐 mmcv.ops.RoIAlignRotated：out_size/sample_num 是 mmcv 的
旧参数名（→ output_size/sampling_ratio），aligned 默认 True。
"""
import math

import jittor as jt
from jittor import nn

from jdet.utils.registry import MODELS
from jdet.ops.roi_align_rotated import ROIAlignRotated


@MODELS.register_module()
class RotatedSingleRoIExtractor(nn.Module):

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56,
                 init_cfg=None):
        super(RotatedSingleRoIExtractor, self).__init__()
        cfg = dict(roi_layer)
        layer_type = cfg.pop('type')
        assert layer_type == 'RoIAlignRotated', f'unsupported roi layer {layer_type}'
        out_size = cfg.pop('out_size', cfg.pop('output_size', None))
        sample_num = cfg.pop('sample_num', cfg.pop('sampling_ratio', 0))
        aligned = cfg.pop('aligned', True)        # mmcv 默认 True
        clockwise = cfg.pop('clockwise', False)   # mmcv 默认 False
        assert not cfg, f'unknown roi_layer args: {cfg}'
        self.roi_layers = [
            ROIAlignRotated(out_size, spatial_scale=1 / s, sampling_ratio=sample_num,
                            aligned=aligned, clockwise=clockwise)
            for s in featmap_strides
        ]
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale
        self.out_size = self.roi_layers[0].output_size

    def map_roi_levels(self, rois, num_levels):
        scale = jt.sqrt(rois[:, 3] * rois[:, 4])
        target_lvls = jt.floor(jt.log2(scale / self.finest_scale + 1e-6))
        return target_lvls.clamp(0, num_levels - 1).int32()

    def execute(self, feats, rois, roi_scale_factor=None):
        num_levels = len(feats)
        if rois.shape[0] == 0:
            return jt.zeros((0, self.out_channels) + tuple(self.out_size), dtype=feats[0].dtype)

        if num_levels == 1:
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)
        if roi_scale_factor is not None:
            rois = jt.concat([rois[:, 0:1], rois[:, 1:3],
                              rois[:, 3:5] * roi_scale_factor, rois[:, 5:6]], dim=1)
        roi_feats = jt.zeros((rois.shape[0], self.out_channels) + tuple(self.out_size),
                             dtype=feats[0].dtype)
        for i in range(num_levels):
            mask = target_lvls == i
            if mask.any():
                inds = jt.where(mask)[0]
                roi_feats[inds] = self.roi_layers[i](feats[i], rois[inds])
        return roi_feats
