"""SKU110K-R 数据集（铁律三第 2 层）。

格式（ref mmrotate/datasets/sku110k.py）：json 列表
[{image_id, rbbox|bbox, ...}]，单类 'object'（label 恒 0）；
ann_file='' 时按图像 glob 出空标注（测试模式）。
"""
import json
import os

import numpy as np

from jdet.utils.registry import DATASETS
from .dota_txt_variant import DOTATxtVariantDataset


@DATASETS.register_module()
class SKU110KDataset(DOTATxtVariantDataset):
    CLASSES = ('object',)
    IMG_SUFFIX = 'jpg'

    def __init__(self, ann_json_file=None, **kwargs):
        self.ann_json_file = ann_json_file
        kwargs.setdefault('annfiles_dir', kwargs.get('images_dir'))
        super().__init__(**kwargs)

    def _load_annfiles(self):
        self.CLASSES = list(type(self).CLASSES)
        self.cls2idx = {c: i for i, c in enumerate(self.CLASSES)}
        img_infos = []
        if self.ann_json_file:
            with open(self.ann_json_file) as f:
                root = json.load(f)
            instances = {}
            for item in root:
                # ref：优先 rbbox（5 元 rbox），否则 bbox
                box = item['rbbox'] if 'rbbox' in item else item['bbox']
                instances.setdefault(item['image_id'], []).append(box)
            for img_id in sorted(instances.keys(), key=str):
                boxes = np.array(instances[img_id], dtype=np.float32)
                if boxes.shape[-1] == 4:  # hbb → rbox(angle=0)
                    cx = (boxes[:, 0] + boxes[:, 2]) / 2
                    cy = (boxes[:, 1] + boxes[:, 3]) / 2
                    boxes = np.stack([cx, cy, boxes[:, 2] - boxes[:, 0],
                                      boxes[:, 3] - boxes[:, 1],
                                      np.zeros_like(cx)], 1)
                img_infos.append(dict(
                    filename=str(img_id) + '.' + self._img_suffix,
                    ann=dict(bboxes=boxes.reshape(-1, 5),
                             labels=np.zeros(len(boxes), dtype=np.int32),
                             bboxes_ignore=np.zeros((0, 5), dtype=np.float32),
                             labels_ignore=np.zeros((0,), dtype=np.int32))))
        else:
            for fname in sorted(os.listdir(self.images_dir)):
                if fname.endswith('.' + self._img_suffix):
                    img_infos.append(dict(
                        filename=fname,
                        ann=dict(bboxes=np.zeros((0, 5), dtype=np.float32),
                                 labels=np.zeros((0,), dtype=np.int32),
                                 bboxes_ignore=np.zeros((0, 5), dtype=np.float32),
                                 labels_ignore=np.zeros((0,), dtype=np.int32))))
        return img_infos
