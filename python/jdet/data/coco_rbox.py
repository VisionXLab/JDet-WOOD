"""COCO-json 系旋转框数据集（铁律三第 2 层）：SARDet-100K / SRSDD / RSDD / HRSID。

格式判断以 ref 为准（与 data.md 的描述不一致处以此为准）：
- sardet100k：ref mmrotate/datasets/sardet100k.py，COCO API，6 类
- srsdd / rsdd / hrsid：ref 配置 `configs/_base_/datasets/*.py` 直接用
  `mmdet.CocoDataset` + config 内 metainfo（srsdd 6 类船型、rsdd/hrsid 单类
  'ship'）——**不是** data.md 表格里写的 DOTA txt / VOC xml。

实现：直接解析标准 COCO json（images/annotations/categories），不依赖
pycocotools；bbox [x,y,w,h] → rbox(angle=0)；若 annotation 带 8 元
segmentation/rbox 字段则走 minAreaRect。复用 P2RV2DOTADataset 的管线机制。
"""
import json
import os

import numpy as np

from jdet.utils.registry import DATASETS
from .dota_txt_variant import DOTATxtVariantDataset
from .p2rv2_dota import poly2rbox_le90_np


class COCORBoxDataset(DOTATxtVariantDataset):
    CLASSES = None
    IMG_SUFFIX = 'jpg'

    def __init__(self, ann_json_file=None, **kwargs):
        self.ann_json_file = ann_json_file
        kwargs.setdefault('annfiles_dir', kwargs.get('images_dir'))
        super().__init__(**kwargs)

    def _load_annfiles(self):
        self.CLASSES = list(type(self).CLASSES)
        self.cls2idx = {c: i for i, c in enumerate(self.CLASSES)}
        with open(self.ann_json_file) as f:
            coco = json.load(f)
        catid2idx = {}
        for cat in coco.get('categories', []):
            if cat['name'] in self.cls2idx:
                catid2idx[cat['id']] = self.cls2idx[cat['name']]
        anns_by_img = {}
        for ann in coco.get('annotations', []):
            anns_by_img.setdefault(ann['image_id'], []).append(ann)
        img_infos = []
        for img in coco.get('images', []):
            rboxes, labels = [], []
            for ann in anns_by_img.get(img['id'], []):
                if ann['category_id'] not in catid2idx:
                    continue
                seg = ann.get('segmentation')
                if seg and isinstance(seg, list) and len(seg) and \
                        len(seg[0]) == 8:
                    poly = np.array(seg[0], dtype=np.float32).reshape(1, 8)
                    rboxes.append(poly2rbox_le90_np(poly)[0])
                else:
                    x, y, w, h = ann['bbox']
                    rboxes.append([x + w / 2, y + h / 2, w, h, 0.0])
                labels.append(catid2idx[ann['category_id']])
            rboxes = np.array(rboxes, dtype=np.float32).reshape(-1, 5)
            img_infos.append(dict(
                filename=img['file_name'],
                ann=dict(bboxes=rboxes,
                         labels=np.array(labels, dtype=np.int32),
                         bboxes_ignore=np.zeros((0, 5), dtype=np.float32),
                         labels_ignore=np.zeros((0,), dtype=np.int32))))
        return img_infos


@DATASETS.register_module()
class SARDet100kDataset(COCORBoxDataset):
    """ref SAR_Det_Finegrained_Dataset METAINFO（6 类）。"""
    CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')


@DATASETS.register_module()
class SRSDDDataset(COCORBoxDataset):
    """ref configs/_base_/datasets/srsdd.py 的 metainfo.classes（6 类船型）。"""
    CLASSES = ('Container', 'Dredger', 'LawEnforce', 'Cell-Container',
               'ore-oil', 'Fishing')


@DATASETS.register_module()
class RSDDDataset(COCORBoxDataset):
    CLASSES = ('ship',)


@DATASETS.register_module()
class HRSIDDataset(COCORBoxDataset):
    CLASSES = ('ship',)
