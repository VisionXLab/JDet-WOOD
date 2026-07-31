"""DIOR 数据集（铁律三第 2 层）。

格式（以 ref mmrotate/datasets/dior.py 为准）：ann_file 是图像 id 列表 txt
（ImageSets/Main/*.txt），旋转框标注在 VOC 风格 XML 的 <robndbox>（8 点 poly）。
data.md 写的「VOC-style txt」指的是 id 列表文件，标注本体是 XML。
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

from jdet.utils.registry import DATASETS
from .dota_txt_variant import DOTATxtVariantDataset
from .p2rv2_dota import poly2rbox_le90_np


@DATASETS.register_module()
class DIORDataset(DOTATxtVariantDataset):
    """20 类。annfiles_dir 传 XML 目录，imgset_file 传 id 列表 txt。"""

    CLASSES = (
        'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
        'chimney', 'expressway-service-area', 'expressway-toll-station',
        'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
        'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
        'windmill')
    IMG_SUFFIX = 'jpg'

    def __init__(self, imgset_file=None, **kwargs):
        self.imgset_file = imgset_file
        super().__init__(**kwargs)

    def _load_annfiles(self):
        self.CLASSES = list(type(self).CLASSES)
        self.cls2idx = {c: i for i, c in enumerate(self.CLASSES)}
        if self.imgset_file:
            # 官方 dior e2e 是 train.txt + val.txt 两个 imgset 的 ConcatDataset，
            # 这里支持传 list 一次性合并（顺序保持官方 train 在前）
            files = (self.imgset_file if isinstance(self.imgset_file, (list, tuple))
                     else [self.imgset_file])
            img_ids = []
            for fp in files:
                with open(fp) as f:
                    img_ids += [l.strip() for l in f if l.strip()]
        else:
            img_ids = [f[:-4] for f in sorted(os.listdir(self.annfiles_dir))
                       if f.endswith('.xml')]
        img_infos = []
        for img_id in img_ids:
            xml_path = os.path.join(self.annfiles_dir, img_id + '.xml')
            polys, labels = [], []
            if os.path.exists(xml_path):
                root = ET.parse(xml_path).getroot()
                for obj in root.findall('object'):
                    cls = obj.find('name').text.lower()
                    if cls not in self.cls2idx:
                        continue
                    b = obj.find('robndbox')
                    if b is None:
                        continue
                    polys.append([
                        float(b.find('x_left_top').text),
                        float(b.find('y_left_top').text),
                        float(b.find('x_right_top').text),
                        float(b.find('y_right_top').text),
                        float(b.find('x_right_bottom').text),
                        float(b.find('y_right_bottom').text),
                        float(b.find('x_left_bottom').text),
                        float(b.find('y_left_bottom').text)])
                    labels.append(self.cls2idx[cls])
            polys = np.array(polys, dtype=np.float32).reshape(-1, 8)
            img_infos.append(dict(
                filename=img_id + '.' + self._img_suffix,
                ann=dict(bboxes=poly2rbox_le90_np(polys),
                         labels=np.array(labels, dtype=np.int32),
                         bboxes_ignore=np.zeros((0, 5), dtype=np.float32),
                         labels_ignore=np.zeros((0,), dtype=np.int32))))
        return img_infos
