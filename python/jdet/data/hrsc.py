"""HRSC2016 数据集（铁律三第 2 层）。

格式（ref mmrotate/datasets/hrsc.py）：id 列表 txt + XML，
旋转框直接给 (mbox_cx, mbox_cy, mbox_w, mbox_h, mbox_ang)（弧度）。
classwise=False（默认）单类 'ship'，与 ref 一致。
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

from jdet.utils.registry import DATASETS
from .dota_txt_variant import DOTATxtVariantDataset


@DATASETS.register_module()
class HRSCDataset(DOTATxtVariantDataset):
    CLASSES = ('ship',)
    IMG_SUFFIX = 'bmp'

    def __init__(self, imgset_file=None, **kwargs):
        self.imgset_file = imgset_file
        super().__init__(**kwargs)

    def _load_annfiles(self):
        self.CLASSES = list(type(self).CLASSES)
        self.cls2idx = {c: i for i, c in enumerate(self.CLASSES)}
        if self.imgset_file:
            with open(self.imgset_file) as f:
                img_ids = [l.strip() for l in f if l.strip()]
        else:
            img_ids = [f[:-4] for f in sorted(os.listdir(self.annfiles_dir))
                       if f.endswith('.xml')]
        img_infos = []
        for img_id in img_ids:
            xml_path = os.path.join(self.annfiles_dir, img_id + '.xml')
            rboxes, labels = [], []
            if os.path.exists(xml_path):
                root = ET.parse(xml_path).getroot()
                for obj in root.findall('HRSC_Objects/HRSC_Object'):
                    rboxes.append([
                        float(obj.find('mbox_cx').text),
                        float(obj.find('mbox_cy').text),
                        float(obj.find('mbox_w').text),
                        float(obj.find('mbox_h').text),
                        float(obj.find('mbox_ang').text)])
                    labels.append(0)  # classwise=False：单类 ship（ref 默认）
            rboxes = np.array(rboxes, dtype=np.float32).reshape(-1, 5)
            img_infos.append(dict(
                filename=img_id + '.' + self._img_suffix,
                ann=dict(bboxes=rboxes,
                         labels=np.array(labels, dtype=np.int32),
                         bboxes_ignore=np.zeros((0, 5), dtype=np.float32),
                         labels_ignore=np.zeros((0,), dtype=np.int32))))
        return img_infos
