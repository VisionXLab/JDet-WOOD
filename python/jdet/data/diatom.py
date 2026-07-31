"""DIATOM 数据集（铁律三第 2 层）。

格式（ref mmrotate/datasets/diatom.py）：id 列表 + XML（objects/object，
hbb: bbox/xmin..ymax）→ 水平框转 rbox(angle=0)。单类 'diatom'。
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

from jdet.utils.registry import DATASETS
from .dota_txt_variant import DOTATxtVariantDataset


@DATASETS.register_module()
class DIATOMDataset(DOTATxtVariantDataset):
    CLASSES = ('diatom',)
    IMG_SUFFIX = 'jpg'

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
                objs = root.find('objects')
                for obj in (objs.findall('object') if objs is not None else []):
                    bb = obj.find('bbox')
                    xmin = float(bb.find('xmin').text)
                    ymin = float(bb.find('ymin').text)
                    xmax = float(bb.find('xmax').text)
                    ymax = float(bb.find('ymax').text)
                    rboxes.append([(xmin + xmax) / 2, (ymin + ymax) / 2,
                                   xmax - xmin, ymax - ymin, 0.0])
                    labels.append(0)
            rboxes = np.array(rboxes, dtype=np.float32).reshape(-1, 5)
            img_infos.append(dict(
                filename=img_id + '.' + self._img_suffix,
                ann=dict(bboxes=rboxes,
                         labels=np.array(labels, dtype=np.int32),
                         bboxes_ignore=np.zeros((0, 5), dtype=np.float32),
                         labels_ignore=np.zeros((0,), dtype=np.int32))))
        return img_infos
