"""DOTA-txt 格式系数据集的通用变体基类（铁律三第 2 层）。

复用 P2RV2DOTADataset 的全部机制（弱监督转换 / transforms / collate），只参数化
类别表与图像后缀。派生数据集（STAR / RSAR / OCDPCB / …）每个仅需给出 CLASSES。

类别表以 /root/ref/Point2RBox-v3/mmrotate/datasets/*.py 的 METAINFO 为唯一真相源，
tests/test_datasets_import.py 用 ast 解析 ref 源码逐字符比对，防止漂移。
"""
import os

import numpy as np

from jdet.utils.registry import DATASETS
from .p2rv2_dota import P2RV2DOTADataset, poly2rbox_le90_np


class DOTATxtVariantDataset(P2RV2DOTADataset):
    """子类置 CLASSES（tuple）与 IMG_SUFFIX；其余与 P2RV2DOTADataset 一致。"""

    CLASSES = None          # 子类必须覆盖
    IMG_SUFFIX = 'png'      # ref 各数据集的 img_suffix 默认值

    def __init__(self, img_suffix=None, **kwargs):
        assert type(self).CLASSES, f'{type(self).__name__}.CLASSES 未定义'
        self._img_suffix = img_suffix or self.IMG_SUFFIX
        # P2RV2DOTADataset.__init__ 会用 get_classes_by_name 覆盖 self.CLASSES，
        # 我们在其后恢复子类类别表并重建索引，再重新加载标注。
        kwargs.setdefault('version', '1')
        super().__init__(**kwargs)

    # P2RV2 __init__ 内部依次调 _load_annfiles；通过覆盖它并在首次调用前
    # 把类别表切到子类定义，保证解析用对表。
    def _load_annfiles(self):
        self.CLASSES = list(type(self).CLASSES)
        self.cls2idx = {c: i for i, c in enumerate(self.CLASSES)}
        img_infos = []
        for fname in sorted(os.listdir(self.annfiles_dir)):
            if not fname.endswith('.txt'):
                continue
            polys, labels = [], []
            with open(os.path.join(self.annfiles_dir, fname)) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue
                    cls_name = parts[8]
                    if cls_name not in self.cls2idx:
                        continue
                    difficulty = int(parts[9]) if len(parts) >= 10 else 0
                    if difficulty > self.diff_thr:
                        continue
                    polys.append([float(v) for v in parts[:8]])
                    labels.append(self.cls2idx[cls_name])
            polys = np.array(polys, dtype=np.float32).reshape(-1, 8)
            rboxes = poly2rbox_le90_np(polys)
            img_infos.append(dict(
                filename=fname[:-4] + '.' + self._img_suffix,
                ann=dict(bboxes=rboxes,
                         labels=np.array(labels, dtype=np.int32),
                         bboxes_ignore=np.zeros((0, 5), dtype=np.float32),
                         labels_ignore=np.zeros((0,), dtype=np.int32))))
        return img_infos
