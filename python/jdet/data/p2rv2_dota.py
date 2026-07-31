"""Point2RBox-v2 的 DOTA 数据集：直读 mmrotate 风格 split_ss_dota（txt annfiles）。

对齐官方 pipeline（configs/point2rbox_v2/point2rbox_v2-1x-dota.py，PLAN §6.1）：
    LoadImageFromFile → LoadAnnotations(qbox) → ConvertBoxType(qbox→rbox)
    → ConvertWeakSupervision(point_proportion=1., hbox_proportion=0)
    → Resize((1024,1024), keep_ratio) → RandomFlip(0.75, [h,v,diag]) → Pack

要点：
- qbox→rbox 逐框 cv2.minAreaRect（mmrotate qbox2rbox 原样，角度不 norm）
- ConvertWeakSupervision 官方默认 point_dummy=1（⚠️ 底座 whollywood_dota 用 0.1，不复用）
- RandomFlip prob=0.75 方向三选一（mmdet 语义：每方向 0.25）
- 标注可能是 CRLF 行尾（B 校验报告 2026-07-26 12:18）
"""
import os

import cv2
import numpy as np
from PIL import Image

from jdet.utils.registry import DATASETS
from jdet.config.constant import get_classes_by_name
from jdet.models.boxes.box_ops import rotated_box_to_bbox_np
from .custom import CustomDataset


def poly2rbox_le90_np(polys):
    """(N, 8) qbox → (N, 5) rbox。mmrotate qbox2rbox：cv2.minAreaRect，弧度。"""
    rboxes = []
    for pts in polys.reshape(-1, 4, 2):
        (x, y), (w, h), angle = cv2.minAreaRect(pts.astype(np.float32))
        rboxes.append([x, y, w, h, angle / 180 * np.pi])
    if not rboxes:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array(rboxes, dtype=np.float32)


@DATASETS.register_module()
class P2RV2DOTADataset(CustomDataset):
    """读 split_ss_dota/{images,annfiles} 的 DOTA 数据集（弱监督点标注）。"""

    def __init__(self,
                 images_dir=None,
                 annfiles_dir=None,
                 version='1',
                 point_proportion=1.0,
                 hbox_proportion=0.0,
                 point_dummy=1.0,
                 hbox_dummy=0.0,
                 weak_supervision=True,
                 ann_json=None,
                 transforms=None,
                 batch_size=1,
                 num_workers=0,
                 shuffle=False,
                 drop_last=False,
                 filter_empty_gt=True,
                 diff_thr=100,
                 buffer_size=512 * 1024 * 1024):
        # 不走 CustomDataset.__init__（其从 labels.pkl 加载），直接初始化 jt Dataset
        from jittor.dataset import Dataset as JtDataset
        JtDataset.__init__(self, batch_size=batch_size, num_workers=num_workers,
                           shuffle=shuffle, drop_last=drop_last,
                           buffer_size=buffer_size)
        from .transforms import Compose
        # 子类（mm_datasets.py）以类属性形式给出 CLASSES；未给出时按 DOTA 版本取
        if getattr(self, 'CLASSES', None) is None:
            self.CLASSES = get_classes_by_name('DOTA' + version)
        self.cls2idx = {c: i for i, c in enumerate(self.CLASSES)}
        self.images_dir = os.path.abspath(images_dir)
        self.annfiles_dir = os.path.abspath(annfiles_dir)
        self.point_proportion = point_proportion
        self.hbox_proportion = hbox_proportion
        self.point_dummy = point_dummy
        self.hbox_dummy = hbox_dummy
        self.weak_supervision = weak_supervision
        self.ann_json = ann_json
        self.diff_thr = diff_thr
        self.transforms = Compose(transforms) if transforms is not None else None

        self.img_infos = (self._load_json() if ann_json is not None
                          else self._load_annfiles())
        if filter_empty_gt:
            self.img_infos = [i for i in self.img_infos
                              if len(i['ann']['bboxes']) > 0]
        self.total_len = len(self.img_infos)

    def _load_annfiles(self):
        img_infos = []
        for fname in sorted(os.listdir(self.annfiles_dir)):
            if not fname.endswith('.txt'):
                continue
            polys, labels = [], []
            with open(os.path.join(self.annfiles_dir, fname)) as f:
                for line in f:
                    parts = line.strip().split()  # strip 兼容 CRLF
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
                filename=fname[:-4] + getattr(self, 'IMG_SUFFIX', '.png'),
                ann=dict(bboxes=rboxes,
                         labels=np.array(labels, dtype=np.int32),
                         bboxes_ignore=np.zeros((0, 5), dtype=np.float32),
                         labels_ignore=np.zeros((0,), dtype=np.int32))))
        return img_infos

    def _load_json(self):
        """读 stage-1.5 产出的 COCO 风格 *.bbox.json（mmrotate DOTADataset json 分支语义）。"""
        import json
        with open(self.ann_json) as f:
            root = json.load(f)
        instances = {}
        for item in root:
            instances.setdefault(item['image_id'], []).append(item)
        img_infos = []
        for img_id in sorted(instances.keys()):
            items = instances[img_id]
            rboxes = np.array([it['bbox'] for it in items], dtype=np.float32).reshape(-1, 5)
            labels = np.array([it['category_id'] for it in items], dtype=np.int32)
            img_infos.append(dict(
                filename=img_id + '.png',
                ann=dict(bboxes=rboxes, labels=labels,
                         bboxes_ignore=np.zeros((0, 5), dtype=np.float32),
                         labels_ignore=np.zeros((0,), dtype=np.int32))))
        return img_infos

    def _convert_weak_supervision(self, bboxes):
        """官方 ConvertWeakSupervision（transforms.py L86），numpy 版。"""
        bboxes = bboxes.copy()
        n = bboxes.shape[0]
        max_idx_p = int(round(n * self.point_proportion))
        bboxes[:max_idx_p, 2] = self.point_dummy
        bboxes[:max_idx_p, 3] = self.point_dummy
        bboxes[:max_idx_p, 4] = 0
        max_idx_h = max_idx_p + int(round(n * self.hbox_proportion))
        if max_idx_h > max_idx_p:
            seg = bboxes[max_idx_p:max_idx_h]
            _, polys = rotated_box_to_bbox_np(seg)
            xmin = polys[:, ::2].min(1)
            ymin = polys[:, 1::2].min(1)
            xmax = polys[:, ::2].max(1)
            ymax = polys[:, 1::2].max(1)
            bboxes[max_idx_p:max_idx_h] = np.stack(
                [(xmin + xmax) / 2, (ymin + ymax) / 2,
                 xmax - xmin, ymax - ymin,
                 np.full_like(xmin, self.hbox_dummy)], 1)
        return bboxes

    def _read_ann_info(self, idx):
        # Empty samples must stay empty at validation/test time.  mmrotate
        # keeps them when ``test_mode=True`` even if filter_empty_gt is set;
        # replacing one with a random non-empty image silently duplicates GT
        # and corrupts AP.  Training samples have already been filtered in
        # __init__, so no retry loop is needed there either.
        img_info = self.img_infos[idx]
        anno = img_info['ann']

        img_path = os.path.join(self.images_dir, img_info['filename'])
        if not os.path.exists(img_path):
            # RSAR 等数据集图像扩展名不统一（ref 按 glob 实际文件取），逐一回退
            stem = os.path.splitext(img_info['filename'])[0]
            for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif'):
                cand = os.path.join(self.images_dir, stem + ext)
                if os.path.exists(cand):
                    img_path = cand
                    break
        image = Image.open(img_path).convert('RGB')
        width, height = image.size

        rboxes = anno['bboxes'].astype(np.float32)
        if self.weak_supervision:
            rboxes = self._convert_weak_supervision(rboxes)

        hboxes, polys = rotated_box_to_bbox_np(rboxes)
        ann = dict(
            rboxes=rboxes,
            hboxes=hboxes.astype(np.float32),
            polys=polys.astype(np.float32),
            labels=anno['labels'].astype(np.int32),
            rboxes_ignore=anno['bboxes_ignore'].astype(np.float32),
            hboxes_ignore=np.zeros((0, 4), dtype=np.float32),
            polys_ignore=np.zeros((0, 8), dtype=np.float32),
            classes=self.CLASSES,
            ori_img_size=(width, height),
            img_size=(width, height),
            scale_factor=1.0,
            filename=img_info['filename'],
            img_file=img_path)
        return image, ann

    def evaluate(self, results, work_dir, epoch, logger=None, save=True):
        """DOTA VOC-style mAP（本地趋势用；最终精度以官方 test 提交为准）。"""
        # 本 loader/head 严格采用 mmdet/mmrotate 的 0-based labels；JDet 底座
        # DOTADataset.evaluate 则是历史 1-based 约定（它会给 det_labels +1，
        # 并以 i+1 筛 GT）。在适配边界复制并平移 GT，不能污染保存的 results，
        # 也不能把 1-based 语义渗回训练路径。
        import copy
        import os
        import jittor as jt
        from jdet.utils.general import check_dir
        from jdet.data.dota import DOTADataset
        # Persist the canonical 0-based representation.  Passing save=True to
        # the legacy evaluator would save the temporary +1 label adapter and
        # a later offline evaluation would shift it a second time.
        if save:
            save_path = os.path.join(work_dir, f'detections/val_{epoch}')
            check_dir(save_path)
            jt.save(results, os.path.join(save_path, 'val.pkl'))
        eval_results = []
        for result, target in results:
            target = copy.deepcopy(target)
            target['labels'] = target['labels'] + 1
            eval_results.append((result, target))
        return DOTADataset.evaluate(self, eval_results, work_dir, epoch,
                                    logger=logger, save=False)


from jdet.utils.registry import TRANSFORMS
import random as _random
from PIL import Image as _PILImage


@TRANSFORMS.register_module()
class MMRotateRandomFlip:
    """mmdet RandomFlip(prob, direction=list) 的 JDet 移植（mmrotate rbox 翻转语义）。

    与底座 RotatedRandomFlip 的差异（按 mmrotate 为准）：
    - 坐标翻转是 x' = W - x（无 -1 像素偏移）
    - 角度：horizontal/vertical → a' = -a；diagonal → 角度不变
    - direction 为列表时按 mmdet 语义：flip 概率 prob，方向在列表中均匀选
    """

    def __init__(self, prob=0.5, direction='horizontal'):
        self.prob = prob
        self.directions = [direction] if isinstance(direction, str) else list(direction)

    def _flip_rboxes(self, bboxes, w, h, direction):
        flipped = bboxes.copy()
        if direction == 'horizontal':
            flipped[..., 0] = w - flipped[..., 0]
            flipped[..., 4] = -flipped[..., 4]
        elif direction == 'vertical':
            flipped[..., 1] = h - flipped[..., 1]
            flipped[..., 4] = -flipped[..., 4]
        elif direction == 'diagonal':
            flipped[..., 0] = w - flipped[..., 0]
            flipped[..., 1] = h - flipped[..., 1]
        return flipped

    def __call__(self, image, target=None):
        if _random.random() >= self.prob:
            return image, target
        direction = self.directions[_random.randint(0, len(self.directions) - 1)]
        if direction == 'horizontal':
            image = image.transpose(_PILImage.FLIP_LEFT_RIGHT)
        elif direction == 'vertical':
            image = image.transpose(_PILImage.FLIP_TOP_BOTTOM)
        elif direction == 'diagonal':
            image = image.transpose(_PILImage.FLIP_LEFT_RIGHT)
            image = image.transpose(_PILImage.FLIP_TOP_BOTTOM)
        if target is not None:
            w, h = target['img_size']
            for key in ['rboxes', 'rboxes_ignore']:
                if key in target and len(target[key]):
                    target[key] = self._flip_rboxes(target[key], w, h, direction)
            # hboxes/polys 由 rboxes 重算
            if 'rboxes' in target:
                hb, pl = rotated_box_to_bbox_np(target['rboxes'])
                target['hboxes'], target['polys'] = hb.astype(np.float32), pl.astype(np.float32)
            target['flip'] = direction
        return image, target


@TRANSFORMS.register_module()
class MMRotateResize:
    """mmdet Resize(scale, keep_ratio=True) 的 rbox 语义：xy/wh 乘 scale、角度不变。

    底座 RotatedResize 会经 poly 往返重新规范角度（点框 0 → -π/2），而 head 的
    伪标签路径要求点标注角度恒 0（上游 L297 注释），故不复用。
    """

    def __init__(self, min_size, max_size, keep_ratio=True):
        self.min_size = min_size
        self.max_size = max_size
        self.keep_ratio = keep_ratio

    def __call__(self, image, target=None):
        w, h = image.size
        scale = min(self.max_size / max(w, h), self.min_size / min(w, h))
        nw, nh = int(w * scale + 0.5), int(h * scale + 0.5)
        if (nw, nh) != (w, h):
            image = image.resize((nw, nh), _PILImage.BILINEAR)
        if target is not None:
            if scale != 1.0:
                for key in ['rboxes', 'rboxes_ignore']:
                    if key in target and len(target[key]):
                        b = target[key].copy()
                        b[..., :4] *= scale
                        target[key] = b
                if 'rboxes' in target and len(target['rboxes']):
                    hb, pl = rotated_box_to_bbox_np(target['rboxes'])
                    target['hboxes'] = hb.astype(np.float32)
                    target['polys'] = pl.astype(np.float32)
            target['img_size'] = (nw, nh)
            target['scale_factor'] = scale
            target['pad_shape'] = (nh, nw)
            target['keep_ratio'] = self.keep_ratio
        return image, target
