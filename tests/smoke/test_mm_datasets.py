# mm_datasets.py（DOTA-txt 系 loaders）冒烟测试：
# 能 import、能注册、能实例化、能按各自 CLASSES 正确加载 txt 标注。
import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))


def _make_mini_dataset(root, cls_a, cls_b, suffix='.png'):
    """两张图：img1 含 cls_a×2 + 未知类×1 + 高 difficulty×1，img2 空标注。"""
    images = os.path.join(root, 'images')
    annfiles = os.path.join(root, 'annfiles')
    os.makedirs(images)
    os.makedirs(annfiles)
    Image.new('RGB', (64, 48)).save(os.path.join(images, 'img1' + suffix))
    Image.new('RGB', (64, 48)).save(os.path.join(images, 'img2' + suffix))
    with open(os.path.join(annfiles, 'img1.txt'), 'w') as f:
        f.write(f'10 10 30 10 30 20 10 20 {cls_a} 0\n')
        f.write(f'5 5 15 5 15 9 5 9 {cls_b} 0\n')
        f.write('1 1 2 1 2 2 1 2 not-a-real-class 0\n')     # 未知类 → 跳过
        f.write(f'40 30 60 30 60 40 40 40 {cls_a} 2\n')      # difficulty 2 > diff_thr=1 → 跳过
    open(os.path.join(annfiles, 'img2.txt'), 'w').close()    # 空 → filter_empty_gt 滤掉
    return images, annfiles


def _check_dataset(cls, suffix='.png'):
    cls_a, cls_b = cls.CLASSES[0], cls.CLASSES[-1]
    with tempfile.TemporaryDirectory() as root:
        images, annfiles = _make_mini_dataset(root, cls_a, cls_b, suffix)
        ds = cls(images_dir=images, annfiles_dir=annfiles,
                 weak_supervision=False, diff_thr=1,
                 transforms=None, batch_size=1)
        assert ds.total_len == 1, f'{cls.__name__}: 空图未被过滤或误滤'
        _, ann = ds._read_ann_info(0)
        assert ann['classes'] == cls.CLASSES
        assert list(ann['labels']) == [0, len(cls.CLASSES) - 1], \
            f'{cls.__name__}: 标签映射错 {ann["labels"]}'
        assert ann['rboxes'].shape == (2, 5)
        assert ann['ori_img_size'] == (64, 48)
    print(f'ok {cls.__name__} ({len(cls.CLASSES)} classes)')


def test_all():
    from jdet.data.mm_datasets import (DOTAv15Dataset, DOTAv2Dataset,
                                       STARDataset, RSARDataset,
                                       OCDPCBDataset)
    from jdet.utils.registry import DATASETS
    for cls in (DOTAv15Dataset, DOTAv2Dataset, STARDataset,
                RSARDataset, OCDPCBDataset):
        assert DATASETS.get(cls.__name__) is cls, f'{cls.__name__} 未注册'
        _check_dataset(cls)
    assert len(DOTAv15Dataset.CLASSES) == 16
    assert len(DOTAv2Dataset.CLASSES) == 18
    assert len(STARDataset.CLASSES) == 48
    assert len(RSARDataset.CLASSES) == 6
    assert len(OCDPCBDataset.CLASSES) == 41


def test_rsar_ext_fallback():
    """RSAR 图像扩展名不统一：filename 记 .png，实际文件是 .jpg 也能读。"""
    from jdet.data.mm_datasets import RSARDataset
    with tempfile.TemporaryDirectory() as root:
        images, annfiles = _make_mini_dataset(root, 'ship', 'harbor',
                                              suffix='.jpg')
        ds = RSARDataset(images_dir=images, annfiles_dir=annfiles,
                         weak_supervision=False, diff_thr=1,
                         transforms=None, batch_size=1)
        img, ann = ds._read_ann_info(0)
        assert img.size == (64, 48)
    print('ok RSAR ext fallback')


def test_base_dota_unaffected():
    """基类改动回归：默认 DOTA-v1.0 类表不受影响。"""
    from jdet.data.p2rv2_dota import P2RV2DOTADataset
    with tempfile.TemporaryDirectory() as root:
        images, annfiles = _make_mini_dataset(root, 'plane', 'helicopter')
        ds = P2RV2DOTADataset(images_dir=images, annfiles_dir=annfiles,
                              weak_supervision=False, diff_thr=1,
                              transforms=None, batch_size=1)
        assert len(ds.CLASSES) == 15 and ds.CLASSES[0] == 'plane'
        assert ds.total_len == 1
    print('ok base P2RV2DOTADataset regression')


def test_empty_sample_is_not_replaced():
    """Validation keeps an empty patch at its original index."""
    from jdet.data.p2rv2_dota import P2RV2DOTADataset
    with tempfile.TemporaryDirectory() as root:
        images, annfiles = _make_mini_dataset(root, 'plane', 'helicopter')
        ds = P2RV2DOTADataset(
            images_dir=images, annfiles_dir=annfiles,
            weak_supervision=False, diff_thr=1, filter_empty_gt=False,
            transforms=None, batch_size=1)
        assert ds.total_len == 2
        image, ann = ds._read_ann_info(1)
        assert image.size == (64, 48)
        assert ann['filename'] == 'img2.png'
        assert ann['rboxes'].shape == (0, 5)
        assert ann['labels'].shape == (0,)
    print('ok empty validation sample preserved')


if __name__ == '__main__':
    test_all()
    test_rsar_ext_fallback()
    test_base_dota_unaffected()
    test_empty_sample_is_not_replaced()
    print('ALL PASS')
