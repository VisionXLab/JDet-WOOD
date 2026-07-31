"""铁律三第 2 层验收：数据集 loader 注册/类别表/解析 + 非 DOTA config 完整性。

运行（CPU 即可）：conda activate p2r-jittor && python tests/test_datasets_import.py

覆盖：
1) 每个新 loader import + DATASETS 注册 + 合成标注实例化 + bboxes/labels shape
2) 类别表与 ref（/root/ref/Point2RBox-v3）逐字符一致（ast 解析 ref 源码为 golden）
3) DOTA v1/1.5/2 类别表：jdet 常量 == ref METAINFO（故 v1.5/v2 无需新 loader）
4) 3 个 cp 的 _base_sam-* 与 ref 逐字节一致（md5）
5) 7 个非 DOTA method config 可被 exec 解析（数值语义不在此验，未跑训练）
"""
import ast
import hashlib
import json
import os
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'python'))
REF = '/root/ref/Point2RBox-v3'


def ref_metainfo(path, cls_name=None):
    tree = ast.parse(open(path).read())
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.Assign) and \
                        getattr(item.targets[0], 'id', '') == 'METAINFO':
                    out[node.name] = ast.literal_eval(item.value)['classes']
    return out if cls_name is None else out[cls_name]


def make_img(path, size=(64, 64)):
    Image.fromarray(np.zeros((size[1], size[0], 3), np.uint8)).save(path)


def main():
    from jdet.utils.registry import DATASETS
    import jdet.data  # noqa: F401  触发注册
    from jdet.config.constant import (DOTA1_CLASSES, DOTA1_5_CLASSES,
                                      DOTA2_CLASSES)
    # STAR/RSAR/OCDPCB are provided by the shared mm_datasets module.
    from jdet.data.mm_datasets import (STARDataset, RSARDataset,
                                       OCDPCBDataset)
    from jdet.data.dior import DIORDataset
    from jdet.data.hrsc import HRSCDataset
    from jdet.data.diatom import DIATOMDataset
    from jdet.data.sku110k import SKU110KDataset
    from jdet.data.coco_rbox import (SARDet100kDataset, SRSDDDataset,
                                     RSDDDataset, HRSIDDataset)

    # ---- 2/3) 类别表 golden ----
    checks = [
        (STARDataset, ref_metainfo(f'{REF}/mmrotate/datasets/star.py',
                                   'STARDataset')),
        (RSARDataset, ref_metainfo(f'{REF}/mmrotate/datasets/rsar.py',
                                   'RSARDataset')),
        (OCDPCBDataset, ref_metainfo(f'{REF}/mmrotate/datasets/ocdpcb.py',
                                     'OCDPCBDataset')),
        (DIORDataset, ref_metainfo(f'{REF}/mmrotate/datasets/dior.py',
                                   'DIORDataset')),
        (DIATOMDataset, ref_metainfo(f'{REF}/mmrotate/datasets/diatom.py',
                                     'DIATOMDataset')),
        (SARDet100kDataset,
         ref_metainfo(f'{REF}/mmrotate/datasets/sardet100k.py',
                      'SAR_Det_Finegrained_Dataset')),
    ]
    for cls, golden in checks:
        assert tuple(cls.CLASSES) == tuple(golden), \
            f'{cls.__name__} 类别表与 ref 不一致'
        print(f'[OK ] {cls.__name__}: {len(cls.CLASSES)} classes == ref')

    # hrsc：ref classwise=False 语义 = 单类 ship（31 类表仅 classwise=True 用）
    assert HRSCDataset.CLASSES == ('ship',)
    # srsdd/rsdd/hrsid：golden 是 ref 配置文件里的 metainfo.classes
    for name, cls in (('srsdd', SRSDDDataset), ('rsdd', RSDDDataset),
                      ('hrsid', HRSIDDataset)):
        src = open(f'{REF}/configs/_base_/datasets/{name}.py').read()
        tree = ast.parse(src)
        golden = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and \
                    getattr(node.targets[0], 'id', '') == 'metainfo':
                # ref 写法是 dict(classes=(...))（Call 节点），取 keyword 求值
                if isinstance(node.value, ast.Call):
                    kw = {k.arg: ast.literal_eval(k.value)
                          for k in node.value.keywords}
                    golden = kw['classes']
                else:
                    golden = ast.literal_eval(node.value)['classes']
        assert tuple(cls.CLASSES) == tuple(golden), name
        print(f'[OK ] {cls.__name__}: classes == ref config metainfo')

    assert tuple(DOTA1_CLASSES) == ref_metainfo(
        f'{REF}/mmrotate/datasets/dota.py', 'DOTADataset')
    assert tuple(DOTA1_5_CLASSES) == ref_metainfo(
        f'{REF}/mmrotate/datasets/dota.py', 'DOTAv15Dataset')
    assert tuple(DOTA2_CLASSES) == ref_metainfo(
        f'{REF}/mmrotate/datasets/dota.py', 'DOTAv2Dataset')
    print('[OK ] DOTA v1/v1.5/v2 类别表 == ref（v1.5/v2 复用 P2RV2DOTADataset(version=...)）')

    # ---- 1) 合成标注实例化 ----
    tmp = tempfile.mkdtemp(prefix='ds_test_')
    img_d = os.path.join(tmp, 'images'); os.makedirs(img_d)
    ann_d = os.path.join(tmp, 'annfiles'); os.makedirs(ann_d)

    # DOTA-txt 系
    with open(os.path.join(ann_d, 'I0.txt'), 'w') as f:
        f.write('10 10 30 10 30 20 10 20 ship 0\n'
                '5 5 9 5 9 9 5 9 aircraft 1\n')
    make_img(os.path.join(img_d, 'I0.png'))
    ds = RSARDataset(images_dir=img_d, annfiles_dir=ann_d,
                     filter_empty_gt=True, batch_size=1)
    info = ds.img_infos[0]['ann']
    assert info['bboxes'].shape == (2, 5) and info['labels'].tolist() == [0, 1]
    assert DATASETS.get('RSARDataset') is RSARDataset
    print(f'[OK ] RSARDataset: 合成 DOTA-txt 解析 bboxes{info["bboxes"].shape} labels ok')

    # DIOR XML
    xml = ('<annotation><object><name>ship</name><robndbox>'
           + ''.join(f'<{k}>{v}</{k}>' for k, v in [
               ('x_left_top', 10), ('y_left_top', 10), ('x_right_top', 30),
               ('y_right_top', 10), ('x_right_bottom', 30),
               ('y_right_bottom', 20), ('x_left_bottom', 10),
               ('y_left_bottom', 20)])
           + '</robndbox></object></annotation>')
    with open(os.path.join(ann_d, 'D0.xml'), 'w') as f:
        f.write(xml)
    with open(os.path.join(tmp, 'ids.txt'), 'w') as f:
        f.write('D0\n')
    make_img(os.path.join(img_d, 'D0.jpg'))
    ds = DIORDataset(images_dir=img_d, annfiles_dir=ann_d,
                     imgset_file=os.path.join(tmp, 'ids.txt'),
                     filter_empty_gt=True, batch_size=1)
    b = ds.img_infos[0]['ann']['bboxes']
    # minAreaRect 的 w/h 顺序随角度约定可交换，比较无序集合
    assert b.shape == (1, 5) and abs(b[0][0] - 20) < 1 and \
        sorted([round(float(b[0][2])), round(float(b[0][3]))]) == [10, 20]
    print(f'[OK ] DIORDataset: XML robndbox → rbox {b[0][:4].round(1).tolist()}')

    # HRSC XML
    with open(os.path.join(ann_d, 'H0.xml'), 'w') as f:
        f.write('<HRSC_Image><HRSC_Objects><HRSC_Object>'
                '<mbox_cx>20</mbox_cx><mbox_cy>15</mbox_cy><mbox_w>20</mbox_w>'
                '<mbox_h>10</mbox_h><mbox_ang>0.3</mbox_ang>'
                '</HRSC_Object></HRSC_Objects></HRSC_Image>')
    make_img(os.path.join(img_d, 'H0.bmp'))
    ds = HRSCDataset(images_dir=img_d, annfiles_dir=ann_d,
                     imgset_file=None, filter_empty_gt=True, batch_size=1)
    hb = [i for i in ds.img_infos if i['filename'] == 'H0.bmp'][0]['ann']
    assert hb['bboxes'].shape == (1, 5) and abs(hb['bboxes'][0][4] - 0.3) < 1e-6
    print('[OK ] HRSCDataset: XML mbox → rbox (含弧度角)')

    # DIATOM XML
    with open(os.path.join(ann_d, 'T0.xml'), 'w') as f:
        f.write('<annotation><objects><object><bbox>'
                '<xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>20</ymax>'
                '</bbox></object></objects></annotation>')
    make_img(os.path.join(img_d, 'T0.jpg'))
    ds = DIATOMDataset(images_dir=img_d, annfiles_dir=ann_d,
                       filter_empty_gt=True, batch_size=1)
    db = [i for i in ds.img_infos if i['filename'] == 'T0.jpg'][0]['ann']
    assert db['bboxes'].shape == (1, 5) and db['bboxes'][0][4] == 0
    print('[OK ] DIATOMDataset: XML hbb → rbox(angle=0)')

    # SKU110K json
    sku = [dict(image_id='S0', rbbox=[20, 15, 20, 10, 0.5])]
    with open(os.path.join(tmp, 'sku.json'), 'w') as f:
        json.dump(sku, f)
    make_img(os.path.join(img_d, 'S0.jpg'))
    ds = SKU110KDataset(images_dir=img_d,
                        ann_json_file=os.path.join(tmp, 'sku.json'),
                        filter_empty_gt=True, batch_size=1)
    assert ds.img_infos[0]['ann']['bboxes'].shape == (1, 5)
    print('[OK ] SKU110KDataset: json rbbox 解析')

    # COCO json（SRSDD 代表 COCO 系）
    coco = dict(
        images=[dict(id=1, file_name='C0.jpg', width=64, height=64)],
        annotations=[dict(id=1, image_id=1, category_id=7,
                          bbox=[10, 10, 20, 10],
                          segmentation=[[10, 10, 30, 10, 30, 20, 10, 20]])],
        categories=[dict(id=7, name='Dredger')])
    with open(os.path.join(tmp, 'coco.json'), 'w') as f:
        json.dump(coco, f)
    make_img(os.path.join(img_d, 'C0.jpg'))
    ds = SRSDDDataset(images_dir=img_d,
                      ann_json_file=os.path.join(tmp, 'coco.json'),
                      filter_empty_gt=True, batch_size=1)
    cb = ds.img_infos[0]['ann']
    assert cb['bboxes'].shape == (1, 5) and cb['labels'][0] == 1  # Dredger=idx1
    print('[OK ] SRSDDDataset: COCO json（8 点 segmentation → rbox）')

    # ---- 4) _base_sam md5 ----
    for n in ('dior', 'star', 'dotav1-5'):
        mine = hashlib.md5(open(
            f'{ROOT}/configs/point2rbox_v3/_base_sam-{n}.py', 'rb').read()).hexdigest()
        ref = hashlib.md5(open(
            f'{REF}/configs/point2rbox_v3/_base_sam-{n}.py', 'rb').read()).hexdigest()
        assert mine == ref, n
    print('[OK ] _base_sam-{dior,star,dotav1-5} 逐字节 == ref (md5)')

    # ---- 5) 7 个非 DOTA config 可解析 ----
    cfgs = ['point2rbox_v3_1x_dior.py', 'point2rbox_v3_1x_star.py',
            'point2rbox_v3_1x_dotav15.py',
            'point2rbox_v3_pseudo_generator_dior.py',
            'point2rbox_v3_pseudo_generator_star.py',
            'rotated_fcos_1x_dior_using_pseudo.py',
            'rotated_fcos_1x_star_using_pseudo.py']
    for c in cfgs:
        path = os.path.join(ROOT, 'configs/point2rbox_v3', c)
        ns = dict(__file__=path)
        exec(open(path).read(), ns)
        assert 'model' in ns and 'dataset' in ns, c
    nc = {'dior': 20, 'star': 48, 'dotav15': 16}
    for key, n in nc.items():
        path = os.path.join(ROOT, f'configs/point2rbox_v3/point2rbox_v3_1x_{key}.py')
        ns = dict(__file__=path)
        exec(open(path).read(), ns)
        assert ns['model']['bbox_head']['num_classes'] == n, key
    print(f'[OK ] {len(cfgs)} 个非 DOTA config exec 解析通过（num_classes 20/48/16 断言）')

    print('PASS: tier-2 datasets + non-DOTA configs')


if __name__ == '__main__':
    main()
