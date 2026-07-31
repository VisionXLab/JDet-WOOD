"""Pseudo-label exporter schema, consumer and smoke tests (CPU).

    conda activate p2r-jittor && python tests/test_pseudo_export.py

1. schema parity：同一组假 results，ref DOTAMetric.results2json（p2r-torch 子进程
   实跑）与本仓库 write_bbox_json 的输出**逐条逐字段完全相等**。
2. 消费端：P2RV2DOTADataset._load_json 能读回，形状/类别/数值正确。
3. 端到端冒烟：随机权重完整 v3 模型 + 2 张真实 trainval patch（CPU），
   run_export 产出合法 json 且每实例 5 元 rbox。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))

TORCH_PY = '/opt/miniconda3/envs/p2r-torch/bin/python'
REF = '/root/ref/Point2RBox-v3'

FAKE = [
    dict(img_id='P0001__1024__0___0',
         labels=np.array([0, 14], dtype=np.int64),
         bboxes=np.array([[100.5, 200.25, 30.0, 12.5, 0.3],
                          [500.0, 400.0, 80.0, 79.0, -1.2]], np.float32),
         scores=np.array([1.0, 0.5], np.float32)),
    dict(img_id='P0002__1024__824___0',
         labels=np.array([7], dtype=np.int64),
         bboxes=np.array([[10.0, 20.0, 5.0, 4.0, 1.5]], np.float32),
         scores=np.array([0.25], np.float32)),
]


def ref_golden(tmp):
    """p2r-torch 子进程里用 ref 的 DOTAMetric.results2json 生成 golden。"""
    data_path = os.path.join(tmp, 'fake.npz')
    np.savez(data_path, **{
        f'{i}_{k}': v for i, r in enumerate(FAKE) for k, v in r.items()
        if k != 'img_id'})
    ids = [r['img_id'] for r in FAKE]
    script = f'''
import sys, numpy as np
sys.path.insert(0, {REF!r})
from mmrotate.evaluation.metrics.dota_metric import DOTAMetric
d = np.load({data_path!r})
ids = {ids!r}
results = [dict(img_id=ids[i], labels=d[f"{{i}}_labels"],
                bboxes=d[f"{{i}}_bboxes"], scores=d[f"{{i}}_scores"])
           for i in range(len(ids))]
m = DOTAMetric.__new__(DOTAMetric)
m.results2json(results, {os.path.join(tmp, 'golden')!r})
'''
    r = subprocess.run([TORCH_PY, '-c', script], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=REF))
    assert r.returncode == 0, r.stderr
    return os.path.join(tmp, 'golden.bbox.json')


def main():
    from export_pseudo_labels import write_bbox_json, load_cfg, run_export

    tmp = tempfile.mkdtemp(prefix='pseudo_export_')
    try:
        # --- 1. schema parity（FAKE 已按 img_id 升序，两侧顺序语义一致）---
        golden_path = ref_golden(tmp)
        golden = json.load(open(golden_path))
        mine_path, n = write_bbox_json(
            {r['img_id']: r for r in FAKE}, os.path.join(tmp, 'mine'))
        mine = json.load(open(mine_path))
        assert n == len(golden) == 3
        assert mine == golden, (
            f'schema mismatch:\nmine   = {mine}\ngolden = {golden}')
        keys = set(mine[0])
        assert keys == {'image_id', 'bbox', 'score', 'category_id'}, keys
        print(f'[OK ] schema parity: {len(mine)} entries field-identical to '
              f'ref results2json (keys={sorted(keys)})')

        # --- 2. Dataset consumer round trip ---
        from jdet.data.p2rv2_dota import P2RV2DOTADataset
        ds = object.__new__(P2RV2DOTADataset)
        ds.ann_json = mine_path
        infos = P2RV2DOTADataset._load_json(ds)
        assert len(infos) == 2
        assert infos[0]['ann']['bboxes'].shape == (2, 5)
        assert infos[0]['ann']['labels'].tolist() == [0, 14]
        assert np.allclose(infos[0]['ann']['bboxes'][0],
                           [100.5, 200.25, 30.0, 12.5, 0.3])
        print('[OK ] consumer: P2RV2DOTADataset._load_json round-trip')

        # --- 3. 端到端冒烟（CPU、随机权重、2 张真实 patch）---
        src_img = '/root/data/split_ss_dota/trainval/images'
        src_ann = '/root/data/split_ss_dota/trainval/annfiles'
        img_dir = os.path.join(tmp, 'images')
        ann_dir = os.path.join(tmp, 'annfiles')
        os.makedirs(img_dir), os.makedirs(ann_dir)
        picked = 0
        for f in sorted(os.listdir(src_ann)):
            p = os.path.join(src_ann, f)
            if os.path.getsize(p) > 0:
                os.symlink(p, os.path.join(ann_dir, f))
                os.symlink(os.path.join(src_img, f[:-4] + '.png'),
                           os.path.join(img_dir, f[:-4] + '.png'))
                picked += 1
                if picked == 2:
                    break
        assert picked == 2

        cfg = load_cfg(os.path.join(
            ROOT, 'configs/point2rbox_v3/point2rbox_v3_pseudo_generator_dota.py'))
        cfg['dataset']['test']['images_dir'] = img_dir
        cfg['dataset']['test']['annfiles_dir'] = ann_dir
        out_path, per_img = run_export(
            cfg, ckpt_path=None, out_prefix=os.path.join(tmp, 'e2e'),
            max_batches=1)
        out = json.load(open(out_path))
        assert len(per_img) >= 1
        n_gt = sum(len(r['labels']) for r in per_img.values())
        assert len(out) == n_gt and n_gt > 0, (len(out), n_gt)
        for entry in out:
            assert len(entry['bbox']) == 5
            assert 0 <= entry['category_id'] <= 14
        print(f'[OK ] e2e smoke: {len(per_img)} imgs, {n_gt} pseudo boxes, '
              f'json valid')

        print('PASS: pseudo label exporter (schema/consumer/e2e)')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
