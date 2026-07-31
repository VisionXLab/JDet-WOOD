"""Use a trained Point2RBox-v3 checkpoint to export pseudo RBox labels.

Reference semantics：
    model.bbox_head.pseudo_generator=True（推理期用 GT 点位取预测框）
    数据 = trainval + 点标注（ConvertWeakSupervision），去 RandomFlip
    产物 = COCO 风格 <prefix>.bbox.json，字段与 mmrotate
           DOTAMetric.results2json 逐字段一致：
           {image_id: str patch 名（无扩展名）, bbox: [cx,cy,w,h,rad] le90,
            score: float, category_id: int 0-based}
    消费端 = P2RV2DOTADataset(ann_json=...)，语义对齐 mmrotate
             DOTADataset 的 .json 分支，用于 stage-2 rotated-FCOS 训练。

The JSON writer matches mmrotate `DOTAMetric.results2json` field-for-field.

用法：
    conda activate p2r-jittor
    CUDA_VISIBLE_DEVICES=0 python tools/export_pseudo_labels.py \
        --config-file configs/point2rbox_v3/point2rbox_v3_pseudo_generator_dota.py \
        --ckpt work_dirs/point2rbox_v3_1x_dota/checkpoints/ckpt_12.pkl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

import numpy as np


def load_cfg(path):
    ns = dict(__file__=os.path.abspath(path))
    with open(path) as f:
        exec(compile(f.read(), path, 'exec'), ns)
    return ns


def write_bbox_json(results_per_img, outfile_prefix):
    """与 mmrotate DOTAMetric.results2json 逐字段一致（tests/test_pseudo_export.py
    以 ref 实跑输出为 golden 锁定）。"""
    bbox_json_results = []
    for img_id in sorted(results_per_img.keys()):
        r = results_per_img[img_id]
        for i in range(len(r['labels'])):
            data = dict()
            data['image_id'] = img_id
            data['bbox'] = [float(v) for v in r['bboxes'][i]]
            data['score'] = float(r['scores'][i])
            data['category_id'] = int(r['labels'][i])
            bbox_json_results.append(data)

    out_path = outfile_prefix + '.bbox.json'
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(bbox_json_results, f)
    return out_path, len(bbox_json_results)


def print_stats(results_per_img):
    """Print pseudo-label aspect-ratio and angle statistics."""
    boxes = [r['bboxes'] for r in results_per_img.values() if len(r['bboxes'])]
    if not boxes:
        print('no pseudo boxes generated')
        return
    all_boxes = np.concatenate(boxes)
    wh = all_boxes[:, 2:4]
    ar = np.maximum(wh[:, 0], wh[:, 1]) \
        / np.maximum(np.minimum(wh[:, 0], wh[:, 1]), 1e-3)
    print('aspect ratio: mean=%.3f median=%.3f' % (ar.mean(), np.median(ar)))
    hist, _ = np.histogram(all_boxes[:, 4], bins=8,
                           range=(-np.pi / 2, np.pi / 2))
    print('angle hist (8 bins):', hist.tolist())


def run_export(cfg, ckpt_path=None, out_prefix=None, max_batches=None):
    import jittor as jt
    from jdet.utils.registry import build_from_cfg, MODELS, DATASETS

    model = build_from_cfg(cfg['model'], MODELS)
    assert model.bbox_head.pseudo_generator, \
        'config 必须设置 bbox_head.pseudo_generator=True（官方语义）'
    if ckpt_path is not None:
        ckpt = jt.load(ckpt_path)
        model.load_parameters(ckpt.get('model', ckpt))
    model.eval()

    dataset = build_from_cfg(cfg['dataset']['test'], DATASETS)
    outfile_prefix = out_prefix or cfg['evaluator']['outfile_prefix']

    results_per_img = {}
    n_batches = 0
    for images, targets in dataset:
        images = images if isinstance(images, jt.Var) else jt.array(images)
        feat = model.backbone(images)
        if model.neck:
            feat = model.neck(feat)
        results = model.bbox_head.predict(feat, targets)
        for target, r in zip(targets, results):
            fname = target['filename']
            fname = fname if isinstance(fname, str) else fname[0]
            img_id = os.path.splitext(os.path.basename(fname))[0]
            results_per_img[img_id] = dict(
                bboxes=r['bboxes'].numpy(),
                scores=r['scores'].numpy(),
                labels=r['labels'].numpy())
        n_batches += 1
        if n_batches % 200 == 0:
            print(f'{n_batches} batches done, {len(results_per_img)} images')
        if max_batches is not None and n_batches >= max_batches:
            break

    out_path, n = write_bbox_json(results_per_img, outfile_prefix)
    print(f'saved {n} instances ({len(results_per_img)} images) -> {out_path}')
    print_stats(results_per_img)
    return out_path, results_per_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', default=None, help='覆盖 outfile_prefix')
    args = parser.parse_args()

    import jittor as jt
    jt.flags.use_cuda = 1

    cfg = load_cfg(args.config_file)
    run_export(cfg, ckpt_path=args.ckpt, out_prefix=args.out)


if __name__ == '__main__':
    main()
