"""generate_pseudo_labels.py — stage-1.5：用训好的 v2 checkpoint 生成伪 RBox 标签。

对齐官方 point2rbox_v2-pseudo-generator-dota.py 语义（PLAN §6.2）：
    model.bbox_head.pseudo_generator=True（推理时用 GT 点位取预测框）
    数据 = trainval + ConvertWeakSupervision（去 RandomFlip）
    产物 = COCO 风格 <prefix>.bbox.json，字段与 mmrotate DOTAMetric.results2json
           逐字段一致：{image_id, bbox[cx,cy,w,h,a], score, category_id}

用法：
    conda activate p2r-jittor
    CUDA_VISIBLE_DEVICES=0 python tools/generate_pseudo_labels.py \
        --config configs/point2rbox_v2/point2rbox_v2_pseudo_generator_dota.py \
        --ckpt work_dirs/point2rbox_v2_1x_dota/checkpoints/ckpt_12.pkl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import jittor as jt


def load_cfg(path):
    ns = {}
    with open(path) as f:
        exec(compile(f.read(), path, 'exec'), ns)
    return ns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', default=None, help='覆盖 outfile_prefix')
    args = parser.parse_args()

    jt.flags.use_cuda = 1

    cfg = load_cfg(args.config)
    base = load_cfg(cfg['_base_cfg'])
    outfile_prefix = args.out or cfg['outfile_prefix']

    from jdet.utils.registry import build_from_cfg, MODELS, DATASETS

    model_cfg = base['model']
    model_cfg['bbox_head'] = dict(model_cfg['bbox_head'],
                                  pseudo_generator=cfg['pseudo_generator'])
    model = build_from_cfg(model_cfg, MODELS)

    ckpt = jt.load(args.ckpt)
    sd = ckpt.get('model', ckpt)
    model.load_parameters(sd)
    model.eval()

    dataset = build_from_cfg(cfg['pseudo_dataset'], DATASETS)

    results_per_img = {}
    n_batches = 0
    for images, targets in dataset:
        images = jt.array(images) if not isinstance(images, jt.Var) else images
        feat = model.backbone(images)
        if model.neck:
            feat = model.neck(feat)
        # pseudo_generator=True → predict_by_feat 走 pseudo 分支（GT 点位取框）
        results = model.bbox_head.predict(feat, targets)
        for target, r in zip(targets, results):
            img_id = os.path.splitext(
                target['filename'] if isinstance(target['filename'], str)
                else target['filename'][0])[0]
            results_per_img[img_id] = dict(
                bboxes=r['bboxes'].numpy(),
                scores=r['scores'].numpy(),
                labels=r['labels'].numpy())
        n_batches += 1
        if n_batches % 200 == 0:
            print(f'{n_batches} batches done, {len(results_per_img)} images')

    # 与 mmrotate DOTAMetric.results2json 逐字段一致
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
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(bbox_json_results, f)
    print(f'saved {len(bbox_json_results)} instances '
          f'({len(results_per_img)} images) -> {out_path}')

    # 伪标签统计（M7 验收：宽高比/角度直方图 vs PyTorch 版对照）
    all_boxes = np.concatenate([r['bboxes'] for r in results_per_img.values()])
    wh = all_boxes[:, 2:4]
    ar = np.maximum(wh[:, 0], wh[:, 1]) / np.maximum(np.minimum(wh[:, 0], wh[:, 1]), 1e-3)
    print('aspect ratio: mean=%.3f median=%.3f' % (ar.mean(), np.median(ar)))
    hist, _ = np.histogram(all_boxes[:, 4], bins=8, range=(-np.pi / 2, np.pi / 2))
    print('angle hist (8 bins):', hist.tolist())


if __name__ == '__main__':
    main()
