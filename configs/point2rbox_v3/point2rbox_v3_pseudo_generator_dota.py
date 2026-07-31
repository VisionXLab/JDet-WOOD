# Point2RBox-v3 伪标签生成（Jittor）
# 逐行对照官方 point2rbox_v3-pseudo-generator-dotav1-0.py（继承端到端 config）。
# 官方语义：bbox_head.pseudo_generator=True（推理期允许用 GT 点）；
# test 数据流 = train 数据（trainval + 点标注）但去掉 RandomFlip；
# test_evaluator=DOTAMetric(metric='mAP', format_only=True,
#   outfile_prefix='data/split_ss_dota/point2rbox_v3_pseudo_labels')
# → 产物 point2rbox_v3_pseudo_labels.bbox.json（COCO 风格）。
# L0 golden：config_point2rbox_v3-pseudo-generator-dotav1-0.json（614 键）。

import os as _os

_base_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                           'point2rbox_v3_1x_dota.py')
_ns = dict(__file__=_base_path)
with open(_base_path) as _f:
    exec(_f.read(), _ns)

model = _ns['model']
optimizer = _ns['optimizer']
scheduler = _ns['scheduler']
logger = _ns['logger']
max_epoch = _ns['max_epoch']
eval_interval = _ns['eval_interval']
checkpoint_interval = _ns['checkpoint_interval']
log_interval = _ns['log_interval']
preprocess_mean = _ns['preprocess_mean']
preprocess_std = _ns['preprocess_std']

# This allows the model to use gt points during inference（官方注释语义）
model['bbox_head']['pseudo_generator'] = True

dataset = _ns['dataset']
# 官方：test_dataloader = train_dataloader（_delete_）+ 去 RandomFlip 的 pipeline
dataset['test'] = dict(
    type='P2RV2DOTADataset',
    images_dir='/root/data/split_ss_dota/trainval/images',
    annfiles_dir='/root/data/split_ss_dota/trainval/annfiles',
    version='1',
    point_proportion=1.0,           # test_pipeline 保留 ConvertWeakSupervision
    hbox_proportion=0.0,
    weak_supervision=True,
    filter_empty_gt=True,           # 继承 train dataset 的 filter（官方 _base_ 语义）
    transforms=[
        # 官方 test_pipeline = train - RandomFlip（关增广）
        dict(type='MMRotateResize', min_size=1024, max_size=1024),
        dict(type='Pad', size_divisor=32),
        dict(type='Normalize', mean=preprocess_mean, std=preprocess_std,
             to_bgr=False),
    ],
    batch_size=2,                   # 官方 test_dataloader=train_dataloader → bs=2
    num_workers=0,                  # E10：jittor 多进程 dataloader 死锁（官方=2，infra 豁免）
    shuffle=False)

# 官方 test_evaluator：DOTAMetric(metric='mAP', format_only=True, outfile_prefix=...)
# JDet 等价物已由 tools/export_pseudo_labels.py 实现：该工具读取这里的
# outfile_prefix，并生成与 ref DOTAMetric.results2json 逐字段一致的 bbox.json。
# type 保留为声明性元数据，不经 JDet Runner 的 evaluator registry 构造。
evaluator = dict(
    type='PseudoLabelExporter',
    metric='mAP',
    format_only=True,
    outfile_prefix='/root/data/split_ss_dota/point2rbox_v3_pseudo_labels')

del _ns, _f, _base_path, _os
