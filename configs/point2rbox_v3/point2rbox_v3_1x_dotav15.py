# Point2RBox-v3 端到端 DOTA-v1.5（Jittor）。⚠️ 不在验收范围，未跑训练。
# 逐行对照官方 point2rbox_v3-1x-dotav1-5.py：与 dotav1-0 的差异（diff 实测）=
#   数据集 base → dotav15、_base_sam-dotav1-5、num_classes=16。其余逐值相同。

import os as _os

_here = _os.path.dirname(_os.path.abspath(__file__))
_ns = dict(__file__=_os.path.join(_here, 'point2rbox_v3_1x_dota.py'))
with open(_ns['__file__']) as _f:
    exec(_f.read(), _ns)

_sam_ns = {}
with open(_os.path.join(_here, '_base_sam-dotav1-5.py')) as _f:
    exec(_f.read(), _sam_ns)

model = _ns['model']
dataset = _ns['dataset']
optimizer = _ns['optimizer']
scheduler = _ns['scheduler']
logger = _ns['logger']
max_epoch = _ns['max_epoch']
eval_interval = _ns['eval_interval']
checkpoint_interval = _ns['checkpoint_interval']
log_interval = _ns['log_interval']
preprocess_mean = _ns['preprocess_mean']
preprocess_std = _ns['preprocess_std']

head = model['bbox_head']
head['num_classes'] = 16
head['loss_voronoi']['mask_filter_config'] = _sam_ns['mask_filter_config']
head['loss_voronoi']['sam_instance_thr'] = _sam_ns['sam_instance_thr']
head['loss_voronoi']['sam_sample_rules'] = _sam_ns['sam_sample_rules']

# 官方 dotav15 base：data/split_ss_dota1_5/，类别表 DOTA1_5（jdet 常量与 ref 一致）
for _split in ('train', 'val'):
    dataset[_split]['version'] = '1_5'
    dataset[_split]['images_dir'] = \
        dataset[_split]['images_dir'].replace('split_ss_dota', 'split_ss_dota1_5')
    dataset[_split]['annfiles_dir'] = \
        dataset[_split]['annfiles_dir'].replace('split_ss_dota', 'split_ss_dota1_5')
dataset['test']['images_dir'] = '/root/data/split_ss_dota1_5/test/images'
dataset['test']['dataset_type'] = 'DOTA1_5'

del _ns, _sam_ns, _f, _here, _os, head, _split
