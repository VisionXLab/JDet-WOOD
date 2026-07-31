# DIOR 伪标签生成（Jittor）。⚠️ 不在验收范围。
# 官方 diff vs dota 版：_base_ → 1x-dior、Resize (800,800)（随 dior e2e 继承）、
# test pipeline 对 Concat 两个子集同时生效（loader 内合并已覆盖）、
# outfile_prefix → data/dior/...

import os as _os

_here = _os.path.dirname(_os.path.abspath(__file__))
_base_ns = dict(__file__=_os.path.join(_here, 'point2rbox_v3_1x_dior.py'))
with open(_base_ns['__file__']) as _f:
    exec(_f.read(), _base_ns)

model = _base_ns['model']
optimizer = _base_ns['optimizer']
scheduler = _base_ns['scheduler']
logger = _base_ns['logger']
max_epoch = _base_ns['max_epoch']
eval_interval = _base_ns['eval_interval']
checkpoint_interval = _base_ns['checkpoint_interval']
log_interval = _base_ns['log_interval']

model['bbox_head']['pseudo_generator'] = True

dataset = _base_ns['dataset']
_train = dataset['train']
dataset['test'] = dict(_train)
dataset['test']['transforms'] = [t for t in _train['transforms']
                                 if t['type'] != 'MMRotateRandomFlip']
dataset['test']['shuffle'] = False

evaluator = dict(
    type='PseudoLabelExporter',
    metric='mAP',
    format_only=True,
    outfile_prefix='/root/data/dior/point2rbox_v3_pseudo_labels')

del _base_ns, _f, _here, _os, _train
