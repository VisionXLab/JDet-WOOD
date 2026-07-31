# STAR 伪标签生成（Jittor）。⚠️ 不在验收范围。
# 官方 diff vs dota 版仅：_base_ → 1x-star、outfile_prefix → data/star/...

import os as _os

_here = _os.path.dirname(_os.path.abspath(__file__))
_ns = dict(__file__=_os.path.join(_here, 'point2rbox_v3_pseudo_generator_dota.py'))
# 复用 dota 伪标签 config 的全部逻辑，再切到 star 的 e2e 基座与数据
_base_ns = dict(__file__=_os.path.join(_here, 'point2rbox_v3_1x_star.py'))
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
dataset['test'] = dict(_train)   # 官方 test_dataloader=train_dataloader
dataset['test']['transforms'] = [t for t in _train['transforms']
                                 if t['type'] != 'MMRotateRandomFlip']  # 去增广
dataset['test']['shuffle'] = False

evaluator = dict(
    type='PseudoLabelExporter',
    metric='mAP',
    format_only=True,
    outfile_prefix='/root/data/star/point2rbox_v3_pseudo_labels')  # 官方路径

del _ns, _base_ns, _f, _here, _os, _train
