# STAR stage-2 rotated-FCOS（Jittor）。⚠️ 不在验收范围，未跑训练。
# 官方 diff vs dota 版仅：数据集 base → star、num_classes=48。

import os as _os

_here = _os.path.dirname(_os.path.abspath(__file__))
_ns = dict(__file__=_os.path.join(_here, 'rotated_fcos_1x_dota_using_pseudo.py'))
with open(_ns['__file__']) as _f:
    exec(_f.read(), _ns)

model = _ns['model']
dataset = _ns['dataset']
optimizer = _ns['optimizer']
scheduler = _ns['scheduler']
logger = _ns['logger']
max_epoch = _ns['max_epoch']
eval_interval = _ns['eval_interval']
checkpoint_interval = _ns['checkpoint_interval']
log_interval = _ns['log_interval']

model['roi_heads']['num_classes'] = 48

for _split in list(dataset.keys()):
    _d = dataset[_split]
    for _k in ('images_dir', 'annfiles_dir'):
        if _k in _d:
            _d[_k] = _d[_k].replace('split_ss_dota', 'split_ss_star') \
                           .replace('/trainval/', '/train/')
    if _d.get('type') == 'P2RV2DOTADataset':
        _d['type'] = 'STARDataset'
        _d.pop('version', None)
    if 'ann_json' in _d:
        _d['ann_json'] = '/root/data/star/point2rbox_v3_pseudo_labels.bbox.json'

del _ns, _f, _here, _os, _split, _d, _k
