# DIOR stage-2 rotated-FCOS（Jittor）。⚠️ 不在验收范围，未跑训练。
# 官方 diff vs dota 版：数据集 base → dior、num_classes=20、
# Resize (800,800)、train 数据 = DIOR trainval + 伪标签 json
# （官方 ann_file='point2rbox_v3_pseudo_labels.bbox.json'，data_root=dior）。

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

model['roi_heads']['num_classes'] = 20

_DIOR_ROOT = '/root/data/dior'
for _split in list(dataset.keys()):
    _d = dataset[_split]
    # Resize 1024 → 800（官方 dior）
    for _t in _d.get('transforms', []):
        if _t.get('type') == 'MMRotateResize':
            _t['min_size'] = 800
            _t['max_size'] = 800
    if 'images_dir' in _d:
        _d['images_dir'] = _DIOR_ROOT + (
            '/JPEGImages-test' if _split == 'test' else '/JPEGImages-trainval')
    if 'annfiles_dir' in _d:
        _d['annfiles_dir'] = _DIOR_ROOT + '/Annotations/Oriented Bounding Boxes'
    if _d.get('type') == 'P2RV2DOTADataset':
        _d['type'] = 'DIORDataset'
        _d.pop('version', None)
    if 'ann_json' in _d:
        _d['ann_json'] = _DIOR_ROOT + '/point2rbox_v3_pseudo_labels.bbox.json'

del _ns, _f, _here, _os, _split, _d, _t
