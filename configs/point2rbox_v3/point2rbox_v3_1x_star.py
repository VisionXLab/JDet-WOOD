# Point2RBox-v3 端到端 STAR（Jittor）。⚠️ 不在验收范围，未跑训练。
# 逐行对照官方 point2rbox_v3-1x-star.py：与 dotav1-0 的全部差异（diff 实测 24 行）=
#   数据集 base → star、_base_sam-star、num_classes=48、
#   voronoi_thres 只留 default（override/square_cls/edge_loss_cls/post_process
#   在官方被注释 → head 默认空值）。其余逐值同 DOTA 版。

import os as _os

_here = _os.path.dirname(_os.path.abspath(__file__))
_ns = dict(__file__=_os.path.join(_here, 'point2rbox_v3_1x_dota.py'))
with open(_ns['__file__']) as _f:
    exec(_f.read(), _ns)

# _base_sam-star.py（逐字节锁定件）
_sam_ns = {}
with open(_os.path.join(_here, '_base_sam-star.py')) as _f:
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
head['num_classes'] = 48
head['voronoi_thres'] = dict(default=[0.994, 0.005])   # 官方注释掉 override
head['square_cls'] = []                                # 官方注释掉 → head 默认
head['edge_loss_cls'] = []
head['post_process'] = {}
head['loss_voronoi']['mask_filter_config'] = _sam_ns['mask_filter_config']
head['loss_voronoi']['sam_instance_thr'] = _sam_ns['sam_instance_thr']
head['loss_voronoi']['sam_sample_rules'] = _sam_ns['sam_sample_rules']

# 官方 star base：data/split_ss_star/{train,val,test}（数据未下载，路径按官方结构）
for _split, _sub in (('train', 'train'), ('val', 'val')):
    dataset[_split]['type'] = 'STARDataset'
    dataset[_split]['images_dir'] = f'/root/data/split_ss_star/{_sub}/images'
    dataset[_split]['annfiles_dir'] = f'/root/data/split_ss_star/{_sub}/annfiles'
    dataset[_split].pop('version', None)
dataset['test']['images_dir'] = '/root/data/split_ss_star/test/images'
dataset['test']['dataset_type'] = 'STAR'

del _ns, _sam_ns, _f, _here, _os, head, _split, _sub
