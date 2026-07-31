# Point2RBox-v3 端到端 DIOR（Jittor）。⚠️ 不在验收范围，未跑训练。
# 逐行对照官方 point2rbox_v3-1x-dior.py。与 dotav1-0 的差异：
#   use_sam=False（官方语义：mask_filter_config=None, sam_instance_thr=-1,
#     sam_sample_rules=None）；label_assign=True（官方顶层旗标，config 内无消费者，
#     照抄保留）；num_classes=20；voronoi_thres default=[0.995,0.005] +
#     override=(([1,2,3,16],[0.96,0.005]),)；square_cls=[2,5,9,14,15,19]；
#     edge_loss_cls=1..19；post_process={15:1.1, 19:1.1}；
#     use_class_specific_watershed=True；Resize (800,800)；
#     train 数据 = train.txt+val.txt 两个 imgset 的 Concat（loader 内合并）。
#   官方 dior 配置未写 label_assign_pseudo_label_switch_eopch → 走 detector
#     默认值 6（ref L137），故此处从 model 里删掉显式项、语义等价。

import os as _os

_here = _os.path.dirname(_os.path.abspath(__file__))
_ns = dict(__file__=_os.path.join(_here, 'point2rbox_v3_1x_dota.py'))
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
preprocess_mean = _ns['preprocess_mean']
preprocess_std = _ns['preprocess_std']

# 官方顶层旗标（照抄；label_assign 在官方 config 内无消费者）
use_sam = False
label_assign = True

model.pop('label_assign_pseudo_label_switch_eopch')  # 官方 dior 未写 → 默认 6

head = model['bbox_head']
head['num_classes'] = 20
head['voronoi_thres'] = dict(
    default=[0.995, 0.005],
    override=(([1, 2, 3, 16], [0.96, 0.005]),))
head['square_cls'] = [2, 5, 9, 14, 15, 19]
head['edge_loss_cls'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                         16, 17, 18, 19]
head['post_process'] = {15: 1.1, 19: 1.1}
# use_sam=False：官方把三个 SAM 变量清空
head['loss_voronoi']['mask_filter_config'] = None
head['loss_voronoi']['sam_instance_thr'] = -1
head['loss_voronoi']['sam_sample_rules'] = None
head['loss_voronoi']['use_class_specific_watershed'] = True

_DIOR_ROOT = '/root/data/dior'
_dior_transforms_train = [
    dict(type='MMRotateResize', min_size=800, max_size=800),  # 官方 (800,800)
    dict(type='MMRotateRandomFlip', prob=0.75,
         direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='Pad', size_divisor=32),
    dict(type='Normalize', mean=preprocess_mean, std=preprocess_std,
         to_bgr=False),
]
_dior_transforms_eval = [t for t in _dior_transforms_train
                         if t['type'] != 'MMRotateRandomFlip']

dataset['train'] = dict(
    type='DIORDataset',
    images_dir=_DIOR_ROOT + '/JPEGImages-trainval',
    annfiles_dir=_DIOR_ROOT + '/Annotations/Oriented Bounding Boxes',
    imgset_file=[_DIOR_ROOT + '/ImageSets/Main/train.txt',
                 _DIOR_ROOT + '/ImageSets/Main/val.txt'],  # 官方 Concat
    point_proportion=1.0,
    hbox_proportion=0.0,
    weak_supervision=True,
    filter_empty_gt=True,
    transforms=list(_dior_transforms_train),
    batch_size=2,
    num_workers=0,   # E10 infra 豁免
    shuffle=True)
dataset['val'] = dict(
    type='DIORDataset',
    images_dir=_DIOR_ROOT + '/JPEGImages-trainval',
    annfiles_dir=_DIOR_ROOT + '/Annotations/Oriented Bounding Boxes',
    imgset_file=[_DIOR_ROOT + '/ImageSets/Main/train.txt',
                 _DIOR_ROOT + '/ImageSets/Main/val.txt'],
    weak_supervision=False,
    filter_empty_gt=False,
    transforms=list(_dior_transforms_eval),
    batch_size=16,
    num_workers=0,
    shuffle=False)
dataset['test'] = dict(
    type='ImageDataset',
    images_dir=_DIOR_ROOT + '/JPEGImages-test',
    transforms=list(_dior_transforms_eval),
    batch_size=4,
    num_workers=0,
    shuffle=False)

del _ns, _f, _here, _os, head
