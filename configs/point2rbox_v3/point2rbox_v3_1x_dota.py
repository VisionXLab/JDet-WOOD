# Point2RBox-v3 端到端 DOTA-v1.0（Jittor）
# 逐行对照官方 /root/ref/Point2RBox-v3/configs/point2rbox_v3/point2rbox_v3-1x-dotav1-0.py
# 及其 _base_ 链（../_base_/datasets/dota.py + schedules/schedule_1x.py +
# default_runtime.py + _base_sam-dotav1-0.py）。
# 铁律一：任何数值/开关/顺序不得偏离；类名映射见 docs/config_parity.md。
# L0 golden：tests/parity/golden/config_point2rbox_v3-1x-dotav1-0.json（616 键）。

import os as _os

# ---- _base_sam-dotav1-0.py：逐字节锁定件，以 exec 引入其三个变量 ----
# （sam_instance_thr / mask_filter_config / sam_sample_rules，铁律二 #7/#8/#9）
_sam_ns = {}
with open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                        '_base_sam-dotav1-0.py')) as _f:
    exec(_f.read(), _sam_ns)
sam_instance_thr = _sam_ns['sam_instance_thr']
mask_filter_config = _sam_ns['mask_filter_config']
sam_sample_rules = _sam_ns['sam_sample_rules']
del _sam_ns, _f

angle_version = 'le90'

# 官方顶层开关（use_sam=False 时按官方语义清空，照抄该行为）
use_sam = True
use_class_specific_watershed = False
if use_sam == False:  # noqa: E712  官方写法照抄
    mask_filter_config = None
    sam_instance_thr = -1

# data_preprocessor（mean/std/bgr_to_rgb/pad_size_divisor）由 dataset transforms
# 实现；mean/std 同时传 model 供 TED 反归一化（映射表，与 v2 config 同款）
preprocess_mean = [123.675, 116.28, 103.53]
preprocess_std = [58.395, 57.12, 57.375]

model = dict(
    type='Point2RBoxV3',
    # 官方 config 显式项
    ss_prob=[0.68, 0.07, 0.25],                     # 铁律二 #12（覆盖默认 [0.6,0.15,0.25]）
    copy_paste_start_epoch=6,
    label_assign_pseudo_label_switch_eopch=6,       # 铁律二 #4：eopch 拼写照抄
    # 官方未覆盖的 detector 默认值（ref point2rbox_v3.py L127-140）
    rotate_range=(0.25, 0.75),
    scale_range=(0.5, 0.9),
    num_copies=10,
    data_preprocessor=dict(
        mean=preprocess_mean,
        std=preprocess_std,
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),                      # 铁律二 #12：显式照抄
    backbone=dict(
        type='Resnet50',
        # 官方 out_indices=(0,1,2,3) → layer1..layer4（铁律二 #11：v3 多尺度，与 v2 各是各的）
        return_stages=['layer1', 'layer2', 'layer3', 'layer4'],
        frozen_stages=1,
        norm_eval=True,
        pretrained=True),                           # init torchvision://resnet50
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),              # 铁律二 #12
    bbox_head=dict(
        type='Point2RBoxV3Head',
        num_classes=15,
        in_channels=256,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],               # 铁律二 #11：v3 五尺度
        use_adaptive_scale=False,
        edge_loss_start_epoch=6,
        joint_angle_start_epoch=1,
        voronoi_type='standard',                    # 官方覆盖 head 默认 gaussian-orientation
        voronoi_thres=dict(
            default=[0.994, 0.005],
            override=(([2, 11], [0.999, 0.6]),
                      ([7, 8, 10, 14], [0.95, 0.005]))),  # 铁律二 #10：tuple 结构照抄
        square_cls=[1, 9, 11],
        edge_loss_cls=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13],
        post_process={11: 1.2},
        angle_coder=dict(
            type='PSCCoder',
            angle_version='le90',
            dual_freq=False,
            num_step=3,
            thr_mod=0),
        loss_cls=dict(
            type='MMDetFocalLoss',   # 官方 mmdet.FocalLoss（底座 FocalLoss 是 1-based，语义不同）
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0),
        loss_overlap=dict(
            type='GaussianOverlapLoss', loss_weight=10.0, lamb=0),
        loss_voronoi=dict(
            type='VoronoiWatershedLoss', loss_weight=5.0,
            mask_filter_config=mask_filter_config,  # 铁律二 #7/#8：_base_sam 原值直传
            sam_instance_thr=sam_instance_thr,
            sam_sample_rules=sam_sample_rules,
            use_class_specific_watershed=use_class_specific_watershed),
        loss_bbox_edg=dict(type='EdgeLoss', loss_weight=0.3),
        loss_ss=dict(type='Point2RBoxV2ConsistencyLoss', loss_weight=1.0),
        test_cfg=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms_rotated', iou_threshold=0.1),
            max_per_img=2000)),
    train_cfg=None)

# 官方 train_pipeline（顺序照抄）：
#   LoadImageFromFile → LoadAnnotations(qbox) → ConvertBoxType(rbox)
#   → ConvertWeakSupervision(point=1., hbox=0) → Resize((1024,1024), keep_ratio)
#   → RandomFlip(0.75, [h,v,diag]) → PackDetInputs
# 前四步由 P2RV2DOTADataset 实现（映射表；point_dummy=1 官方默认）
dataset = dict(
    train=dict(
        type='P2RV2DOTADataset',
        images_dir='/root/data/split_ss_dota/trainval/images',
        annfiles_dir='/root/data/split_ss_dota/trainval/annfiles',
        version='1',
        point_proportion=1.0,
        hbox_proportion=0.0,
        weak_supervision=True,
        filter_empty_gt=True,                       # 铁律二 #12
        transforms=[
            dict(type='MMRotateResize', min_size=1024, max_size=1024),
            dict(type='MMRotateRandomFlip', prob=0.75,
                 direction=['horizontal', 'vertical', 'diagonal']),
            dict(type='Pad', size_divisor=32),
            dict(type='Normalize', mean=preprocess_mean, std=preprocess_std,
                 to_bgr=False),
        ],
        batch_size=2,                               # 官方总 batch=2（单卡，PLAN §9.1）
        num_workers=0,  # 官方=2（torch DataLoader）；jittor 多进程 dataset 有环形缓冲死锁（A commit 3d87c60），infra 层映射为 0，不影响训练数学
        shuffle=True),
    val=dict(
        type='P2RV2DOTADataset',
        # 铁律二 #5：官方 val 指向 trainval/，照抄（非干净验证集）
        images_dir='/root/data/split_ss_dota/trainval/images',
        annfiles_dir='/root/data/split_ss_dota/trainval/annfiles',
        version='1',
        weak_supervision=False,
        filter_empty_gt=False,
        transforms=[
            dict(type='MMRotateResize', min_size=1024, max_size=1024),
            dict(type='Pad', size_divisor=32),
            dict(type='Normalize', mean=preprocess_mean, std=preprocess_std,
                 to_bgr=False),
        ],
        batch_size=16,                              # 官方 val batch_size=16
        num_workers=0,  # 官方=2（torch DataLoader）；jittor 多进程 dataset 有环形缓冲死锁（A commit 3d87c60），infra 层映射为 0，不影响训练数学
        shuffle=False),
    test=dict(
        type='ImageDataset',
        dataset_type='DOTA1',
        images_dir='/root/data/split_ss_dota/test/images',
        transforms=[
            dict(type='MMRotateResize', min_size=1024, max_size=1024),
            dict(type='Pad', size_divisor=32),
            dict(type='Normalize', mean=preprocess_mean, std=preprocess_std,
                 to_bgr=False),
        ],
        batch_size=4,                               # 官方 test batch_size=4
        num_workers=0,  # 官方=2（torch DataLoader）；jittor 多进程 dataset 有环形缓冲死锁（A commit 3d87c60），infra 层映射为 0，不影响训练数学
        shuffle=False))

# 官方 optim_wrapper：AdamW(_delete_=True) 但 clip_grad 仍生效（铁律二 #1）
optimizer = dict(
    type='AdamW',
    lr=0.00005,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    grad_clip=dict(max_norm=35, norm_type=2))

# 官方 param_scheduler（铁律二 #2）
scheduler = dict(
    type='LinearWarmupMultiStepLR',
    start_factor=1.0 / 3,
    warmup_iters=500,
    milestones=[8, 11],
    gamma=0.1)

logger = dict(type='RunLogger')

# 官方 train_cfg=EpochBasedTrainLoop(max_epochs=12, val_interval=12)
# custom_hooks=[mmdet.SetEpochInfoHook] → Runner 内建 model.set_epoch（C3，映射表）
max_epoch = 12
eval_interval = 12
checkpoint_interval = 1
log_interval = 50
