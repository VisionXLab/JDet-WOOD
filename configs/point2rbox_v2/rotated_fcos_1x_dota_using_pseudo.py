# Stage-2：rotated FCOS 以伪标签全监督训练（PLAN §6.3，逐行对照官方
# rotated-fcos-1x-dota-using-pseudo.py；铁律二 #5/#7 照抄）
preprocess_mean = [123.675, 116.28, 103.53]
preprocess_std = [58.395, 57.12, 57.375]

model = dict(
    type='FCOS',                       # 官方 model.type='mmdet.FCOS'
    backbone=dict(
        type='Resnet50',
        # 官方 out_indices=(0,1,2,3)（铁律二 #7：与 v2 端到端不同，各是各的）
        return_stages=['layer1', 'layer2', 'layer3', 'layer4'],
        frozen_stages=1,
        norm_eval=True,
        pretrained=True),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=512,              # 官方就是 512 通道
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    roi_heads=dict(
        type='RotatedFCOSHead',
        num_classes=15,
        in_channels=512,
        stacked_convs=4,
        feat_channels=512,
        strides=[8, 16, 32, 64, 128],
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512, 1e8)),
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        use_hbbox_loss=False,
        scale_angle=True,
        angle_version='le90',
        bbox_coder=dict(type='DistanceAnglePointCoder', angle_version='le90'),
        loss_cls=dict(
            type='MMDetFocalLoss',     # 官方 mmdet.FocalLoss
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_angle=None,
        loss_centerness=dict(
            type='MMDetCrossEntropyLoss',  # 官方 mmdet.CrossEntropyLoss(use_sigmoid=True)
            use_sigmoid=True,
            loss_weight=1.0),
        test_cfg=dict(                 # 官方与 v2 端到端相同
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms_rotated', iou_threshold=0.1),
            max_per_img=2000)))

# 官方 train_pipeline：LoadAnnotations(box_type='rbox')，无 ConvertWeakSupervision
dataset = dict(
    train=dict(
        type='P2RV2DOTADataset',
        images_dir='/root/data/split_ss_dota/trainval/images',
        annfiles_dir='/root/data/split_ss_dota/trainval/annfiles',
        # 官方 ann_file='point2rbox_v2_pseudo_labels.bbox.json'
        ann_json='/root/data/split_ss_dota/point2rbox_v2_pseudo_labels.bbox.json',
        version='1',
        weak_supervision=False,        # 全监督阶段
        filter_empty_gt=True,
        transforms=[
            dict(type='MMRotateResize', min_size=1024, max_size=1024),
            dict(type='MMRotateRandomFlip', prob=0.75,
                 direction=['horizontal', 'vertical', 'diagonal']),
            dict(type='Pad', size_divisor=32),
            dict(type='Normalize', mean=preprocess_mean, std=preprocess_std,
                 to_bgr=False),
        ],
        batch_size=4,                  # 官方 stage-2 batch_size=4
        num_workers=0,                 # 同 [plan-deviation]（jittor dataloader 死锁）
        shuffle=True),
    val=dict(
        type='P2RV2DOTADataset',
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
        batch_size=4,                  # 官方 stage-2 val batch_size=4
        num_workers=0,
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
        batch_size=4,
        num_workers=0,
        shuffle=False))

optimizer = dict(
    type='AdamW',
    lr=0.00005,
    betas=(0.9, 0.999),
    weight_decay=0.005,                # 铁律二 #5：就是 0.005，与端到端差 10 倍
    grad_clip=dict(max_norm=35, norm_type=2))

scheduler = dict(
    type='LinearWarmupMultiStepLR',
    start_factor=1.0 / 3,
    warmup_iters=500,
    milestones=[8, 11],
    gamma=0.1)

logger = dict(type='RunLogger')

max_epoch = 12
eval_interval = 12
checkpoint_interval = 1
log_interval = 50
