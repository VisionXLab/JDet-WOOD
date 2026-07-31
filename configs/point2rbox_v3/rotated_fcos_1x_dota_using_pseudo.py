# Stage-2 rotated-FCOS on Point2RBox-v3 伪标签（Jittor）
# 逐行对照官方 rotated-fcos-1x-dotav1-0-using-pseudo.py。
# L0 golden：config_rotated-fcos-1x-dotav1-0-using-pseudo.json（270 键）。
#
# 保留 mmrotate 数值语义，同时使用 JDet 的
# SingleStageDetector/P2RV2DOTADataset 接口名。

preprocess_mean = [123.675, 116.28, 103.53]
preprocess_std = [58.395, 57.12, 57.375]

model = dict(
    type='FCOS',                          # 官方 mmdet.FCOS
    backbone=dict(
        type='Resnet50',
        # 官方 out_indices=(0,1,2,3) → layer1..layer4（铁律二 #7：与端到端一致的四层）
        return_stages=['layer1', 'layer2', 'layer3', 'layer4'],
        frozen_stages=1,
        norm_eval=True,
        pretrained=True),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=512,                 # 铁律二：512 通道，不是 256
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
        center_sampling=True,
        center_sample_radius=1.5,         # 官方值（≠v3 head 的 0.75）
        norm_on_bbox=True,
        centerness_on_reg=True,
        use_hbbox_loss=False,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version='le90'),
        loss_cls=dict(
            type='MMDetFocalLoss',   # 官方 mmdet.FocalLoss（底座 FocalLoss 是 1-based，语义不同）
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_angle=None,
        loss_centerness=dict(
            type='MMDetCrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        test_cfg=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms_rotated', iou_threshold=0.1),
            max_per_img=2000)))

# 官方 train_pipeline：LoadAnnotations(box_type='rbox')，无 ConvertWeakSupervision
# （全监督阶段）；Resize/Flip 同端到端
dataset = dict(
    train=dict(
        type='P2RV2DOTADataset',
        images_dir='/root/data/split_ss_dota/trainval/images',
        annfiles_dir='/root/data/split_ss_dota/trainval/annfiles',
        ann_json='/root/data/split_ss_dota/point2rbox_v3_pseudo_labels.bbox.json',
        version='1',
        weak_supervision=False,           # 无 ConvertWeakSupervision
        filter_empty_gt=True,
        transforms=[
            dict(type='MMRotateResize', min_size=1024, max_size=1024),
            dict(type='MMRotateRandomFlip', prob=0.75,
                 direction=['horizontal', 'vertical', 'diagonal']),
            dict(type='Pad', size_divisor=32),
            dict(type='Normalize', mean=preprocess_mean, std=preprocess_std,
                 to_bgr=False),
        ],
        batch_size=4,                     # 官方 stage-2 batch_size=4
        # Jittor Dataset 的多进程 ring buffer 在该数据量下可能死锁；仅影响
        # 加载性能，不改变训练数值语义。
        num_workers=0,
        shuffle=True),
    val=dict(
        type='P2RV2DOTADataset',
        images_dir='/root/data/split_ss_dota/trainval/images',   # 铁律二 #5
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
        batch_size=4,                     # 官方 stage-2 val batch_size=4
        num_workers=0,
        shuffle=False),
    test=dict(
        type='ImageDataset',
        # get_classes_by_name 同时支持 DOTA/DOTA1，但 JDet 的自动 merge
        # 入口只接受 DOTA；使用 canonical 名避免完整 test 后在最后一步退出。
        dataset_type='DOTA',
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

# 官方：AdamW lr=5e-5, betas 同, weight_decay=0.005 —— 铁律二 #6：与端到端差 10 倍，照抄
optimizer = dict(
    type='AdamW',
    lr=0.00005,
    betas=(0.9, 0.999),
    weight_decay=0.005,
    grad_clip=dict(max_norm=35, norm_type=2))   # 铁律二 #1：clip_grad 仍生效

scheduler = dict(
    type='LinearWarmupMultiStepLR',
    start_factor=1.0 / 3,
    warmup_iters=500,
    milestones=[8, 11],
    gamma=0.1)

# 官方 val_evaluator：DOTAMetric(metric='mAP', iou_thrs=[0.5, 0.75])
evaluator = dict(type='DOTAMetric', metric='mAP', iou_thrs=[0.5, 0.75])

logger = dict(type='RunLogger')

max_epoch = 12
eval_interval = 12
checkpoint_interval = 1
log_interval = 50
