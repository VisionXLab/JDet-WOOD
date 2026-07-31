# Point2RBox-v2 端到端 DOTA-v1.0（Jittor）
# 逐行对照官方 /root/ref/Point2RBox-v3/configs/point2rbox_v2/point2rbox_v2-1x-dota.py
# （PLAN §6.0/§6.1；铁律一：任何数值/开关/顺序不得偏离；对照表见 docs/config_parity.md）

# model.data_preprocessor（mean/std/bgr_to_rgb/pad_size_divisor）在 JDet 由
# dataset transforms 实现（Normalize/Pad），mean/std 同时传给 model 供 TED 反归一化
preprocess_mean = [123.675, 116.28, 103.53]
preprocess_std = [58.395, 57.12, 57.375]

model = dict(
    type='Point2RBoxV2',
    # detector 级参数（官方 config 顶层 model.*）
    rotate_range=(0.25, 0.75),        # 官方默认（config 未覆盖）
    scale_range=(0.5, 0.9),           # 官方默认
    ss_prob=[0.68, 0.07, 0.25],       # 铁律二 #9：非整数概率照抄
    copy_paste_start_epoch=6,
    num_copies=10,                    # 官方默认
    data_preprocessor=dict(
        mean=preprocess_mean,
        std=preprocess_std,
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),        # 铁律二 #11：显式照抄
    backbone=dict(
        type='Resnet50',
        # 官方 out_indices=(1,2,3) → layer2/3/4（铁律二 #7：v2 与 stage-2 各是各的）
        return_stages=['layer2', 'layer3', 'layer4'],
        frozen_stages=1,
        norm_eval=True,
        pretrained=True),             # init torchvision://resnet50
    neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=128,
        start_level=0,
        add_extra_convs='on_output',  # 铁律二 #11
        num_outs=3,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='Point2RBoxV2Head',
        num_classes=15,
        in_channels=128,
        feat_channels=128,
        strides=[8],                  # 铁律二 #6：单层特征，照抄
        edge_loss_start_epoch=6,
        joint_angle_start_epoch=1,
        voronoi_type='standard',
        voronoi_thres=dict(
            default=[0.994, 0.005],
            override=(([2, 11], [0.999, 0.6]),
                      ([7, 8, 10, 14], [0.95, 0.005]))),  # 铁律二 #8：一个数都不能动
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
            type='MMDetFocalLoss',    # 官方 mmdet.FocalLoss（底座 FocalLoss 是 1-based 标签，语义不同）
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0),
        loss_overlap=dict(type='GaussianOverlapLoss', loss_weight=10.0, lamb=0),
        loss_voronoi=dict(type='VoronoiWatershedLoss', loss_weight=5.0),
        loss_bbox_edg=dict(type='EdgeLoss', loss_weight=0.3),
        loss_ss=dict(type='Point2RBoxV2ConsistencyLoss', loss_weight=1.0),
        test_cfg=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms_rotated', iou_threshold=0.1),
            max_per_img=2000)),
    train_cfg=None)

# 官方 train_pipeline（§6.1，顺序照抄）：
#   LoadImageFromFile → LoadAnnotations(qbox) → ConvertBoxType(rbox)
#   → ConvertWeakSupervision(point=1., hbox=0) → Resize((1024,1024), keep_ratio)
#   → RandomFlip(0.75, [h,v,diag]) → PackDetInputs
# 前四步由 P2RV2DOTADataset 实现（weak_supervision=True, point_dummy=1 官方默认）
dataset = dict(
    train=dict(
        type='P2RV2DOTADataset',
        images_dir='/root/data/split_ss_dota/trainval/images',
        annfiles_dir='/root/data/split_ss_dota/trainval/annfiles',
        version='1',
        point_proportion=1.0,
        hbox_proportion=0.0,
        weak_supervision=True,
        filter_empty_gt=True,         # 铁律二 #11
        transforms=[
            dict(type='MMRotateResize', min_size=1024, max_size=1024),
            dict(type='MMRotateRandomFlip', prob=0.75,
                 direction=['horizontal', 'vertical', 'diagonal']),
            dict(type='Pad', size_divisor=32),
            dict(type='Normalize', mean=preprocess_mean, std=preprocess_std,
                 to_bgr=False),
        ],
        batch_size=2,                 # 官方总 batch=2（单卡，PLAN §9.1）
        # [plan-deviation] 官方 num_workers=2；jittor dataset 多进程环形缓冲存在
        # send/idqueue 竞态死锁（实测 40min 触发，worker 卡 buffer.send、主进程卡
        # idqueue.pop）。num_workers 是加载性能参数、不影响任何数值语义，改 0 规避。
        num_workers=0,
        shuffle=True),
    val=dict(
        type='P2RV2DOTADataset',
        # 铁律二 #4：官方 val 指向 trainval/，照抄（不是干净验证集）
        images_dir='/root/data/split_ss_dota/trainval/images',
        annfiles_dir='/root/data/split_ss_dota/trainval/annfiles',
        version='1',
        weak_supervision=False,       # val pipeline 无 ConvertWeakSupervision
        filter_empty_gt=False,
        transforms=[
            dict(type='MMRotateResize', min_size=1024, max_size=1024),
            dict(type='Pad', size_divisor=32),
            dict(type='Normalize', mean=preprocess_mean, std=preprocess_std,
                 to_bgr=False),
        ],
        batch_size=16,                # 官方 val batch_size=16
        # [plan-deviation] 同 train：评测也走同一 Jittor ring-buffer 实现。
        num_workers=0,
        shuffle=False),
    test=dict(
        type='ImageDataset',
        # JDet merger registry calls DOTA-v1.0 simply "DOTA".
        dataset_type='DOTA',
        images_dir='/root/data/split_ss_dota/test/images',
        transforms=[
            dict(type='MMRotateResize', min_size=1024, max_size=1024),
            dict(type='Pad', size_divisor=32),
            dict(type='Normalize', mean=preprocess_mean, std=preprocess_std,
                 to_bgr=False),
        ],
        batch_size=4,                 # 官方 test batch_size=4
        # [plan-deviation] 同 train：仅加载性能变化，不改样本或数值语义。
        num_workers=0,
        shuffle=False))

# 官方 optim_wrapper（§6.1）：AdamW(_delete_=True) 但 clip_grad 仍生效（铁律二 #1）
optimizer = dict(
    type='AdamW',
    lr=0.00005,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    grad_clip=dict(max_norm=35, norm_type=2))

# 官方 param_scheduler（铁律二 #2）：LinearLR(1/3, by_epoch=False, 0→500)
# × MultiStepLR([8,11], 0.1, by_epoch=True)，与 mmengine 逐点相等（tests L1）
scheduler = dict(
    type='LinearWarmupMultiStepLR',
    start_factor=1.0 / 3,
    warmup_iters=500,
    milestones=[8, 11],
    gamma=0.1)

logger = dict(type='RunLogger')

# 官方 train_cfg=EpochBasedTrainLoop(max_epochs=12, val_interval=12)
max_epoch = 12
eval_interval = 12
checkpoint_interval = 1               # 官方 default_hooks.checkpoint interval=1
log_interval = 50                     # 官方 default_hooks.logger interval=50
