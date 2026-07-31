# Point2RBox-v2 伪标签生成（stage-1.5，纯推理）
# 对照官方 point2rbox_v2-pseudo-generator-dota.py（PLAN §6.2）：
#   _base_ = v2 端到端 config；model.bbox_head.pseudo_generator=True；
#   test_pipeline = train 数据（含 ConvertWeakSupervision）但去掉 RandomFlip；
#   test_evaluator = DOTAMetric(format_only=True,
#       outfile_prefix='data/split_ss_dota/point2rbox_v2_pseudo_labels')
# 执行入口：tools/generate_pseudo_labels.py --config 本文件 --ckpt <ckpt>
_base_cfg = 'configs/point2rbox_v2/point2rbox_v2_1x_dota.py'

# 覆盖项（其余全部继承 _base_cfg，由工具展开）
pseudo_generator = True
outfile_prefix = '/root/data/split_ss_dota/point2rbox_v2_pseudo_labels'

# 官方 test_dataloader = _base_.train_dataloader（去 RandomFlip）
pseudo_dataset = dict(
    type='P2RV2DOTADataset',
    images_dir='/root/data/split_ss_dota/trainval/images',
    annfiles_dir='/root/data/split_ss_dota/trainval/annfiles',
    version='1',
    point_proportion=1.0,
    hbox_proportion=0.0,
    weak_supervision=True,
    filter_empty_gt=True,
    transforms=[
        dict(type='MMRotateResize', min_size=1024, max_size=1024),
        dict(type='Pad', size_divisor=32),
        dict(type='Normalize', mean=[123.675, 116.28, 103.53],
             std=[58.395, 57.12, 57.375], to_bgr=False),
    ],
    batch_size=2,   # 官方 test_dataloader=_base_.train_dataloader → batch_size=2
    num_workers=0,
    shuffle=False)
