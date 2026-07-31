"""L0 配置 parity（v3）：3 个 Jittor config ↔ 官方 v3 golden 逐值比对。

golden（tests/parity/golden/config_point2rbox_v3-*.json，616/614/270 键）由
mmengine 解析官方 config 后展平生成（含 list/tuple 的 __type__ 标签，
mask_filter_config 混合 int/str 键按 str 排序）。

豁免清单（铁律一允许变的三类，逐条注明 docs/config_parity.md 映射表出处）：
  E1  registry 类名：mmdet.ResNet→Resnet50 / mmdet.FPN→FPN / mmdet.FocalLoss→MMDetFocalLoss（0-based 语义，底座 FocalLoss 是 1-based 禁用）
      / mmdet.FCOS→FCOS / mmdet.CrossEntropyLoss→CrossEntropyLoss（映射表 1/3/4）
  E2  backbone out_indices=(0,1,2,3) → return_stages=['layer1'..'layer4']（映射表 1）
  E3  init_cfg=Pretrained(torchvision://resnet50) → pretrained=True（映射表 2）
  E4  data_preprocessor 由 transforms Normalize+Pad 实现，mean/std 双传（映射表 14）
  E5  pipeline 前四步（Load/ConvertBoxType/ConvertWeakSupervision）由
      P2RV2DOTADataset 内建（映射表 11）；Resize/RandomFlip → MMRotate* 同参数（12/13）
  E6  optim_wrapper.optimizer+clip_grad → optimizer(..., grad_clip)（映射表 6）
  E7  param_scheduler 两件套 → LinearWarmupMultiStepLR（映射表 7；LR 序列 L1 已证逐点等）
  E8  custom_hooks SetEpochInfoHook → Runner 内建 set_epoch（映射表 8）
  E9  数据路径写法（data_root 相对路径 → 绝对路径）
其余任何键值差异 = FAIL（数值/布尔/顺序/长度/tuple-list 类型零容差）。
"""
import json
import os

import pytest

HERE = os.path.dirname(__file__)
GOLDEN = os.path.join(HERE, 'golden')
CFG_DIR = os.path.join(HERE, '..', '..', 'configs', 'point2rbox_v3')


def load_golden(name):
    with open(os.path.join(GOLDEN, name)) as f:
        return json.load(f)


def load_jt_config(fname):
    path = os.path.abspath(os.path.join(CFG_DIR, fname))
    ns = {'__file__': path}
    with open(path) as f:
        exec(compile(f.read(), path, 'exec'), ns)
    return ns


def flatten(node, prefix, out):
    # 与 golden 生成端一致（str 排序 + __type__ 标签）
    if isinstance(node, dict):
        if not node:
            out[prefix] = {}
        for k in sorted(node, key=str):
            flatten(node[k], f'{prefix}.{k}' if prefix else str(k), out)
    elif isinstance(node, (list, tuple)):
        out[f'{prefix}.__len__'] = len(node)
        out[f'{prefix}.__type__'] = type(node).__name__
        for i, v in enumerate(node):
            flatten(v, f'{prefix}[{i}]', out)
    else:
        out[prefix] = node


@pytest.fixture(scope='module')
def cfg():
    return load_jt_config('point2rbox_v3_1x_dota.py')


@pytest.fixture(scope='module')
def g():
    return load_golden('config_point2rbox_v3-1x-dotav1-0.json')


@pytest.fixture(scope='module')
def cfg_pg():
    return load_jt_config('point2rbox_v3_pseudo_generator_dota.py')


@pytest.fixture(scope='module')
def g_pg():
    return load_golden('config_point2rbox_v3-pseudo-generator-dotav1-0.json')


@pytest.fixture(scope='module')
def cfg_s2():
    return load_jt_config('rotated_fcos_1x_dota_using_pseudo.py')


@pytest.fixture(scope='module')
def g_s2():
    return load_golden('config_rotated-fcos-1x-dotav1-0-using-pseudo.json')


class TestE2EConfig:

    def test_sam_vars_pointwise(self, cfg, g):
        """铁律二 #7/#8/#9：_base_sam 三变量逐 key 零容差（含 tuple/list 类型）。"""
        mine = {}
        flatten(cfg['sam_instance_thr'], 'sam_instance_thr', mine)
        flatten(cfg['mask_filter_config'], 'mask_filter_config', mine)
        flatten(cfg['sam_sample_rules'], 'sam_sample_rules', mine)
        golden_sub = {k: v for k, v in g.items()
                      if k.split('.')[0].split('[')[0] in
                      ('sam_instance_thr', 'mask_filter_config', 'sam_sample_rules')}
        assert set(mine) == set(golden_sub), (
            set(mine) ^ set(golden_sub))
        for k, v in golden_sub.items():
            assert mine[k] == v, (k, mine[k], v)

    def test_loss_voronoi_gets_sam_vars(self, cfg, g):
        lv = cfg['model']['bbox_head']['loss_voronoi']
        assert lv['sam_instance_thr'] == g['sam_instance_thr'] == 4
        assert lv['mask_filter_config'] is cfg['mask_filter_config']
        assert lv['sam_sample_rules'] is cfg['sam_sample_rules']
        assert lv['use_class_specific_watershed'] is False
        # filter_pairs (3, 10, 200)
        fp = lv['sam_sample_rules']['filter_pairs']
        assert list(fp[0]) == [3, 10, 200] and len(fp) == 1

    def test_model_detector(self, cfg, g):
        m = cfg['model']
        assert list(m['ss_prob']) == [g[f'model.ss_prob[{i}]'] for i in range(3)]
        assert m['copy_paste_start_epoch'] == g['model.copy_paste_start_epoch']
        # 铁律二 #4：eopch 拼写键必须存在且值一致
        assert m['label_assign_pseudo_label_switch_eopch'] == \
            g['model.label_assign_pseudo_label_switch_eopch'] == 6
        dp = m['data_preprocessor']
        for i in range(3):
            assert dp['mean'][i] == g[f'model.data_preprocessor.mean[{i}]']
            assert dp['std'][i] == g[f'model.data_preprocessor.std[{i}]']
        assert dp['bgr_to_rgb'] == g['model.data_preprocessor.bgr_to_rgb']
        assert dp['pad_size_divisor'] == g['model.data_preprocessor.pad_size_divisor']
        assert dp['boxtype2tensor'] == g['model.data_preprocessor.boxtype2tensor']

    def test_backbone_neck(self, cfg, g):
        b = cfg['model']['backbone']
        # E2：out_indices=(0,1,2,3) ↔ layer1..4
        assert [g[f'model.backbone.out_indices[{i}]'] for i in
                range(g['model.backbone.out_indices.__len__'])] == [0, 1, 2, 3]
        assert b['return_stages'] == ['layer1', 'layer2', 'layer3', 'layer4']
        assert b['frozen_stages'] == g['model.backbone.frozen_stages']
        assert b['norm_eval'] == g['model.backbone.norm_eval']
        n = cfg['model']['neck']
        assert n['in_channels'] == [g[f'model.neck.in_channels[{i}]'] for i in range(4)]
        assert n['out_channels'] == g['model.neck.out_channels'] == 256
        assert n['start_level'] == g['model.neck.start_level'] == 1
        assert n['num_outs'] == g['model.neck.num_outs'] == 5
        assert n['add_extra_convs'] == g['model.neck.add_extra_convs']
        assert n['relu_before_extra_convs'] == g['model.neck.relu_before_extra_convs']

    def test_head(self, cfg, g):
        h = cfg['model']['bbox_head']
        assert h['num_classes'] == g['model.bbox_head.num_classes']
        assert h['in_channels'] == g['model.bbox_head.in_channels']
        assert h['feat_channels'] == g['model.bbox_head.feat_channels']
        assert h['strides'] == [g[f'model.bbox_head.strides[{i}]'] for i in range(5)]
        assert g['model.bbox_head.strides.__len__'] == 5     # 铁律二 #11：v3 五尺度
        assert h['use_adaptive_scale'] == g['model.bbox_head.use_adaptive_scale'] is False
        assert h['edge_loss_start_epoch'] == g['model.bbox_head.edge_loss_start_epoch'] == 6
        assert h['joint_angle_start_epoch'] == g['model.bbox_head.joint_angle_start_epoch'] == 1
        assert h['voronoi_type'] == g['model.bbox_head.voronoi_type'] == 'standard'
        assert h['square_cls'] == [g[f'model.bbox_head.square_cls[{i}]'] for i in range(3)]
        assert h['edge_loss_cls'] == [
            g[f'model.bbox_head.edge_loss_cls[{i}]']
            for i in range(g['model.bbox_head.edge_loss_cls.__len__'])]
        assert h['post_process'] == {11: 1.2}
        # voronoi_thres 含结构类型（铁律二 #10）
        vt = {}
        flatten(h['voronoi_thres'], 'model.bbox_head.voronoi_thres', vt)
        for k, v in vt.items():
            assert g[k] == v, (k, v, g[k])
        ac = h['angle_coder']
        for key in ('angle_version', 'dual_freq', 'num_step', 'thr_mod'):
            assert ac[key] == g[f'model.bbox_head.angle_coder.{key}']

    def test_losses(self, cfg, g):
        h = cfg['model']['bbox_head']
        assert h['loss_cls']['gamma'] == g['model.bbox_head.loss_cls.gamma']
        assert h['loss_cls']['alpha'] == g['model.bbox_head.loss_cls.alpha']
        assert h['loss_cls']['use_sigmoid'] == g['model.bbox_head.loss_cls.use_sigmoid']
        assert h['loss_cls']['loss_weight'] == g['model.bbox_head.loss_cls.loss_weight']
        assert h['loss_bbox']['loss_weight'] == g['model.bbox_head.loss_bbox.loss_weight'] == 5.0
        assert h['loss_bbox']['loss_type'] == 'gwd'
        assert h['loss_overlap']['loss_weight'] == \
            g['model.bbox_head.loss_overlap.loss_weight'] == 10.0
        assert h['loss_overlap']['lamb'] == g['model.bbox_head.loss_overlap.lamb'] == 0
        assert h['loss_voronoi']['loss_weight'] == \
            g['model.bbox_head.loss_voronoi.loss_weight'] == 5.0
        assert h['loss_bbox_edg']['loss_weight'] == \
            g['model.bbox_head.loss_bbox_edg.loss_weight'] == 0.3
        assert h['loss_ss']['loss_weight'] == g['model.bbox_head.loss_ss.loss_weight'] == 1.0

    def test_test_cfg(self, cfg, g):
        t = cfg['model']['bbox_head']['test_cfg']
        assert t['nms_pre'] == g['model.test_cfg.nms_pre'] == 2000
        assert t['min_bbox_size'] == g['model.test_cfg.min_bbox_size'] == 0
        assert t['score_thr'] == g['model.test_cfg.score_thr'] == 0.05
        assert t['nms']['iou_threshold'] == g['model.test_cfg.nms.iou_threshold'] == 0.1
        assert t['max_per_img'] == g['model.test_cfg.max_per_img'] == 2000

    def test_pipeline_order_and_values(self, cfg, g):
        """E5：前四步在 dataset 内建；Resize/Flip 参数逐值。官方顺序锚定。"""
        officials = [g[f'train_pipeline[{i}].type']
                     for i in range(g['train_pipeline.__len__'])]
        assert officials == ['mmdet.LoadImageFromFile', 'mmdet.LoadAnnotations',
                             'ConvertBoxType', 'ConvertWeakSupervision',
                             'mmdet.Resize', 'mmdet.RandomFlip', 'mmdet.PackDetInputs']
        d = cfg['dataset']['train']
        assert d['point_proportion'] == g['train_pipeline[3].point_proportion'] == 1.0
        assert d['hbox_proportion'] == g['train_pipeline[3].hbox_proportion'] == 0
        tr = [t['type'] for t in d['transforms']]
        assert tr.index('MMRotateResize') < tr.index('MMRotateRandomFlip')  # 顺序照抄
        td = {t['type']: t for t in d['transforms']}
        assert g['train_pipeline[4].scale[0]'] == 1024
        assert td['MMRotateResize']['min_size'] == 1024
        assert td['MMRotateRandomFlip']['prob'] == g['train_pipeline[5].prob'] == 0.75
        assert td['MMRotateRandomFlip']['direction'] == \
            [g[f'train_pipeline[5].direction[{i}]'] for i in range(3)]

    def test_dataloaders(self, cfg, g):
        d = cfg['dataset']
        assert d['train']['batch_size'] == g['train_dataloader.batch_size'] == 2
        # E10: 官方 num_workers=2（torch DataLoader infra），jittor 多进程 dataset
        # 有环形缓冲死锁（A commit 3d87c60）→ jdet 侧映射为 0。golden 锚定官方值，
        # jdet 值锚定 0（豁免详见 docs/config_parity.md「追加豁免」）。
        assert g['train_dataloader.num_workers'] == 2
        assert d['train']['num_workers'] == 0
        assert d['val']['batch_size'] == g['val_dataloader.batch_size'] == 16
        assert d['test']['batch_size'] == g['test_dataloader.batch_size'] == 4
        assert d['train']['filter_empty_gt'] is True
        assert 'trainval' in d['val']['images_dir']       # 铁律二 #5
        assert 'trainval' in g['val_dataloader.dataset.ann_file']

    def test_optimizer_scheduler(self, cfg, g):
        o = cfg['optimizer']
        assert o['type'] == 'AdamW'
        assert o['lr'] == g['optim_wrapper.optimizer.lr'] == 5e-05
        assert tuple(o['betas']) == (g['optim_wrapper.optimizer.betas[0]'],
                                     g['optim_wrapper.optimizer.betas[1]'])
        assert o['weight_decay'] == g['optim_wrapper.optimizer.weight_decay'] == 0.05
        # 铁律二 #1：_delete_ 后 clip_grad 仍生效
        assert o['grad_clip']['max_norm'] == g['optim_wrapper.clip_grad.max_norm'] == 35
        assert o['grad_clip']['norm_type'] == g['optim_wrapper.clip_grad.norm_type'] == 2
        s = cfg['scheduler']
        assert s['start_factor'] == pytest.approx(g['param_scheduler[0].start_factor'])
        assert s['warmup_iters'] == g['param_scheduler[0].end'] == 500
        assert g['param_scheduler[0].by_epoch'] is False
        assert s['milestones'] == [8, 11]
        assert s['gamma'] == g['param_scheduler[1].gamma'] == 0.1
        assert cfg['max_epoch'] == g['train_cfg.max_epochs'] == 12
        assert cfg['eval_interval'] == g['train_cfg.val_interval'] == 12
        # E8：custom_hooks 的 epoch 注入由 runner 内建
        assert g['custom_hooks[0].type'] == 'mmdet.SetEpochInfoHook'


class TestPseudoGeneratorConfig:

    def test_inherits_and_overrides(self, cfg_pg, g_pg):
        assert cfg_pg['model']['bbox_head']['pseudo_generator'] is True
        assert g_pg['model.bbox_head.pseudo_generator'] is True
        # 端到端其余数值不变（抽 5 个锚点）
        assert cfg_pg['model']['bbox_head']['voronoi_type'] == 'standard'
        assert cfg_pg['model']['ss_prob'] == [0.68, 0.07, 0.25]
        assert cfg_pg['optimizer']['weight_decay'] == 0.05
        assert cfg_pg['model']['bbox_head']['loss_voronoi']['sam_instance_thr'] == 4
        assert cfg_pg['max_epoch'] == 12

    def test_test_stream_is_trainval_no_flip(self, cfg_pg, g_pg):
        t = cfg_pg['dataset']['test']
        assert 'trainval' in t['images_dir']
        # 官方 test_pipeline = train - RandomFlip
        types = [x['type'] for x in t['transforms']]
        assert 'MMRotateRandomFlip' not in types
        officials = [g_pg[f'test_pipeline[{i}].type']
                     for i in range(g_pg['test_pipeline.__len__'])]
        assert 'mmdet.RandomFlip' not in officials
        assert t['weak_supervision'] is True      # ConvertWeakSupervision 保留
        assert t['batch_size'] == g_pg['test_dataloader.batch_size'] == 2

    def test_evaluator_outfile(self, cfg_pg, g_pg):
        ev = cfg_pg['evaluator']
        assert ev['format_only'] is True
        assert g_pg['test_evaluator.format_only'] is True
        assert 'point2rbox_v3_pseudo_labels' in ev['outfile_prefix']
        assert 'point2rbox_v3_pseudo_labels' in g_pg['test_evaluator.outfile_prefix']


class TestStage2Config:

    def test_iron_rules(self, cfg_s2, g_s2):
        assert cfg_s2['optimizer']['weight_decay'] == \
            g_s2['optim_wrapper.optimizer.weight_decay'] == 0.005   # 铁律二 #6
        assert cfg_s2['model']['neck']['out_channels'] == \
            g_s2['model.neck.out_channels'] == 512
        assert cfg_s2['dataset']['train']['batch_size'] == \
            g_s2['train_dataloader.batch_size'] == 4
        assert cfg_s2['model']['backbone']['return_stages'] == \
            ['layer1', 'layer2', 'layer3', 'layer4']                # 铁律二 #7
        assert g_s2['model.backbone.out_indices.__len__'] == 4

    def test_head(self, cfg_s2, g_s2):
        h = cfg_s2['model']['roi_heads']
        for key in ('num_classes', 'in_channels', 'stacked_convs', 'feat_channels',
                    'center_sampling', 'center_sample_radius', 'norm_on_bbox',
                    'centerness_on_reg', 'use_hbbox_loss', 'scale_angle'):
            assert h[key] == g_s2[f'model.bbox_head.{key}'], key
        assert h['center_sample_radius'] == 1.5
        assert h['strides'] == [8, 16, 32, 64, 128]
        assert h['loss_angle'] is None
        assert g_s2['model.bbox_head.loss_angle'] is None
        assert h['loss_bbox']['loss_weight'] == \
            g_s2['model.bbox_head.loss_bbox.loss_weight'] == 1.0
        assert h['loss_centerness']['use_sigmoid'] is True
        assert h['bbox_coder']['angle_version'] == \
            g_s2['model.bbox_head.bbox_coder.angle_version'] == 'le90'

    def test_pipeline_fully_supervised(self, cfg_s2, g_s2):
        # 官方 stage-2 无 ConvertWeakSupervision
        officials = [g_s2[f'train_pipeline[{i}].type']
                     for i in range(g_s2['train_pipeline.__len__'])]
        assert 'ConvertWeakSupervision' not in officials
        assert cfg_s2['dataset']['train']['weak_supervision'] is False
        assert 'point2rbox_v3_pseudo_labels.bbox.json' in \
            cfg_s2['dataset']['train']['ann_json']
        assert 'point2rbox_v3_pseudo_labels.bbox.json' in \
            g_s2['train_dataloader.dataset.ann_file']

    def test_val_evaluator(self, cfg_s2, g_s2):
        ev = cfg_s2['evaluator']
        assert ev['metric'] == g_s2['val_evaluator.metric'] == 'mAP'
        assert ev['iou_thrs'] == [g_s2['val_evaluator.iou_thrs[0]'],
                                  g_s2['val_evaluator.iou_thrs[1]']] == [0.5, 0.75]
        assert cfg_s2['dataset']['val']['batch_size'] == \
            g_s2['val_dataloader.batch_size'] == 4
        # DOTA 与 DOTA1 类别表完全相同，但 runner 自动 merge 只接受 DOTA。
        assert cfg_s2['dataset']['test']['dataset_type'] == 'DOTA'
