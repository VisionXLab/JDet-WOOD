"""L0 配置 parity：Jittor config ↔ 官方 config golden 逐值零容差比对。

golden 由 tools/dump_config.py 在 p2r-torch 环境生成（296/297/246 个扁平键）。
Jittor 侧 config（configs/point2rbox_v2/*.py）在 M4 编写；写好后在
CONFIG_PAIRS 里登记「Jittor config 路径 + 键映射函数」，测试自动逐值比对。

映射规则（铁律一允许变的三类）在 to_official_key()/normalize_value() 中集中声明，
其余任何键值差异都是 FAIL——数值、布尔、pipeline 顺序、列表长度零容差。
"""
import json
import os

import pytest

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden')

# (golden json, jittor config path, 适配器)；M4 起逐个填充
CONFIG_PAIRS = [
    # ('config_point2rbox_v2-1x-dota.json', 'configs/point2rbox_v2/point2rbox_v2_1x_dota.py', ...),
]


def load_golden(name):
    with open(os.path.join(GOLDEN, name)) as f:
        return json.load(f)


class TestGoldenIntegrity:
    """golden 自身的完整性 + 铁律二关键值锚定（防 golden 被误重dump 后漂移）。"""

    def test_iron_rules_anchored(self):
        d = load_golden('config_point2rbox_v2-1x-dota.json')
        assert d['optim_wrapper.clip_grad.max_norm'] == 35            # 铁律二 #1
        assert d['optim_wrapper.clip_grad.norm_type'] == 2
        assert d['param_scheduler[0].type'] == 'LinearLR'             # 铁律二 #2
        assert d['param_scheduler[0].start_factor'] == pytest.approx(1 / 3)
        assert d['param_scheduler[0].end'] == 500
        assert d['param_scheduler[0].by_epoch'] is False
        assert d['param_scheduler[1].type'] == 'MultiStepLR'
        assert d['param_scheduler[1].milestones[0]'] == 8
        assert d['param_scheduler[1].milestones[1]'] == 11
        assert d['custom_hooks[0].type'] == 'mmdet.SetEpochInfoHook'  # 铁律二 #3
        assert d['model.bbox_head.strides.__len__'] == 1              # 铁律二 #6
        assert d['model.bbox_head.strides[0]'] == 8
        assert d['model.bbox_head.square_cls[0]'] == 1                # 铁律二 #8
        assert d['model.bbox_head.square_cls[1]'] == 9
        assert d['model.bbox_head.square_cls[2]'] == 11
        assert d['model.ss_prob[0]'] == 0.68                          # 铁律二 #9（挂在 model 下）
        assert d['model.ss_prob[1]'] == 0.07
        assert d['model.ss_prob[2]'] == 0.25
        assert d['optim_wrapper.optimizer.weight_decay'] == 0.05
        assert d['optim_wrapper.optimizer.lr'] == 5e-05
        assert d['train_cfg.max_epochs'] == 12
        assert d['train_dataloader.batch_size'] == 2

    def test_stage2_iron_rules(self):
        d = load_golden('config_rotated-fcos-1x-dota-using-pseudo.json')
        assert d['optim_wrapper.optimizer.weight_decay'] == 0.005     # 铁律二 #5：就是差 10 倍
        assert d['model.backbone.out_indices[0]'] == 0                # 铁律二 #7
        assert d['model.backbone.out_indices.__len__'] == 4
        assert d['model.neck.out_channels'] == 512
        assert d['train_dataloader.batch_size'] == 4

    def test_val_points_to_trainval(self):
        d = load_golden('config_point2rbox_v2-1x-dota.json')
        # 铁律二 #4：val 指向 trainval/，照抄
        assert 'trainval' in d['val_dataloader.dataset.ann_file']


@pytest.mark.skipif(not CONFIG_PAIRS, reason='Jittor 侧 config 尚未编写（M4）')
class TestConfigParity:
    def test_pointwise(self):
        raise NotImplementedError  # M4 实现：加载 Jittor config → 映射 → 与 golden 全键零容差


def _load_jt_config(path):
    ns = {}
    with open(path) as f:
        exec(compile(f.read(), path, 'exec'), ns)
    return ns


class TestJittorConfigParity:
    """Jittor config ↔ 官方 golden 逐值比对（§6.1 勾选表的可执行形式）。

    允许差异仅限铁律一的三类：registry 类名、config 语法、数据路径写法。
    """

    @pytest.fixture(scope='class')
    def cfg(self):
        path = os.path.join(os.path.dirname(__file__), '..', '..',
                            'configs', 'point2rbox_v2', 'point2rbox_v2_1x_dota.py')
        return _load_jt_config(path)

    @pytest.fixture(scope='class')
    def g(self):
        return load_golden('config_point2rbox_v2-1x-dota.json')

    def test_model_detector(self, cfg, g):
        m = cfg['model']
        assert list(m['ss_prob']) == [g[f'model.ss_prob[{i}]'] for i in range(3)]
        assert m['copy_paste_start_epoch'] == g['model.copy_paste_start_epoch']
        dp = m['data_preprocessor']
        for i in range(3):
            assert dp['mean'][i] == g[f'model.data_preprocessor.mean[{i}]']
            assert dp['std'][i] == g[f'model.data_preprocessor.std[{i}]']
        assert dp['bgr_to_rgb'] == g['model.data_preprocessor.bgr_to_rgb']
        assert dp['pad_size_divisor'] == g['model.data_preprocessor.pad_size_divisor']
        assert dp['boxtype2tensor'] == g['model.data_preprocessor.boxtype2tensor']

    def test_model_backbone_neck(self, cfg, g):
        b = cfg['model']['backbone']
        # out_indices=(1,2,3) ↔ return_stages layer2/3/4（类名/写法映射）
        assert [g[f'model.backbone.out_indices[{i}]'] for i in
                range(g['model.backbone.out_indices.__len__'])] == [1, 2, 3]
        assert b['return_stages'] == ['layer2', 'layer3', 'layer4']
        assert b['frozen_stages'] == g['model.backbone.frozen_stages']
        assert b['norm_eval'] == g['model.backbone.norm_eval']
        n = cfg['model']['neck']
        assert n['in_channels'] == [g[f'model.neck.in_channels[{i}]'] for i in range(3)]
        assert n['out_channels'] == g['model.neck.out_channels']
        assert n['start_level'] == g['model.neck.start_level']
        assert n['num_outs'] == g['model.neck.num_outs']
        assert n['add_extra_convs'] == g['model.neck.add_extra_convs']
        assert n['relu_before_extra_convs'] == g['model.neck.relu_before_extra_convs']

    def test_model_head(self, cfg, g):
        h = cfg['model']['bbox_head']
        assert h['num_classes'] == g['model.bbox_head.num_classes']
        assert h['in_channels'] == g['model.bbox_head.in_channels']
        assert h['feat_channels'] == g['model.bbox_head.feat_channels']
        assert h['strides'] == [g['model.bbox_head.strides[0]']]
        assert g['model.bbox_head.strides.__len__'] == 1
        assert h['edge_loss_start_epoch'] == g['model.bbox_head.edge_loss_start_epoch']
        assert h['joint_angle_start_epoch'] == g['model.bbox_head.joint_angle_start_epoch']
        assert h['voronoi_type'] == g['model.bbox_head.voronoi_type']
        assert h['square_cls'] == [g[f'model.bbox_head.square_cls[{i}]'] for i in range(3)]
        assert h['edge_loss_cls'] == [g[f'model.bbox_head.edge_loss_cls[{i}]']
                                      for i in range(g['model.bbox_head.edge_loss_cls.__len__'])]
        assert h['post_process'] == {11: 1.2}
        vt = h['voronoi_thres']
        assert vt['default'] == [g['model.bbox_head.voronoi_thres.default[0]'],
                                 g['model.bbox_head.voronoi_thres.default[1]']]
        ac = h['angle_coder']
        assert ac['angle_version'] == g['model.bbox_head.angle_coder.angle_version']
        assert ac['dual_freq'] == g['model.bbox_head.angle_coder.dual_freq']
        assert ac['num_step'] == g['model.bbox_head.angle_coder.num_step']
        assert ac['thr_mod'] == g['model.bbox_head.angle_coder.thr_mod']

    def test_model_losses(self, cfg, g):
        h = cfg['model']['bbox_head']
        assert h['loss_cls']['gamma'] == g['model.bbox_head.loss_cls.gamma']
        assert h['loss_cls']['alpha'] == g['model.bbox_head.loss_cls.alpha']
        assert h['loss_cls']['loss_weight'] == g['model.bbox_head.loss_cls.loss_weight']
        assert h['loss_cls']['use_sigmoid'] == g['model.bbox_head.loss_cls.use_sigmoid']
        assert h['loss_bbox']['loss_weight'] == g['model.bbox_head.loss_bbox.loss_weight']
        assert h['loss_overlap']['loss_weight'] == g['model.bbox_head.loss_overlap.loss_weight']
        assert h['loss_overlap']['lamb'] == g['model.bbox_head.loss_overlap.lamb']
        assert h['loss_voronoi']['loss_weight'] == g['model.bbox_head.loss_voronoi.loss_weight']
        assert h['loss_bbox_edg']['loss_weight'] == g['model.bbox_head.loss_bbox_edg.loss_weight']
        assert h['loss_ss']['loss_weight'] == g['model.bbox_head.loss_ss.loss_weight']

    def test_test_cfg(self, cfg, g):
        t = cfg['model']['bbox_head']['test_cfg']
        assert t['nms_pre'] == g['model.test_cfg.nms_pre']
        assert t['min_bbox_size'] == g['model.test_cfg.min_bbox_size']
        assert t['score_thr'] == g['model.test_cfg.score_thr']
        assert t['nms']['iou_threshold'] == g['model.test_cfg.nms.iou_threshold']
        assert t['max_per_img'] == g['model.test_cfg.max_per_img']

    def test_dataloaders(self, cfg, g):
        d = cfg['dataset']
        assert d['train']['batch_size'] == g['train_dataloader.batch_size']
        # [plan-deviation] 官方 num_workers=2；jittor 多进程 dataloader 环形缓冲
        # 死锁（见 docs/porting_notes.md），改 0。纯加载性能参数，数值语义不变。
        assert g['train_dataloader.num_workers'] == 2
        assert d['train']['num_workers'] == 0
        assert d['val']['batch_size'] == g['val_dataloader.batch_size']
        assert d['test']['batch_size'] == g['test_dataloader.batch_size']
        assert d['train']['filter_empty_gt'] is True
        # 铁律二 #4：val 指向 trainval
        assert 'trainval' in d['val']['images_dir']
        # pipeline 关键值
        tr = {t['type']: t for t in d['train']['transforms']}
        assert tr['MMRotateRandomFlip']['prob'] == 0.75
        assert tr['MMRotateRandomFlip']['direction'] == \
            ['horizontal', 'vertical', 'diagonal']
        assert tr['Pad']['size_divisor'] == 32
        assert d['train']['point_proportion'] == 1.0
        assert d['train']['hbox_proportion'] == 0.0

    def test_optimizer_scheduler_epochs(self, cfg, g):
        o = cfg['optimizer']
        assert o['type'] == 'AdamW'
        assert o['lr'] == g['optim_wrapper.optimizer.lr']
        assert tuple(o['betas']) == (g['optim_wrapper.optimizer.betas[0]'],
                                     g['optim_wrapper.optimizer.betas[1]'])
        assert o['weight_decay'] == g['optim_wrapper.optimizer.weight_decay']
        assert o['grad_clip']['max_norm'] == g['optim_wrapper.clip_grad.max_norm']
        assert o['grad_clip']['norm_type'] == g['optim_wrapper.clip_grad.norm_type']
        s = cfg['scheduler']
        assert s['start_factor'] == pytest.approx(g['param_scheduler[0].start_factor'])
        assert s['warmup_iters'] == g['param_scheduler[0].end']
        assert s['milestones'] == [g['param_scheduler[1].milestones[0]'],
                                   g['param_scheduler[1].milestones[1]']]
        assert s['gamma'] == g['param_scheduler[1].gamma']
        assert cfg['max_epoch'] == g['train_cfg.max_epochs']
        assert cfg['eval_interval'] == g['train_cfg.val_interval']
        assert cfg['checkpoint_interval'] == 1
        assert cfg['log_interval'] == 50


class TestStage2ConfigParity:
    """stage-2 Jittor config ↔ 官方 rotated-fcos golden 逐值比对（§6.3 勾选表）。"""

    @pytest.fixture(scope='class')
    def cfg(self):
        path = os.path.join(os.path.dirname(__file__), '..', '..',
                            'configs', 'point2rbox_v2',
                            'rotated_fcos_1x_dota_using_pseudo.py')
        return _load_jt_config(path)

    @pytest.fixture(scope='class')
    def g(self):
        return load_golden('config_rotated-fcos-1x-dota-using-pseudo.json')

    def test_backbone_neck(self, cfg, g):
        b = cfg['model']['backbone']
        assert [g[f'model.backbone.out_indices[{i}]'] for i in range(4)] == [0, 1, 2, 3]
        assert b['return_stages'] == ['layer1', 'layer2', 'layer3', 'layer4']
        n = cfg['model']['neck']
        assert n['in_channels'] == [g[f'model.neck.in_channels[{i}]'] for i in range(4)]
        assert n['out_channels'] == g['model.neck.out_channels'] == 512
        assert n['start_level'] == g['model.neck.start_level']
        assert n['num_outs'] == g['model.neck.num_outs']

    def test_head(self, cfg, g):
        h = cfg['model']['roi_heads']
        assert h['num_classes'] == g['model.bbox_head.num_classes']
        assert h['in_channels'] == g['model.bbox_head.in_channels'] == 512
        assert h['feat_channels'] == g['model.bbox_head.feat_channels']
        assert h['stacked_convs'] == g['model.bbox_head.stacked_convs']
        assert h['strides'] == [g[f'model.bbox_head.strides[{i}]'] for i in range(5)]
        assert h['center_sampling'] == g['model.bbox_head.center_sampling']
        assert h['center_sample_radius'] == g['model.bbox_head.center_sample_radius']
        assert h['norm_on_bbox'] == g['model.bbox_head.norm_on_bbox']
        assert h['centerness_on_reg'] == g['model.bbox_head.centerness_on_reg']
        assert h['use_hbbox_loss'] == g['model.bbox_head.use_hbbox_loss']
        assert h['scale_angle'] == g['model.bbox_head.scale_angle']
        assert h['bbox_coder']['angle_version'] == \
            g['model.bbox_head.bbox_coder.angle_version']
        assert h['loss_bbox']['loss_weight'] == g['model.bbox_head.loss_bbox.loss_weight']
        assert h['loss_angle'] is None and g['model.bbox_head.loss_angle'] is None
        assert h['loss_centerness']['loss_weight'] == \
            g['model.bbox_head.loss_centerness.loss_weight']

    def test_optim_and_data(self, cfg, g):
        o = cfg['optimizer']
        # 铁律二 #5：stage-2 wd 就是 0.005
        assert o['weight_decay'] == g['optim_wrapper.optimizer.weight_decay'] == 0.005
        assert o['lr'] == g['optim_wrapper.optimizer.lr']
        assert o['grad_clip']['max_norm'] == g['optim_wrapper.clip_grad.max_norm']
        d = cfg['dataset']
        assert d['train']['batch_size'] == g['train_dataloader.batch_size'] == 4
        assert d['val']['batch_size'] == g['val_dataloader.batch_size'] == 4
        assert 'pseudo_labels.bbox.json' in d['train']['ann_json']
        assert 'pseudo_labels.bbox.json' in g['train_dataloader.dataset.ann_file']
        assert d['train']['weak_supervision'] is False  # 全监督，无 ConvertWeakSupervision
        assert cfg['max_epoch'] == g['train_cfg.max_epochs']
