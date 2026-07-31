"""L2 模块级 parity（M2.5 复用件核对起步）。

golden 来源：
    gdloss_gwd.npz  <- tools/dump_golden_gdloss.py（p2r-torch 环境，mmrotate 实现）

容差（PLAN §7）：前向 rtol 1e-4，梯度 rtol 1e-3（梯度必须比——防 in-place 静默断梯度）。
"""
import os

import numpy as np
import pytest

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden')


def _assert_close(a, b, rtol, name=''):
    """rtol 逐元素相对容差 + 按张量量级缩放的绝对容差。

    纯逐元素相对误差对「代数上恰好为 0、浮点上有舍入」的元素会爆炸
    （例：w==h 时 sigma 非对角元 golden=0.0、jittor=1.3e-5，而矩阵量级 1e3），
    因此 atol = rtol * max|golden|，即把近零元素的误差换算到张量量级上衡量。"""
    scale = float(np.max(np.abs(b))) or 1.0
    np.testing.assert_allclose(a, b, rtol=rtol, atol=rtol * scale, err_msg=name)


def _rel_err(a, b):
    denom = np.maximum(np.abs(b), 1e-12)
    return np.max(np.abs(a - b) / denom)


class TestGDLossGWD:
    """底座 gaussian_dist_loss.py vs mmrotate 官方（含退化用例：w==h、±π/2、极小框、大长宽比）。"""

    @pytest.fixture(scope='class')
    def golden(self):
        return np.load(os.path.join(GOLDEN, 'gdloss_gwd.npz'))

    def test_xy_wh_r_2_xy_sigma_forward(self, golden):
        import jittor as jt
        from jdet.models.losses.gaussian_dist_loss import xy_wh_r_2_xy_sigma

        pred = jt.array(golden['pred'])
        xy, sigma = xy_wh_r_2_xy_sigma(pred)
        _assert_close(xy.numpy(), golden['xy'], 1e-4, 'xy')
        _assert_close(sigma.numpy(), golden['sigma'], 1e-4, 'sigma')

    def test_xy_wh_r_2_xy_sigma_grad(self, golden):
        import jittor as jt
        from jdet.models.losses.gaussian_dist_loss import xy_wh_r_2_xy_sigma

        pred = jt.array(golden['pred'])
        xy, sigma = xy_wh_r_2_xy_sigma(pred)
        g = jt.grad(xy.sum() + sigma.sum(), pred)
        _assert_close(g.numpy(), golden['xy_sigma_grad'], 1e-3, 'xy_sigma_grad')

    def test_gwd_loss_forward(self, golden):
        import jittor as jt
        from jdet.models.losses.gaussian_dist_loss import GDLoss

        loss_fn = GDLoss(loss_type='gwd', loss_weight=5.0)
        loss = loss_fn(jt.array(golden['pred']), jt.array(golden['target']))
        rel = abs(float(loss.item()) - float(golden['gwd_loss'])) / abs(float(golden['gwd_loss']))
        assert rel < 1e-4, f'forward rel err = {rel}'

    def test_gwd_loss_grad(self, golden):
        import jittor as jt
        from jdet.models.losses.gaussian_dist_loss import GDLoss

        loss_fn = GDLoss(loss_type='gwd', loss_weight=5.0)
        pred = jt.array(golden['pred'])
        loss = loss_fn(pred, jt.array(golden['target']))
        g = jt.grad(loss, pred)
        gn = g.numpy()
        assert np.abs(gn).sum() > 0, '梯度全 0——疑似 in-place 断梯度'
        _assert_close(gn, golden['gwd_grad'], 1e-3, 'gwd_grad')

    def test_gwd_loss_weighted(self, golden):
        """带 weight + avg_factor 的路径（head 实际调用形态）：前向 + 梯度。"""
        import jittor as jt
        from jdet.models.losses.gaussian_dist_loss import GDLoss

        loss_fn = GDLoss(loss_type='gwd', loss_weight=5.0)
        pred = jt.array(golden['pred'])
        loss = loss_fn(pred, jt.array(golden['target']),
                       weight=jt.array(golden['weight']),
                       avg_factor=float(golden['avg_factor']))
        rel = abs(float(loss.item()) - float(golden['gwd_loss_weighted'])) \
            / abs(float(golden['gwd_loss_weighted']))
        assert rel < 1e-4, f'weighted forward rel err = {rel}'
        g = jt.grad(loss, pred)
        _assert_close(g.numpy(), golden['gwd_grad_weighted'], 1e-3, 'gwd_grad_weighted')


class TestGaussianOverlapLoss:
    """M3 §1：GaussianOverlapLoss vs mmrotate（官方参数 w=10,lamb=0 + 默认 lamb 路径）。"""

    @pytest.fixture(scope='class')
    def g(self):
        return np.load(os.path.join(GOLDEN, 'p2rv2_loss.npz'))

    def _run(self, g, loss_weight, lamb, tag):
        import jittor as jt
        from jdet.models.losses.point2rbox_v2_loss import GaussianOverlapLoss

        mu = jt.array(g['gol_mu'])
        sigma = jt.array(g['gol_sigma'])
        loss = GaussianOverlapLoss(loss_weight=loss_weight, lamb=lamb)((mu, sigma))
        rel = abs(float(loss.item()) - float(g[f'gol_{tag}_loss'])) \
            / abs(float(g[f'gol_{tag}_loss']))
        assert rel < 1e-4, f'{tag} forward rel err = {rel}'
        gm, gs = jt.grad(loss, [mu, sigma])
        assert np.abs(gm.numpy()).sum() > 0, 'mu 梯度全 0'
        _assert_close(gm.numpy(), g[f'gol_{tag}_mu_grad'], 1e-3, f'{tag}_mu_grad')
        _assert_close(gs.numpy(), g[f'gol_{tag}_sigma_grad'], 1e-3, f'{tag}_sigma_grad')

    def test_official_cfg(self, g):
        self._run(g, 10.0, 0, 'cfg')

    def test_default_lamb(self, g):
        self._run(g, 1.0, 1e-4, 'lamb')


class TestGwdSigmaLoss:
    def test_forward_grad(self):
        import jittor as jt
        from jdet.models.losses.point2rbox_v2_loss import gwd_sigma_loss

        g = np.load(os.path.join(GOLDEN, 'p2rv2_loss.npz'))
        a = jt.array(g['gws_a'])
        loss = gwd_sigma_loss(a, jt.array(g['gws_b']), reduction='mean')
        rel = abs(float(loss.item()) - float(g['gws_loss'])) / abs(float(g['gws_loss']))
        assert rel < 1e-4, f'forward rel err = {rel}'
        grad = jt.grad(loss, a)
        _assert_close(grad.numpy(), g['gws_grad'], 1e-3, 'gws_grad')


class TestVoronoiWatershedLoss:
    """M3 §2：官方路径 voronoi='standard', w=5.0（watershed 走 cv2，两边共享）。"""

    def test_forward_grad_and_markers(self):
        import jittor as jt
        from jdet.models.losses.point2rbox_v2_loss import VoronoiWatershedLoss

        g = np.load(os.path.join(GOLDEN, 'p2rv2_loss.npz'))
        loss_fn = VoronoiWatershedLoss(loss_weight=5.0)
        mu = jt.array(g['vws_mu'])
        sigma = jt.array(g['vws_sigma'])
        loss = loss_fn((mu, sigma), jt.array(g['vws_label']), jt.array(g['vws_image']),
                       jt.array(g['vws_pos']), jt.array(g['vws_neg']),
                       voronoi='standard')
        # markers（watershed 结果）应逐像素一致（输入图完全相同时 cv2 确定性）
        markers = loss_fn.vis[1].numpy()
        mismatch = (markers != g['vws_markers']).mean()
        assert mismatch < 0.001, f'markers 不一致像素比例 = {mismatch}'
        rel = abs(float(loss.item()) - float(g['vws_loss'])) / abs(float(g['vws_loss']))
        assert rel < 1e-3, f'forward rel err = {rel}'
        gs = jt.grad(loss, sigma).numpy()
        assert np.abs(gs).sum() > 0, 'sigma 梯度全 0'
        # w==h 各向同性行：eigh 特征基任意 → 逐特征值子梯度在 a/c 间的分配不唯一
        # （torch 选 I 基），只比基无关的 trace；非简并行逐位比
        sg = g['vws_sigma']
        tr = sg[:, 0, 0] + sg[:, 1, 1]
        disc = np.sqrt(((sg[:, 0, 0] - sg[:, 1, 1]) / 2) ** 2 + sg[:, 0, 1] ** 2)
        nondeg = disc / (tr / 2) > 1e-3
        want = g['vws_sigma_grad']
        _assert_close(gs[nondeg], want[nondeg], 1e-2, 'vws_sigma_grad(nondeg)')
        np.testing.assert_allclose(gs[~nondeg, 0, 0] + gs[~nondeg, 1, 1],
                                   want[~nondeg, 0, 0] + want[~nondeg, 1, 1],
                                   rtol=1e-2, err_msg='vws_sigma_grad(trace)')


class TestEdgeLoss:
    def test_forward_grad(self):
        import jittor as jt
        from jdet.models.losses.point2rbox_v2_loss import EdgeLoss

        g = np.load(os.path.join(GOLDEN, 'p2rv2_loss.npz'))
        jt.flags.use_cuda = 1  # RoIAlignRotated 只有 CUDA 实现
        loss_fn = EdgeLoss(loss_weight=0.3)
        b = jt.array(g['edge_boxes'])
        loss = loss_fn([b], jt.array(g['edge_map']))
        rel = abs(float(loss.item()) - float(g['edge_loss'])) / abs(float(g['edge_loss']))
        grad = jt.grad(loss, b)
        jt.flags.use_cuda = 0
        assert rel < 1e-3, f'forward rel err = {rel}'
        assert np.abs(grad.numpy()).sum() > 0, '梯度全 0'
        _assert_close(grad.numpy(), g['edge_grad'], 1e-2, 'edge_grad')


class TestConsistencyLoss:
    def _run(self, aug_type, aug_val):
        import jittor as jt
        from jdet.models.losses.point2rbox_v2_loss import Point2RBoxV2ConsistencyLoss

        g = np.load(os.path.join(GOLDEN, 'p2rv2_loss.npz'))
        loss_fn = Point2RBoxV2ConsistencyLoss(loss_weight=1.0)
        go = jt.array(g['con_gaus_o'])
        gt_ = jt.array(g['con_gaus_t'])
        ao = jt.array(g['con_ang_o'])
        at = jt.array(g['con_ang_t'])
        loss = loss_fn((go, ao), (gt_, at), jt.array(g['con_sq']), aug_type, aug_val)
        want = float(g[f'con_{aug_type}_loss'])
        rel = abs(float(loss.item()) - want) / abs(want)
        assert rel < 1e-4, f'{aug_type} forward rel err = {rel}'
        ggo, gao = jt.grad(loss, [go, ao])
        assert np.abs(ggo.numpy()).sum() > 0
        _assert_close(ggo.numpy(), g[f'con_{aug_type}_go_grad'], 1e-3, f'{aug_type}_go_grad')
        _assert_close(gao.numpy(), g[f'con_{aug_type}_ao_grad'], 1e-3, f'{aug_type}_ao_grad')

    def test_rot(self):
        self._run('rot', 0.6)

    def test_flp(self):
        self._run('flp', 0.0)

    def test_sca(self):
        self._run('sca', 1.3)

    def test_angle_keepdim_masking(self):
        """v3 的 (N,1) angle 保持 torch 沿第 0 维布尔索引语义。"""
        import jittor as jt
        from jdet.models.losses.point2rbox_v2_loss import (
            Point2RBoxV2ConsistencyLoss)

        n = 4
        eye = np.eye(2, dtype=np.float32)[None].repeat(n, axis=0)
        ori_gaus = jt.array(eye)
        trs_gaus = jt.array(eye)
        ori_angle = jt.zeros((n, 1))
        trs_angle = jt.array(
            np.array([[0.02], [0.20], [-0.04], [0.30]], np.float32))
        square_mask = jt.array(np.array([False, True, False, True]))

        loss_fn = Point2RBoxV2ConsistencyLoss()
        loss = loss_fn(
            (ori_gaus, ori_angle), (trs_gaus, trs_angle),
            square_mask, 'sca', 1.0)
        baseline = loss_fn(
            (ori_gaus, ori_angle), (trs_gaus, jt.zeros((n, 1))),
            square_mask, 'sca', 1.0)

        # torch: smooth_l1([0.02, -0.04], beta=.1).sum() / 2
        expected = (0.5 * 0.02 ** 2 / 0.1
                    + 0.5 * 0.04 ** 2 / 0.1) / 2
        assert abs(float((loss - baseline).item()) - expected) < 1e-6


class TestTED:
    """third_parties/ted 移植：加载转换权重后 4 个输出与 torch 版逐位对齐。"""

    def test_forward(self):
        import pickle
        import jittor as jt
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from third_parties.ted.ted import TED

        g = np.load(os.path.join(GOLDEN, 'ted_forward.npz'))
        model = TED()
        pkl = os.path.join(os.path.dirname(__file__), '..', '..',
                           'third_parties', 'ted', 'ted.pkl')
        with open(pkl, 'rb') as f:
            sd = pickle.load(f)
        model.load_parameters({k: jt.array(v) for k, v in sd.items()})
        model.eval()
        with jt.no_grad():
            outs = model(jt.array(g['x']))
        for i, o in enumerate(outs):
            want = g[f'out{i}']
            _assert_close(o.numpy(), want, 1e-4, f'ted_out{i}')


class TestHeadParity:
    """M4 验收：固定权重+输入下 head 的 loss_dict 各项 rel<1e-3（epoch=1）。

    梯度比对分两档：GPU 松验（共卡时 cudnn 反向算法漂移，实测 rel_l2 偶发到 ~4%、
    逐元素违约可达 20%+，只兜 O(1) 级断链/错链 bug）+ CPU 紧验（确定性算术，
    实测 rel_l2 2.7e-6，阈 1e-4——真正的数值验收）。"""

    def test_loss_dict(self):
        import jittor as jt
        try:
            jt.flags.use_cuda = 1
            self._run_gpu()
            jt.flags.use_cuda = 0
            self._run_cpu_grad()
        finally:
            jt.flags.use_cuda = 0

    def _build(self):
        import jittor as jt
        from jdet.models.roi_heads.point2rbox_v2_head import Point2RBoxV2Head

        g = np.load(os.path.join(GOLDEN, 'head_parity.npz'))
        head = Point2RBoxV2Head(
            num_classes=15, in_channels=128, feat_channels=128, strides=[8],
            square_cls=[1, 9, 11],
            edge_loss_cls=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13],
            post_process={11: 1.2},
            voronoi_type='standard',
            voronoi_thres=dict(default=[0.994, 0.005],
                               override=(([2, 11], [0.999, 0.6]),
                                         ([7, 8, 10, 14], [0.95, 0.005]))),
            loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0),
            loss_overlap=dict(type='GaussianOverlapLoss', loss_weight=10.0, lamb=0))
        head.epoch = 1
        head.train()
        head.load_parameters({k[3:]: jt.array(g[k])
                              for k in g.files if k.startswith('w__')})

        head.images = jt.array(g['images'])
        targets = []
        for i in range(4):
            targets.append(dict(
                rboxes=jt.array(g[f'gt{i}_rb']),
                labels=jt.array(g[f'gt{i}_lb'].astype(np.int32)),
                bids=jt.array(g[f'gt{i}_bid'].astype(np.int32)),
                ss=('rot', float(g['ss_val']))))

        feat = jt.array(g['feat'])
        losses = head.loss([feat], targets)
        total = sum(v.sum() for v in losses.values())
        grad = jt.grad(total, feat).numpy()
        return g, losses, grad

    def _check_losses(self, g, losses):
        for k, v in losses.items():
            got = float(v.sum().item())
            want = float(g[f'loss_{k}'])
            if abs(want) < 1e-9:
                assert abs(got) < 1e-6, f'{k}: got {got}, want 0'
            else:
                rel = abs(got - want) / abs(want)
                assert rel < 1e-3, f'{k}: got {got}, want {want}, rel {rel}'

    def _run_gpu(self):
        g, losses, grad = self._build()
        self._check_losses(g, losses)
        assert np.isfinite(grad).all()
        want = g['feat_grad']
        rel_l2 = np.linalg.norm(grad - want) / (np.linalg.norm(want) + 1e-12)
        assert rel_l2 < 5e-2, f'[GPU] feat_grad 整体相对 L2 = {rel_l2}'

    def _run_cpu_grad(self):
        g, losses, grad = self._build()
        self._check_losses(g, losses)
        want = g['feat_grad']
        rel_l2 = np.linalg.norm(grad - want) / (np.linalg.norm(want) + 1e-12)
        assert rel_l2 < 1e-4, f'[CPU] feat_grad 整体相对 L2 = {rel_l2}'
        scale = np.abs(want).max()
        viol = (np.abs(grad - want)
                > 1e-3 * np.maximum(np.abs(want), scale * 1e-3)).mean()
        assert viol < 1e-4, f'[CPU] feat_grad 逐元素违约比例 = {viol}'


class TestFCOSHeadParity:
    """M7 验收：stage-2 RotatedFCOSHead 固定权重+输入下与 mmrotate 官方对齐。

    golden: tools/dump_golden_fcos_head.py（配置＝官方 rotated-fcos-1x-dota-
    using-pseudo.py 的 bbox_head，仅缩通道 + conv_reg bias 抬正防退化框；
    见脚本头注释）。梯度 GPU 松验 + CPU 紧验（同 TestHeadParity 的理由）。
    """

    def test_forward_and_loss(self):
        import jittor as jt
        try:
            jt.flags.use_cuda = 1
            self._run(gpu=True)
            jt.flags.use_cuda = 0
            self._run(gpu=False)
        finally:
            jt.flags.use_cuda = 0  # 失败也要复位，避免污染后续 CPU 测试

    def _run(self, gpu):
        import jittor as jt
        from jdet.models.roi_heads.rotated_fcos_head import RotatedFCOSHead

        g = np.load(os.path.join(GOLDEN, 'fcos_head_parity.npz'))
        head = RotatedFCOSHead(
            num_classes=15, in_channels=64, feat_channels=64, stacked_convs=2,
            strides=[8, 16, 32, 64, 128],
            center_sampling=True, center_sample_radius=1.5,
            norm_on_bbox=True, centerness_on_reg=True,
            use_hbbox_loss=False, scale_angle=True,
            bbox_coder=dict(type='DistanceAnglePointCoder',
                            angle_version='le90'),
            loss_cls=dict(type='MMDetFocalLoss', use_sigmoid=True, gamma=2.0,
                          alpha=0.25, loss_weight=1.0),
            loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
            loss_angle=None,
            loss_centerness=dict(type='MMDetCrossEntropyLoss',
                                 use_sigmoid=True, loss_weight=1.0))
        head.train()
        head.load_parameters({k[3:]: jt.array(g[k])
                              for k in g.files if k.startswith('w__')})

        feats = [jt.array(g[f'feat{i}']) for i in range(5)]
        targets = [dict(rboxes=g[f'gt{i}_rb'],
                        labels=jt.array(g[f'gt{i}_lb'].astype(np.int32)))
                   for i in range(2)]

        # 前向输出逐层对齐（比 loss 更细粒度）
        outs = head.forward(feats)
        for name, group in zip(('cls', 'bbox', 'angle', 'ctr'), outs):
            for lvl, t in enumerate(group):
                got = t.numpy()
                want = g[f'out_{name}{lvl}']
                rel_l2 = (np.linalg.norm(got - want)
                          / (np.linalg.norm(want) + 1e-12))
                assert rel_l2 < 1e-3, f'out_{name}{lvl}: rel L2 = {rel_l2}'

        losses = head.loss_by_feat(*outs, targets)
        assert set(losses) == {'loss_cls', 'loss_bbox', 'loss_centerness'}
        for k, v in losses.items():
            got = float(v.sum().item())
            want = float(g[f'loss_{k}'])
            rel = abs(got - want) / max(abs(want), 1e-9)
            assert rel < 1e-3, f'{k}: got {got}, want {want}, rel {rel}'

        # 梯度回传到各层 feat（golden 是三项 loss 之和的梯度）
        total = sum(v.sum() for v in losses.values())
        grads = jt.grad(total, feats)
        # GPU 共卡 cudnn 漂移只做 5e-2 松验；CPU 全和阈 5e-3：
        # loss_bbox 反向经 mmcv 的 shoelace（原始图像坐标，大数相消），
        # golden 梯度自带 ~1e-3 噪声底（TF32 关闭后实测 loss_bbox 部分
        # rel 0.7-1.3e-3、loss_cls+ctr 部分 1e-6，见 porting_notes.md），
        # 与我侧中心化坐标实现的差即该噪声（ops/diff_iou_rotated.py 注释）
        grad_tol = 5e-2 if gpu else 5e-3
        for i, gr in enumerate(grads):
            got = gr.numpy()
            want = g[f'feat{i}_grad']
            assert np.isfinite(got).all()
            rel_l2 = (np.linalg.norm(got - want)
                      / (np.linalg.norm(want) + 1e-12))
            assert rel_l2 < grad_tol, \
                f'[{"GPU" if gpu else "CPU"}] feat{i}_grad: rel L2 = {rel_l2}'
        if not gpu:
            # 紧验走无 IoU 噪声的支路：loss_cls+loss_centerness 的梯度
            # 应与 torch 在 1e-4 内逐层吻合（实锤断链/错链类 bug 的防线）
            grads2 = jt.grad(losses['loss_cls'].sum()
                             + losses['loss_centerness'].sum(), feats)
            for i, gr in enumerate(grads2):
                got = gr.numpy()
                want = g[f'feat{i}_grad_clsctr']
                rel_l2 = (np.linalg.norm(got - want)
                          / (np.linalg.norm(want) + 1e-12))
                assert rel_l2 < 1e-4, \
                    f'[CPU] feat{i}_grad_clsctr: rel L2 = {rel_l2}'
