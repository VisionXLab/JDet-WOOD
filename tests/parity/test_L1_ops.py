"""L1 算子/组件级 parity 测试（CPU 可跑，进 CI）。

golden 来源：
    lr_sequence.npz  <- tools/dump_lr_mmengine.py（p2r-torch 环境）
"""
import os
import tempfile

import numpy as np
import pytest

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden')


class TestLRSchedule:
    """C2：LinearWarmupMultiStepLR 与 mmengine LinearLR+MultiStepLR 逐点相等。"""

    def test_lr_sequence_pointwise(self):
        import jittor as jt
        from jdet.optims.optimizer import AdamW
        from jdet.optims.lr_scheduler import LinearWarmupMultiStepLR

        g = np.load(os.path.join(GOLDEN, 'lr_sequence.npz'))
        golden_lr = g['lr']
        ipe, epochs, base_lr = int(g['iters_per_epoch']), int(g['epochs']), float(g['base_lr'])

        w = jt.ones(2)
        opt = AdamW([w], lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05)
        sched = LinearWarmupMultiStepLR(
            opt, start_factor=float(g['start_factor']), warmup_iters=int(g['warmup_iters']),
            milestones=[int(m) for m in g['milestones']], gamma=float(g['gamma']))

        # 模拟 Runner.train 的调用时序：每 iter 更新前 step(iter, epoch)
        lrs = []
        it = 0
        for e in range(epochs):
            for _ in range(ipe):
                sched.step(it, e, by_epoch=True)
                lrs.append(opt.param_groups[0].get('lr', opt.lr))
                it += 1

        lrs = np.array(lrs, dtype=np.float64)
        assert lrs.shape == golden_lr.shape
        # mmengine 侧是乘法递推，带 ~1e-19 浮点累积误差；1e-9 相对容差即"逐点相等"
        np.testing.assert_allclose(lrs, golden_lr, rtol=1e-9, atol=0)

    def test_lr_key_points(self):
        """独立于 golden 的关键点自检（防 golden 本身 dump 错）。"""
        import jittor as jt
        from jdet.optims.optimizer import AdamW
        from jdet.optims.lr_scheduler import LinearWarmupMultiStepLR

        base = 5e-5
        w = jt.ones(2)
        opt = AdamW([w], lr=base)
        sched = LinearWarmupMultiStepLR(opt, start_factor=1 / 3, warmup_iters=500,
                                        milestones=[8, 11], gamma=0.1)
        cases = [
            (0, 0, base / 3),                          # warmup 起点
            (250, 0, base * (1 / 3 + 2 / 3 * 250 / 499)),  # warmup 中段（分母 W-1=499）
            (499, 0, base),                            # warmup 在第 W-1 iter 到顶（mmengine 语义）
            (500, 0, base),                            # 之后保持
            (5000, 7, base),                           # epoch 7 全量
            (5000, 8, base * 0.1),                     # epoch 8 第一档衰减
            (5000, 11, base * 0.01),                   # epoch 11 第二档
        ]
        for it, ep, expect in cases:
            sched.step(it, ep)
            got = opt.param_groups[0].get('lr', opt.lr)
            assert abs(got - expect) < 1e-15, f'iter={it} epoch={ep}: {got} != {expect}'


class TestPSCCoder:
    """底座 PSCCoder vs mmrotate（官方 config 参数：le90, dual_freq=False, num_step=3, thr_mod=0）。"""

    def test_encode_decode(self):
        import jittor as jt
        from jdet.models.boxes.coder import PSCCoder

        g = np.load(os.path.join(GOLDEN, 'ops_misc.npz'))
        coder = PSCCoder(angle_version='le90', dual_freq=False, num_step=3, thr_mod=0)
        assert coder.encode_size == 3
        enc = coder.encode(jt.array(g['psc_angles']))
        np.testing.assert_allclose(enc.numpy(), g['psc_encoded'], rtol=1e-5, atol=1e-6)
        dec = coder.decode(jt.array(g['psc_encoded']), keepdim=True)
        # 角度按模 π 的角距离比较：±π/2 端点处 atan2(∓0,-1) 的符号翻转会给出
        # +π/2 vs -π/2——le90 下同一朝向（周期 π），不是语义差异
        d = np.abs(dec.numpy() - g['psc_decoded'])
        d = np.minimum(d, np.abs(d - np.pi))
        assert d.max() < 1e-5, f'decode 角距离 max = {d.max()}'


class TestBoxIouRotated:
    def test_iou_vs_mmcv(self):
        import jittor as jt
        from jdet.ops.box_iou_rotated import box_iou_rotated

        g = np.load(os.path.join(GOLDEN, 'ops_misc.npz'))
        iou = box_iou_rotated(jt.array(g['iou_boxes1']), jt.array(g['iou_boxes2']))
        np.testing.assert_allclose(iou.numpy(), g['iou'], rtol=1e-4, atol=1e-5)


class TestNMSRotated:
    def test_keep_set_vs_mmcv(self):
        """keep 集合与 mmcv 一致（iou_threshold=0.1，官方 test_cfg 值）。

        已知语义差异（M4 head 层需适配）：jdet 返回原始下标顺序，
        mmcv 返回按分数降序——影响 max_per_img 截断，此处只比集合。"""
        import jittor as jt
        from jdet.ops.nms_rotated import nms_rotated

        g = np.load(os.path.join(GOLDEN, 'ops_misc.npz'))
        keep = nms_rotated(jt.array(g['nms_boxes']), jt.array(g['nms_scores']), 0.1)
        assert sorted(keep.numpy().tolist()) == sorted(g['nms_keep'].tolist())


class TestRoIAlignRotated:
    """B 问的角度约定：jdet ROIAlignRotated（无 clockwise 参数）vs mmcv clockwise=True。"""

    def _run(self, osize):
        import jittor as jt
        from jdet.ops.roi_align_rotated import ROIAlignRotated

        g = np.load(os.path.join(GOLDEN, 'ops_misc.npz'))
        jt.flags.use_cuda = 1  # jdet 该算子只有 CUDA 实现
        # mmcv(clockwise=True) 等价参数：aligned=True + clockwise=True（=JDet 原生方向）
        op = ROIAlignRotated(osize, spatial_scale=1.0, sampling_ratio=2,
                             aligned=True, clockwise=True)
        out = op(jt.array(g['ra_feat']), jt.array(g['ra_rois']))
        jt.flags.use_cuda = 0
        return out.numpy(), g[f'ra_out_{osize}']

    def test_out7_matches_clockwise_true(self):
        got, want = self._run(7)
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

    def test_out49_matches_clockwise_true(self):
        got, want = self._run(49)
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


class TestLinalg2x2:
    """M3 地基：2×2 eigh/solve 闭式解 vs torch.linalg（含 w==h 等退化用例）。"""

    @pytest.fixture(scope='class')
    def g(self):
        return np.load(os.path.join(GOLDEN, 'linalg2x2.npz'))

    def test_eigh_eigvals(self, g):
        import jittor as jt
        from jdet.ops.linalg2x2 import eigh_2x2

        L, V = eigh_2x2(jt.array(g['sigma']))
        want = g['eigvals']
        scale = np.abs(want).max(axis=-1, keepdims=True)  # 每个矩阵按自身量级
        assert (np.abs(L.numpy() - want) / (scale + 1e-12)).max() < 1e-5

    def test_eigh_eigvecs_reconstruct(self, g):
        """特征向量符号任意 → 比重建 V diag(L) V^T == sigma 以及正交性。"""
        import jittor as jt
        from jdet.ops.linalg2x2 import eigh_2x2, diag_embed_2x2

        sigma = jt.array(g['sigma'])
        L, V = eigh_2x2(sigma)
        recon = jt.matmul(jt.matmul(V, diag_embed_2x2(L)), V.transpose(0, 2, 1))
        diff = np.abs(recon.numpy() - g['sigma'])
        scale = np.abs(g['sigma']).reshape(len(diff), -1).max(1)[:, None, None]
        assert (diff / (scale + 1e-12)).max() < 1e-5
        # 正交性
        vtv = jt.matmul(V.transpose(0, 2, 1), V).numpy()
        assert np.abs(vtv - np.eye(2)).max() < 1e-5
        # 与 torch 特征向量对齐（至符号）：|<v_i, v_i_torch>| == 1。
        # 仅对非简并矩阵有意义——特征值重复时特征基任意（torch 选 I，我们选旋转基），
        # 两者都对，只保证上面的重建/正交即可
        lv = g['eigvals']
        nondeg = (lv[:, 1] - lv[:, 0]) / (np.abs(lv[:, 1]) + 1e-12) > 1e-4
        dots = np.abs(np.einsum('nij,nij->nj', V.numpy(), g['eigvecs']))
        assert np.abs(dots[nondeg] - 1).max() < 1e-4

    def test_eigh_eigval_grad(self, g):
        import jittor as jt
        from jdet.ops.linalg2x2 import eigh_2x2

        sigma = jt.array(g['sigma'])
        L, V = eigh_2x2(sigma)
        grad = jt.grad((L * jt.array(g['eig_wgt'])).sum(), sigma).numpy()
        assert np.isfinite(grad).all(), '梯度出现 NaN/Inf（退化情形保护失效）'
        want = g['eigval_grad']
        scale = np.abs(want).reshape(len(want), -1).max(1)[:, None, None] + 1e-12
        # 特征值重复时 dλi/dA 是子梯度、随特征基选择而变（非简并部分才唯一）；
        # 简并用例只验证基无关的不变量：d(λ1+λ2)/dA = I（trace 的梯度）
        lv = g['eigvals']
        nondeg = (lv[:, 1] - lv[:, 0]) / (np.abs(lv[:, 1]) + 1e-12) > 1e-4
        assert (np.abs(grad - want) / scale)[nondeg].max() < 1e-3
        gsum = jt.grad(L.sum(), sigma).numpy()
        assert np.abs(gsum[~nondeg] - np.eye(2)).max() < 1e-5

    def test_eigh_grad_finite_at_exact_isotropy(self):
        """w==h 完全退化点：torch 在此处梯度也未必稳定，只要求我们有限、无 NaN。"""
        import jittor as jt
        from jdet.ops.linalg2x2 import eigh_2x2

        sigma = jt.array(np.stack([np.eye(2, dtype=np.float32) * 64.0] * 3))
        L, V = eigh_2x2(sigma)
        grad = jt.grad(L.sum(), sigma).numpy()
        assert np.isfinite(grad).all()

    def test_solve(self, g):
        import jittor as jt
        from jdet.ops.linalg2x2 import solve_2x2

        A, B = jt.array(g['sigma']), jt.array(g['solve_b'])
        X = solve_2x2(A, B)
        want = g['solve_x']
        scale = np.abs(want).reshape(len(want), -1).max(1)[:, None, None] + 1e-12
        assert (np.abs(X.numpy() - want) / scale).max() < 1e-4

    def test_solve_grad(self, g):
        import jittor as jt
        from jdet.ops.linalg2x2 import solve_2x2

        A, B = jt.array(g['sigma']), jt.array(g['solve_b'])
        grad = jt.grad(solve_2x2(A, B).sum(), A).numpy()
        assert np.isfinite(grad).all()
        want = g['solve_grad']
        scale = np.abs(want).reshape(len(want), -1).max(1)[:, None, None] + 1e-12
        assert (np.abs(grad - want) / scale).max() < 1e-3


class TestClipGrad:
    """C1：AdamW(grad_clip=dict(max_norm=35, norm_type=2)) 裁剪后全局 L2 范数 == 35。"""

    def test_clip_grad_norm_35(self):
        import jittor as jt
        from jdet.optims.optimizer import AdamW

        w1 = jt.ones(10)
        w2 = jt.ones(5)
        opt = AdamW([w1, w2], lr=5e-5, grad_clip=dict(max_norm=35, norm_type=2))
        # loss = 100*sum(w1) + 200*sum(w2) → grad w1 全 100、w2 全 200
        # 全局 L2 = sqrt(10*100^2 + 5*200^2) = sqrt(300000) ≈ 547.7 > 35 → 触发裁剪
        loss = (w1 * 100).sum() + (w2 * 200).sum()
        opt.pre_step(loss)
        sq = 0.0
        for pg in opt.param_groups:
            for p, g in zip(pg['params'], pg['grads']):
                if p.is_stop_grad():
                    continue
                sq += float((g * g).sum())
        assert abs(np.sqrt(sq) - 35.0) < 1e-3, f'clipped norm = {np.sqrt(sq)}'
        opt.zero_grad()

    def test_no_clip_when_below(self):
        import jittor as jt
        from jdet.optims.optimizer import AdamW

        w = jt.ones(4)
        opt = AdamW([w], lr=5e-5, grad_clip=dict(max_norm=35, norm_type=2))
        loss = w.sum()  # grad 全 1，L2 = 2 < 35 → 不裁剪
        opt.pre_step(loss)
        g = opt.param_groups[0]['grads'][0]
        np.testing.assert_allclose(g.numpy(), np.ones(4), rtol=1e-6)
        opt.zero_grad()


class TestResizedCropAA:
    """torchvision resized_crop 的越界补零与 antialias 语义。"""

    def test_out_of_bounds_crop(self):
        import jittor as jt
        from jdet.models.networks.point2rbox_v2 import _resized_crop_aa

        image = jt.arange(9).float32().reshape(1, 1, 3, 3)
        got = _resized_crop_aa(image, 5, 5, 3, 3).numpy()
        # torchvision 0.17 functional.resized_crop golden.
        want = np.array([[[[1.7142857, 2.3333333, 0.0],
                           [4.3333333, 4.1481481, 0.0],
                           [0.0, 0.0, 0.0]]]], dtype=np.float32)
        np.testing.assert_allclose(got, want, rtol=2e-6, atol=2e-6)


class TestDiffIoURotated:
    """M7：diff_iou_rotated_2d / RotatedIoULoss vs mmcv golden（GPU）。"""

    def test_iou_and_loss(self):
        import jittor as jt
        from jdet.ops.diff_iou_rotated import diff_iou_rotated_2d
        from jdet.models.losses.rotated_iou_loss import RotatedIoULoss

        g = np.load(os.path.join(GOLDEN, 'riou.npz'))
        jt.flags.use_cuda = 1
        b1, b2 = jt.array(g['b1']), jt.array(g['b2'])
        iou = diff_iou_rotated_2d(b1.unsqueeze(0), b2.unsqueeze(0)).squeeze(0)
        np.testing.assert_allclose(iou.numpy(), g['iou'], rtol=1e-4, atol=1e-5)

        gi = jt.grad(iou.sum(), b1).numpy()
        loss = RotatedIoULoss(loss_weight=1.0)(b1, b2)
        rel = abs(float(loss.item()) - float(g['loss'])) / abs(float(g['loss']))
        gl = jt.grad(loss, b1).numpy()
        jt.flags.use_cuda = 0
        assert rel < 1e-4, f'loss rel = {rel}'
        # 梯度逐行比较；row 0 是完全重合框（IoU=1，重复顶点的子梯度分配任意，
        # mmcv 的 CUDA 排序与本实现在重复点间分配不同——数学上均为有效子梯度），
        # 只要求有限；其余行 rel<1e-3
        for grad, want in [(gi, g['iou_grad']), (gl, g['loss_grad'])]:
            assert np.isfinite(grad).all()
            per = np.linalg.norm(grad - want, axis=1) \
                / (np.linalg.norm(want, axis=1) + 1e-9)
            assert per[1:].max() < 1e-3, f'nondegenerate max rel = {per[1:].max()}'


class TestOptimizerCheckpoint:
    def test_adamw_resume_preserves_moments(self):
        """A resumed second update must equal two uninterrupted updates."""
        import jittor as jt
        from jdet.optims.optimizer import AdamW

        p = jt.array(np.asarray([1.5, -0.7], np.float32))
        opt = AdamW([p], lr=3e-4, betas=(0.9, 0.999),
                    weight_decay=0.05)
        opt.step(((p - jt.array([0.2, -0.1])) ** 2).sum())
        p_after_one = p.numpy().copy()

        with tempfile.NamedTemporaryFile(suffix='.pkl') as f:
            jt.save(opt.parameters(), f.name)
            saved_state = jt.load(f.name)

        opt.step(((p - jt.array([-0.3, 0.4])) ** 2).sum())
        expected = p.numpy().copy()

        resumed_p = jt.array(p_after_one)
        resumed = AdamW([resumed_p], lr=3e-4, betas=(0.9, 0.999),
                        weight_decay=0.05)
        resumed.load_parameters(saved_state)
        resumed.step(((resumed_p - jt.array([-0.3, 0.4])) ** 2).sum())
        np.testing.assert_allclose(resumed_p.numpy(), expected,
                                   rtol=1e-7, atol=1e-8)
        assert resumed.n_step == opt.n_step == 2
