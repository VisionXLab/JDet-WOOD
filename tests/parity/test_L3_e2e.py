"""L3 端到端：完整 detector（backbone→FPN→head）用真实训练权重的前向 parity。

golden 来源：实验 X（本机 PyTorch baseline）epoch_6 checkpoint 经
tools/convert_torch_ckpt.py 转换；tools 侧 dump 同输入的 torch 前向。
比 20-iter 曲线更强的静态集成证据（权重映射 + 全链路数值同时验证）。
曲线级对照在训练完成后以两边 loss 日志对照补充（PROGRESS 记录）。
"""
import os
import pickle

import numpy as np
import pytest

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden')
CKPT_PKL = '/tmp/claude-0/-root-work-A/971d1b32-0c6b-4dc6-9fc8-3ea4a9f49270/scratchpad/x_ep6.pkl'


@pytest.mark.skipif(not os.path.exists(CKPT_PKL),
                    reason='需要 convert_torch_ckpt.py 产出的 X ckpt（见文件头）')
def test_full_model_forward_parity():
    import sys
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, repo)
    import jittor as jt
    jt.flags.use_cuda = 1
    ns = {}
    cfg_path = os.path.join(repo, 'configs/point2rbox_v2/point2rbox_v2_1x_dota.py')
    exec(compile(open(cfg_path).read(), cfg_path, 'exec'), ns)
    from jdet.utils.registry import build_from_cfg, MODELS
    model = build_from_cfg(ns['model'], MODELS)
    with open(CKPT_PKL, 'rb') as f:
        sd = pickle.load(f)
    model.load_parameters({k: jt.array(v) for k, v in sd.items()})
    model.eval()

    g = np.load(os.path.join(GOLDEN, 'model_forward.npz'))
    with jt.no_grad():
        feat = model.backbone(jt.array(g['x']))
        if model.neck:
            feat = model.neck(feat)
        outs = model.bbox_head.forward(feat)
    jt.flags.use_cuda = 0
    for name, o in zip(['cls', 'bbox', 'angle'],
                       [outs[0][0], outs[1][0], outs[2][0]]):
        got, want = o.numpy(), g[name]
        rel = np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-12)
        # 跨框架 conv/GN 实现差异经 50+ 层累积；3e-3 内视为一致
        assert rel < 3e-3, f'{name} rel L2 = {rel}'
