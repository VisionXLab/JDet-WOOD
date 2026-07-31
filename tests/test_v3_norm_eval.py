"""Regression for Jittor parent-train DFS bypassing ResNet.train()."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import jittor as jt  # noqa: E402
from jittor import nn  # noqa: E402
import jdet.models  # noqa: F401,E402
from jdet.utils.registry import HEADS, MODELS, build_from_cfg  # noqa: E402
from test_v3_detector_smoke import MODEL_CFG  # noqa: E402


@HEADS.register_module()
class _NormEvalDummyHead(nn.Module):
    """Minimal head used to exercise the real FCOS train-mode dispatch."""

    def execute(self, feats, targets):
        return feats


def test_v3_train_preserves_resnet_norm_eval():
    cfg = dict(MODEL_CFG)
    cfg['backbone'] = dict(MODEL_CFG['backbone'])
    cfg['bbox_head'] = {
        k: v for k, v in MODEL_CFG['bbox_head'].items()
        if k != 'loss_cfg_placeholder'
    }
    model = build_from_cfg(cfg, MODELS)

    # Exercise the same parent eval -> train transition used by Runner.
    model.eval()
    model.train()

    bns = [m for m in model.backbone.modules()
           if isinstance(m, nn.BatchNorm)]
    assert bns and all(not m.is_training() for m in bns)

    # frozen_stages=1 freezes layer1, while norm_eval must not stop gradients
    # through affine parameters in later backbone stages.
    assert model.backbone.layer1[0].bn1.weight.is_stop_grad()
    assert not model.backbone.layer2[0].bn1.weight.is_stop_grad()
    assert not model.ted_model.is_training()


def test_stage2_fcos_train_preserves_resnet_norm_eval():
    """Stage-2 uses FCOS, not Point2RBoxV3, so lock its dispatch separately."""
    model = build_from_cfg(
        dict(
            type='FCOS',
            backbone=dict(
                type='Resnet18',
                pretrained=False,
                return_stages=['layer1', 'layer2', 'layer3', 'layer4'],
                frozen_stages=1,
                norm_eval=True),
            neck=None,
            roi_heads=dict(type='_NormEvalDummyHead')),
        MODELS)

    model.eval()
    model.train()

    bns = [m for m in model.backbone.modules()
           if isinstance(m, nn.BatchNorm)]
    assert bns and all(not m.is_training() for m in bns)
    assert model.backbone.layer1[0].bn1.weight.is_stop_grad()
    assert not model.backbone.layer2[0].bn1.weight.is_stop_grad()
