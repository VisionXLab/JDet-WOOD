"""Exact routing test for Point2RBox-v3's epoch-6 switches.

The test proves which branches ``forward_train`` calls at epochs 5 and 6.

No backbone, SAM, TED, or CUDA kernel is executed here.  The production
``Point2RBoxV3.forward_train`` method is exercised with small instrumented
components, so a future refactor cannot silently bypass the switches.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'python'))

import jittor as jt  # noqa: E402
import jdet.models.networks.point2rbox_v3 as p2rv3_mod  # noqa: E402
from jdet.models.networks.point2rbox_v3 import Point2RBoxV3  # noqa: E402


class FakeHead:
    edge_loss_start_epoch = 6
    voronoi_type = 'gaussian-orientation'
    square_cls = []
    epoch = -1

    def __init__(self, calls):
        self.calls = calls

    def predict(self, feat, targets):
        self.calls['predict'] += 1
        return [
            dict(bboxes=t['rboxes'].clone(), labels=t['labels'].clone())
            for t in targets
        ]

    def execute(self, feat):
        return (jt.zeros((1,), dtype='float32'),)

    def loss(self, *args):
        return {'loss_dummy': args[0].sum() * 0}


class Harness:
    """Minimal object that runs the production forward_train implementation."""

    forward_train = Point2RBoxV3.forward_train
    set_epoch = Point2RBoxV3.set_epoch

    def __init__(self):
        self.calls = dict(predict=0, generate=0, edges=0, paste=0, cache=0)
        self.bbox_head = FakeHead(self.calls)
        self.backbone = lambda images: images
        self.neck = None
        self.copy_paste_start_epoch = 6
        self.label_assign_pseudo_label_switch_eopch = 6
        self.copy_paste_cache = None
        self.num_copies = 10
        self.epoch = 0

    def prepare_dual_stream(self, images, targets):
        # Preserve the production contract: original B first, augmented B next.
        aug = []
        for t in targets:
            u = {k: (v.clone() if isinstance(v, jt.Var) else v)
                 for k, v in t.items()}
            u['bids'][:, 0] += len(targets)
            u['bids'][:, 2] = 1
            u['ss'] = ('rot', 0.5)
            aug.append(u)
            t['ss'] = ('rot', 0.5)
        return jt.concat([images, images], dim=0), targets + aug

    def prepare_edges(self, images):
        self.calls['edges'] += 1

    def prepare_copy_paste_step2(self, images, targets):
        self.calls['paste'] += 1

    def generate_pseudo_targets(self, targets, results_list_assist=None):
        self.calls['generate'] += 1
        return [
            dict(bboxes=t['rboxes'].clone(), labels=t['labels'].clone())
            for t in targets
        ]


class DispatchHarness(jt.nn.Module):
    """Regression harness: eval targets may still contain GT ``rboxes``."""

    execute = Point2RBoxV3.execute

    def forward_train(self, images, targets):
        return 'train'

    def forward_test(self, images, targets):
        return 'test'


def make_batch():
    images = jt.zeros((1, 3, 16, 16), dtype='float32')
    targets = [dict(
        rboxes=jt.array(np.float32([[8, 8, 4, 2, 0.1]])),
        labels=jt.array(np.int32([3])),
    )]
    return images, targets


def main():
    dispatch = DispatchHarness()
    dispatch.train()
    assert dispatch(None, [dict(rboxes=None)]) == 'train'
    dispatch.eval()
    assert dispatch(None, [dict(rboxes=None)]) == 'test'
    assert dispatch(None, [dict(filename='patch.png')]) == 'test'

    model = Harness()
    original_cache_fn = p2rv3_mod.get_copy_paste_cache

    def fake_cache(*args, **kwargs):
        model.calls['cache'] += 1
        return SimpleNamespace()

    p2rv3_mod.get_copy_paste_cache = fake_cache
    try:
        # epoch 5: gaussian assist predict + watershed/voronoi pseudo targets;
        # no TED edge branch and no copy-paste cache construction.
        model.set_epoch(5)
        model.forward_train(*make_batch())
        assert model.epoch == model.bbox_head.epoch == 5
        assert model.calls == dict(
            predict=1, generate=1, edges=0, paste=0, cache=0), model.calls

        # epoch 6: direct head.predict replaces generate_pseudo_targets;
        # edge branch runs and a 2B cache is built.
        model.calls = dict(predict=0, generate=0, edges=0, paste=0, cache=0)
        model.bbox_head.calls = model.calls
        model.set_epoch(6)
        model.forward_train(*make_batch())
        assert model.epoch == model.bbox_head.epoch == 6
        assert model.calls == dict(
            predict=1, generate=0, edges=1, paste=0, cache=2), model.calls
        assert len(model.copy_paste_cache) == 2

        # The upstream v3 condition compares B with a 2B cache, so step2 is
        # intentionally dead.  A second call proves we preserved that behavior.
        model.forward_train(*make_batch())
        assert model.calls['paste'] == 0
        assert model.calls['edges'] == 2
        assert model.calls['predict'] == 2
        assert model.calls['generate'] == 0
    finally:
        p2rv3_mod.get_copy_paste_cache = original_cache_fn

    print('PASS: epoch-6 routing (pseudo switch / edge / copy-paste dead code)')


if __name__ == '__main__':
    main()
