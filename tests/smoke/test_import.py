"""冒烟：核心包可 import、numpy 版本未被顶到 2.x（docs/environment.md 坑 #1）。"""


def test_numpy_version():
    import numpy
    major = int(numpy.__version__.split('.')[0])
    assert major < 2, (
        f'numpy=={numpy.__version__}：jittor 1.3.8.5 + numpy>=2 会静默产出错误数值，'
        '必须 pip install numpy==1.26.4（见 docs/environment.md 坑 #1）')


def test_jittor_numerics():
    import jittor as jt
    a = jt.float32([1, 2, 3])
    assert (a + a).numpy().tolist() == [2., 4., 6.], \
        'jittor elementwise 输出错误——大概率是 numpy 2.x 兼容问题'


def test_jdet_imports():
    import jdet  # noqa: F401
    from jdet.optims.optimizer import AdamW  # noqa: F401
    from jdet.optims.lr_scheduler import LinearWarmupMultiStepLR  # noqa: F401
    from jdet.runner import Runner  # noqa: F401


def test_resnet_norm_eval_keeps_affine_gradients():
    """Freeze BN statistics without freezing non-frozen affine weights."""
    from jdet.models.backbones.resnet import Resnet50
    from jittor import nn

    model = Resnet50(pretrained=False, frozen_stages=1, norm_eval=True)
    model.train()
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm)]
    assert bns and all(not m.is_training() for m in bns)
    assert model.layer1[0].bn1.weight.is_stop_grad()
    assert not model.layer2[0].bn1.weight.is_stop_grad()
