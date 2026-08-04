"""JDet-WOOD integration smoke tests for every advertised model family."""

from pathlib import Path

import pytest


pytest.importorskip('jittor')


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_CONFIGS = {
    'H2RBox': 'configs/whollywood/h2rbox_obb_r50_adamw_fpn_1x_dota.py',
    'H2RBoxV2P': 'configs/whollywood/h2rbox_v2p_obb_r50_adamw_fpn_1x_dota.py',
    'WhollyWood': 'configs/whollywood/whollywood_obb_r50_adamw_fpn_1x_dota.py',
    'Point2RBox': 'configs/whollywood/point2rbox_obb_r50_adamw_fpn_1x_dota.py',
    'Point2RBoxV2': 'configs/point2rbox_v2/point2rbox_v2_final_fixed.py',
    'Point2RBoxV3': 'configs/point2rbox_v3/point2rbox_v3_1x_dota.py',
}


def test_all_supported_models_are_registered():
    import jdet  # noqa: F401 - importing populates the registries
    from jdet.utils.registry import MODELS

    missing = sorted(set(SUPPORTED_CONFIGS) - set(MODELS._modules))
    assert not missing, f'missing JDet-WOOD model registrations: {missing}'


def test_all_supported_configs_load_and_select_the_expected_model():
    from jdet.config.config import Config

    for model_type, relative_path in SUPPORTED_CONFIGS.items():
        config = Config(str(ROOT / relative_path))
        assert config.model.type == model_type, relative_path
