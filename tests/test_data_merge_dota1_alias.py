"""Regression for the DOTA-v1.0 dataset-name alias at auto-test merge."""

from pathlib import Path

from jdet.config.constant import get_classes_by_name
from jdet.data.devkits import data_merge as data_merge_module


def test_dota1_alias_reaches_merge(monkeypatch, tmp_path):
    calls = []

    def fake_merge(result_pkl, save_path, final_path, dataset_type):
        calls.append((result_pkl, save_path, final_path, dataset_type))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(data_merge_module, "data_merge", fake_merge)
    monkeypatch.setattr(data_merge_module.os, "system", lambda command: 0)

    data_merge_module.data_merge_result(
        result_pkl="/tmp/test_12.pkl",
        work_dir=str(tmp_path / "work"),
        epoch=12,
        name="point2rbox_v3_1x_dota",
        dataset_type="DOTA1",
        images_dir="/tmp/images",
    )

    assert len(calls) == 1
    assert calls[0][3] == "DOTA1"
    assert get_classes_by_name("DOTA1") == get_classes_by_name("DOTA")
    assert Path("submit_zips").is_dir()
