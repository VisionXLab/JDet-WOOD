"""Tier-2 dataset config inventory and upstream byte-parity."""

from pathlib import Path


ROOT = Path(__file__).parents[2]
MINE = ROOT / "configs/_base_/datasets"
REF = Path("/root/ref/Point2RBox-v3/configs/_base_/datasets")
EXPECTED = {
    "dota.py", "dota_ms.py", "dota_qbox.py", "dota_coco.py",
    "dotav15.py", "dotav2.py", "dior.py", "hrsc.py", "hrsid.py",
    "fair.py", "rsar.py", "rsdd.py", "sardet100k.py", "sku110k.py",
    "srsdd.py", "ssdd.py", "star.py", "diatom.py", "ocdpcb.py",
}


def test_all_19_dataset_configs_present():
    actual = {path.name for path in MINE.glob("*.py")}
    assert actual == EXPECTED


def test_dataset_configs_match_upstream_bytes():
    if not REF.exists():
        return
    for name in EXPECTED:
        assert (MINE / name).read_bytes() == (REF / name).read_bytes(), name
