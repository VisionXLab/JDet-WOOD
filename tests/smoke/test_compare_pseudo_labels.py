import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from tools.compare_pseudo_labels import compare, load_records  # noqa: E402


def _records(angle_delta=0):
    return [
        {"image_id": "P0", "bbox": [10, 20, 8, 4, 0.1 + angle_delta],
         "score": 1.0, "category_id": 0},
        {"image_id": "P1", "bbox": [30, 40, 6, 3, -0.2 + angle_delta],
         "score": 1.0, "category_id": 1},
    ]


def test_aligned_comparison_reports_box_delta():
    ref, cand, result = compare(_records(), _records(0.01))
    assert ref["records"] == cand["records"] == 2
    assert result["records_aligned"]
    assert result["bbox_abs_mean"][4] == pytest.approx(0.01)
    assert result["image_sets_equal"] and result["category_sets_equal"]


def test_schema_validation(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"image_id": "P0"}]))
    with pytest.raises(ValueError, match="missing"):
        load_records(path)


def test_cli_fails_on_missing_image(tmp_path):
    reference = tmp_path / "ref.json"
    candidate = tmp_path / "cand.json"
    reference.write_text(json.dumps(_records()))
    candidate.write_text(json.dumps(_records()[:1]))
    run = subprocess.run(
        [sys.executable, str(ROOT / "tools/compare_pseudo_labels.py"),
         str(reference), str(candidate)],
        text=True, capture_output=True)
    assert run.returncode == 1
    assert '"image_sets_equal": false' in run.stdout
