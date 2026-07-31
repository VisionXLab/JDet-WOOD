"""Tier-2 heterogeneous dataset registration and parsing smoke tests."""

import ast
import json
from pathlib import Path

import numpy as np
from PIL import Image

from jdet.data.coco_rbox import (
    SARDet100kDataset, SAR_Det_Finegrained_Dataset, SRSDDDataset)
from jdet.data.diatom import DIATOMDataset
from jdet.data.dior import DIORDataset
from jdet.data.hrsc import HRSCDataset
from jdet.data.sku110k import SKU110KDataset
from jdet.utils.registry import DATASETS


REF = Path("/root/ref/Point2RBox-v3")


def _ref_classes(filename, class_name):
    tree = ast.parse((REF / "mmrotate/datasets" / filename).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (isinstance(item, ast.Assign)
                        and getattr(item.targets[0], "id", "") == "METAINFO"):
                    return ast.literal_eval(item.value)["classes"]
    raise AssertionError(f"METAINFO not found: {filename}:{class_name}")


def _image(path):
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(path)


def _dirs(tmp_path):
    images = tmp_path / "images"
    anns = tmp_path / "anns"
    images.mkdir()
    anns.mkdir()
    return images, anns


def test_registry_and_ref_class_tables():
    checks = (
        (DIORDataset, "dior.py", "DIORDataset"),
        (DIATOMDataset, "diatom.py", "DIATOMDataset"),
        (SARDet100kDataset, "sardet100k.py",
         "SAR_Det_Finegrained_Dataset"),
    )
    for cls, filename, ref_name in checks:
        assert tuple(cls.CLASSES) == tuple(_ref_classes(filename, ref_name))
        assert DATASETS.get(cls.__name__) is cls
    assert HRSCDataset.CLASSES == ("ship",)
    assert SKU110KDataset.CLASSES == ("object",)
    assert DATASETS.get("SAR_Det_Finegrained_Dataset") is \
        SAR_Det_Finegrained_Dataset


def test_dior_hrsc_and_diatom_xml(tmp_path):
    images, anns = _dirs(tmp_path)
    (anns / "D0.xml").write_text(
        "<annotation><object><name>ship</name><robndbox>"
        "<x_left_top>10</x_left_top><y_left_top>10</y_left_top>"
        "<x_right_top>30</x_right_top><y_right_top>10</y_right_top>"
        "<x_right_bottom>30</x_right_bottom><y_right_bottom>20</y_right_bottom>"
        "<x_left_bottom>10</x_left_bottom><y_left_bottom>20</y_left_bottom>"
        "</robndbox></object></annotation>")
    (tmp_path / "ids.txt").write_text("D0\n")
    _image(images / "D0.jpg")
    ds = DIORDataset(images_dir=str(images), annfiles_dir=str(anns),
                     imgset_file=str(tmp_path / "ids.txt"),
                     filter_empty_gt=True, batch_size=1)
    assert ds.img_infos[0]["ann"]["bboxes"].shape == (1, 5)
    assert ds.img_infos[0]["ann"]["labels"].tolist() == [13]

    (anns / "H0.xml").write_text(
        "<HRSC_Image><HRSC_Objects><HRSC_Object>"
        "<mbox_cx>20</mbox_cx><mbox_cy>15</mbox_cy>"
        "<mbox_w>20</mbox_w><mbox_h>10</mbox_h><mbox_ang>0.3</mbox_ang>"
        "</HRSC_Object></HRSC_Objects></HRSC_Image>")
    _image(images / "H0.bmp")
    ds = HRSCDataset(images_dir=str(images), annfiles_dir=str(anns),
                     filter_empty_gt=True, batch_size=1)
    info = next(x for x in ds.img_infos if x["filename"] == "H0.bmp")
    assert np.isclose(info["ann"]["bboxes"][0, 4], 0.3)

    (anns / "T0.xml").write_text(
        "<annotation><objects><object><bbox><xmin>10</xmin><ymin>10</ymin>"
        "<xmax>30</xmax><ymax>20</ymax></bbox></object></objects></annotation>")
    _image(images / "T0.jpg")
    ds = DIATOMDataset(images_dir=str(images), annfiles_dir=str(anns),
                       filter_empty_gt=True, batch_size=1)
    info = next(x for x in ds.img_infos if x["filename"] == "T0.jpg")
    np.testing.assert_allclose(info["ann"]["bboxes"][0],
                               [20, 15, 20, 10, 0])


def test_sku_and_coco_json(tmp_path):
    images, _ = _dirs(tmp_path)
    _image(images / "S0.jpg")
    sku_path = tmp_path / "sku.json"
    sku_path.write_text(json.dumps([
        {"image_id": "S0", "rbbox": [20, 15, 20, 10, 0.5]}]))
    ds = SKU110KDataset(images_dir=str(images), ann_json_file=str(sku_path),
                        filter_empty_gt=True, batch_size=1)
    np.testing.assert_allclose(ds.img_infos[0]["ann"]["bboxes"][0],
                               [20, 15, 20, 10, 0.5])

    _image(images / "C0.jpg")
    coco_path = tmp_path / "coco.json"
    coco_path.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "C0.jpg"}],
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": 7,
            "bbox": [10, 10, 20, 10],
            "segmentation": [[10, 10, 30, 10, 30, 20, 10, 20]],
        }],
        "categories": [{"id": 7, "name": "Dredger"}],
    }))
    ds = SRSDDDataset(images_dir=str(images), ann_json_file=str(coco_path),
                      filter_empty_gt=True, batch_size=1)
    assert ds.img_infos[0]["ann"]["bboxes"].shape == (1, 5)
    assert ds.img_infos[0]["ann"]["labels"].tolist() == [1]
