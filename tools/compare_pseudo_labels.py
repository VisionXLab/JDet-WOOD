"""Validate and compare mmrotate/Jittor pseudo-label bbox JSON files.

The comparison is intentionally both structural and distributional.  It catches
schema/category/image coverage regressions even when floating-point predictions
cannot be matched element-by-element, and reports direct box deltas when record
identity/order is the same.
"""

import argparse
import json
import math
from collections import Counter

import numpy as np


REQUIRED = ("image_id", "bbox", "score", "category_id")


def load_records(path):
    with open(path) as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"{path}: top-level value must be a list")
    for i, record in enumerate(records):
        missing = [key for key in REQUIRED if key not in record]
        if missing:
            raise ValueError(f"{path}[{i}]: missing {missing}")
        box = np.asarray(record["bbox"], dtype=np.float64)
        if box.shape != (5,) or not np.isfinite(box).all():
            raise ValueError(f"{path}[{i}]: bbox must be five finite values")
        # 官方 v2 产物中存在一个 w=h=0 的退化框；保持并显式统计，
        # 只把不可能的负尺寸视为 schema 错误。
        if box[2] < 0 or box[3] < 0:
            raise ValueError(f"{path}[{i}]: bbox width/height must be nonnegative")
        score = float(record["score"])
        if not math.isfinite(score):
            raise ValueError(f"{path}[{i}]: score must be finite")
        int(record["category_id"])
    return records


def summarize(records):
    boxes = np.asarray([record["bbox"] for record in records],
                       dtype=np.float64).reshape(-1, 5)
    images = Counter(str(record["image_id"]) for record in records)
    classes = Counter(int(record["category_id"]) for record in records)
    scores = np.asarray([record["score"] for record in records],
                        dtype=np.float64)
    if len(boxes):
        aspect = np.maximum(boxes[:, 2], boxes[:, 3]) / np.maximum(
            np.minimum(boxes[:, 2], boxes[:, 3]), 1e-12)
        angle_hist = np.histogram(
            boxes[:, 4], bins=18, range=(-np.pi / 2, np.pi / 2))[0]
    else:
        aspect = np.empty(0)
        angle_hist = np.zeros(18, dtype=np.int64)
    return {
        "records": len(records),
        "images": len(images),
        "classes": dict(sorted(classes.items())),
        "degenerate_boxes": int(
            np.sum((boxes[:, 2] == 0) | (boxes[:, 3] == 0))),
        "per_image_min": min(images.values(), default=0),
        "per_image_max": max(images.values(), default=0),
        "score_min": float(scores.min()) if len(scores) else None,
        "score_max": float(scores.max()) if len(scores) else None,
        "aspect_mean": float(aspect.mean()) if len(aspect) else None,
        "aspect_median": float(np.median(aspect)) if len(aspect) else None,
        "angle_hist": angle_hist.tolist(),
    }


def _js_divergence(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / max(a.sum(), 1)
    b = b / max(b.sum(), 1)
    m = (a + b) / 2

    def kl(x):
        keep = x > 0
        return np.sum(x[keep] * np.log(x[keep] / m[keep]))

    return float((kl(a) + kl(b)) / 2)


def compare(reference, candidate):
    ref_summary = summarize(reference)
    cand_summary = summarize(candidate)
    result = {
        "record_count_delta": len(candidate) - len(reference),
        "record_count_rel": abs(len(candidate) - len(reference))
                            / max(len(reference), 1),
        "image_sets_equal": {
            str(x["image_id"]) for x in reference
        } == {
            str(x["image_id"]) for x in candidate
        },
        "category_sets_equal": {
            int(x["category_id"]) for x in reference
        } == {
            int(x["category_id"]) for x in candidate
        },
        "angle_hist_js": _js_divergence(
            ref_summary["angle_hist"], cand_summary["angle_hist"]),
    }
    ref_ids = [(str(x["image_id"]), int(x["category_id"]))
               for x in reference]
    cand_ids = [(str(x["image_id"]), int(x["category_id"]))
                for x in candidate]
    result["records_aligned"] = ref_ids == cand_ids
    if result["records_aligned"] and reference:
        ref_boxes = np.asarray([x["bbox"] for x in reference], np.float64)
        cand_boxes = np.asarray([x["bbox"] for x in candidate], np.float64)
        delta = np.abs(cand_boxes - ref_boxes)
        result["bbox_abs_mean"] = delta.mean(axis=0).tolist()
        result["bbox_abs_p95"] = np.quantile(delta, .95, axis=0).tolist()
    return ref_summary, cand_summary, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", help="official mmrotate .bbox.json")
    parser.add_argument("candidate", help="Jittor .bbox.json")
    parser.add_argument("--max-count-rel", type=float, default=0.01)
    parser.add_argument("--max-angle-js", type=float, default=0.02)
    args = parser.parse_args()

    reference = load_records(args.reference)
    candidate = load_records(args.candidate)
    ref_summary, cand_summary, result = compare(reference, candidate)
    print(json.dumps({
        "reference": ref_summary,
        "candidate": cand_summary,
        "comparison": result,
    }, indent=2, sort_keys=True))
    passed = (
        result["record_count_rel"] <= args.max_count_rel
        and result["image_sets_equal"]
        and result["category_sets_equal"]
        and result["angle_hist_js"] <= args.max_angle_js
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
