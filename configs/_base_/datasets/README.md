# Dataset config baselines

These 19 files are byte-for-byte snapshots of
`Point2RBox-v3/configs/_base_/datasets/`. They are retained to satisfy the
tier-2 port scope and to keep every upstream dataset pipeline/config available
for parity inspection.

JDet does not implement MMEngine `_base_` inheritance. Runnable Point2RBox-v2
configs are therefore flattened under `configs/point2rbox_v2/`; the loaders
referenced here are registered under `python/jdet/data/`.
