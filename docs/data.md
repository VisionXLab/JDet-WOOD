# DOTA-v1.0 data preparation

Download DOTA-v1.0 from the official project or an equivalent complete mirror.
The original dataset contains 1,411 training images, 458 validation images and
937 unlabeled test images with 15 classes.

## Patch generation

Use the mmrotate DOTA splitter with the single-scale settings used by
Point2RBox-v3:

| Option | Value |
|---|---|
| patch size | 1024 |
| gap | 200 |
| scale rate | 1.0 |
| image-rate threshold | 0.6 |
| instance IoF threshold | 0.7 |
| padding value | `[104, 116, 124]` |

The JSON splitter configurations are provided in `tools/dota_split/`. A correct
single-scale split contains 21,046 trainval patches and 10,833 test patches.

Expected layout:

```text
split_ss_dota/
├── trainval/
│   ├── images/
│   └── annfiles/
└── test/
    └── images/
```

The pseudo-label exporter writes
`point2rbox_v3_pseudo_labels.bbox.json` under the configured data root. The
second-stage dataset reads this JSON together with `trainval/images`.

Update `images_dir`, `annotations_file` and pseudo-label paths in
`configs/point2rbox_v3/` if your data root differs from the provided config.

## Input validation

Before training, verify that no Git LFS pointer files remain, every image can be
decoded, trainval image and annotation stems match, and each annotation line has
eight polygon coordinates followed by class name and difficulty flag. Parsers
accept both official headers and headerless files, as well as LF or CRLF line
endings.
