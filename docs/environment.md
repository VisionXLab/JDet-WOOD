# Environment

## Validated Jittor setup

| Component | Version |
|---|---|
| Python | 3.10 |
| Jittor | 1.3.8.5 |
| NumPy | 1.26.4 |
| CUDA toolchain | Jittor CUDA 11.2 + cuDNN 8 |
| C++ compiler | g++-10 |

Install the Python dependencies and expose the repository package:

```bash
pip install -r requirements.txt
export cc_path=/usr/bin/g++-10
export PYTHONPATH="$PWD:$PWD/python"
```

Jittor compiles operators on first use, so the first model construction can
take several minutes.

## Compatibility notes

### NumPy

Keep NumPy at 1.26.4. Jittor 1.3.8.5 combined with NumPy 2.x can produce
incorrect values for operations consuming NumPy-backed arrays without raising
an exception. Verify the installation with:

```bash
python - <<'PY'
import jittor as jt
x = jt.float32([1, 2, 3])
assert (x + x).numpy().tolist() == [2.0, 4.0, 6.0]
print('Jittor array check passed')
PY
```

### Compiler

Jittor's CUDA 11.2 frontend is incompatible with newer system compiler headers
on some Linux distributions. The validated toolchain uses g++-10 selected by
the lowercase `cc_path` environment variable.

### Reference environment

Regenerating PyTorch goldens or using the reference DOTA merge metric requires
the original Point2RBox-v3 environment: PyTorch 2.2, torchvision 0.17,
mmengine 0.10.7, mmcv 2.2.0, mmdet 3.3.0 and the reference mmrotate package.
Regular Jittor training and inference do not require this second environment.
