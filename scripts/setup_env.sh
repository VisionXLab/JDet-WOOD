#!/usr/bin/env bash
# Create the supported JDet-WOOD Jittor environment.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_NAME=${ENV_NAME:-jdet-wood}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
CC_PATH=${CC_PATH:-/usr/bin/g++-10}

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required; install Miniconda or Anaconda first" >&2
    exit 1
fi
if [[ ! -x "$CC_PATH" ]]; then
    echo "Jittor 1.3.8.5 requires a compatible compiler; expected $CC_PATH" >&2
    echo "Install g++-10 or rerun with CC_PATH=/path/to/g++" >&2
    exit 1
fi

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi
conda activate "$ENV_NAME"
python -m pip install -r "$REPO_ROOT/requirements.txt"
python -m pip install -e "$REPO_ROOT"

ENV_PREFIX=$(python -c 'import sys; print(sys.prefix)')
mkdir -p "$ENV_PREFIX/etc/conda/activate.d" "$ENV_PREFIX/etc/conda/deactivate.d"
printf 'export cc_path=%q\n' "$CC_PATH" > \
    "$ENV_PREFIX/etc/conda/activate.d/jdet_wood_cc.sh"
printf 'unset cc_path\n' > \
    "$ENV_PREFIX/etc/conda/deactivate.d/jdet_wood_cc.sh"
export cc_path="$CC_PATH"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

python -m jittor.test.test_example
python - <<'PY'
import jittor as jt

x = jt.float32([1, 2, 3])
assert (x + x).numpy().tolist() == [2.0, 4.0, 6.0]
print('JDet-WOOD Jittor environment is ready')
PY

echo "Activate later with: conda activate $ENV_NAME"
