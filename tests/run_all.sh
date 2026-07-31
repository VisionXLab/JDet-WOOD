#!/usr/bin/env bash
# One-command parity panel. CPU by default; set TEST_GPU=1 for CUDA checks.
set -uo pipefail
cd "$(dirname "$0")/.."
CONDA_SH=${CONDA_SH:-/opt/miniconda3/etc/profile.d/conda.sh}
P2R_REF=${P2R_REF:-/root/ref/Point2RBox-v3}
TEST_CUDA_DEVICE=${TEST_CUDA_DEVICE:-0}
source "$CONDA_SH"

JT_TESTS=(test_linalg2x2_degenerate test_ted test_sam test_filter_masks
          test_v3_detector_funcs test_v3_detector_smoke test_v3_detector_stream
          test_v3_head_stage1 test_v3_head_loss test_pseudo_export
          test_datasets_import)
TORCH_TESTS=(test_merge_submission test_convert_test_results)

pass=0; fail=0; failed=()

conda activate p2r-jittor
for t in "${JT_TESTS[@]}"; do
  out=$(timeout 1800 python "tests/$t.py" 2>&1 | tail -1)
  if [[ "$out" == PASS* ]]; then pass=$((pass+1)); echo "[PASS] $t"
  else fail=$((fail+1)); failed+=("$t"); echo "[FAIL] $t: $out"; fi
done
# Anchor PYTHONPATH to this checkout rather than an editable JDet install.
out=$(PYTHONPATH=$PWD:$PWD/python timeout 600 python -m pytest \
      tests/parity tests/test_v3_norm_eval.py -q 2>&1 | tail -1)
if [[ "$out" == *passed* && "$out" != *failed* ]]; then
  pass=$((pass+1)); echo "[PASS] pytest parity+norm_eval ($out)"
else fail=$((fail+1)); failed+=("parity"); echo "[FAIL] parity: $out"; fi

conda activate p2r-torch
for t in "${TORCH_TESTS[@]}"; do
  out=$(PYTHONPATH="$P2R_REF" timeout 900 python "tests/$t.py" 2>&1 | tail -1)
  if [[ "$out" == PASS* ]]; then pass=$((pass+1)); echo "[PASS] $t (torch)"
  else fail=$((fail+1)); failed+=("$t"); echo "[FAIL] $t: $out"; fi
done

if [[ "${TEST_GPU:-0}" == "1" ]]; then
  conda activate p2r-jittor
  for t in test_sam test_v3_head_loss; do
    out=$(CUDA_VISIBLE_DEVICES="$TEST_CUDA_DEVICE" TEST_SAM_CUDA=1 TEST_CUDA=1 timeout 1800 \
          python "tests/$t.py" 2>&1 | tail -1)
    if [[ "$out" == PASS* ]]; then pass=$((pass+1)); echo "[PASS] $t (GPU)"
    else fail=$((fail+1)); failed+=("$t-gpu"); echo "[FAIL] $t GPU: $out"; fi
  done
fi

echo "=============================="
echo "PASS $pass / FAIL $fail ${failed[*]:-}"
[[ $fail -eq 0 ]]
