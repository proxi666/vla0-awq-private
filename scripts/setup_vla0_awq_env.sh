#!/usr/bin/env bash
set -euo pipefail

SOURCE_ENV="${SOURCE_ENV:-qwen}"
TARGET_ENV="${TARGET_ENV:-vla0-awq}"

if conda env list | awk '{print $1}' | grep -qx "${TARGET_ENV}"; then
  echo "[awq-env] Conda env ${TARGET_ENV} already exists."
else
  echo "[awq-env] Cloning ${SOURCE_ENV} -> ${TARGET_ENV}"
  conda create -y -n "${TARGET_ENV}" --clone "${SOURCE_ENV}"
fi

echo "[awq-env] Installing AWQ tooling into ${TARGET_ENV}"
conda run -n "${TARGET_ENV}" python -m pip install -r "$(dirname "$0")/../requirements-awq.txt"

echo "[awq-env] Verifying imports"
conda run -n "${TARGET_ENV}" python - <<'PY'
modules = ["torch", "transformers", "qwen_vl_utils", "llmcompressor"]
for module in modules:
    imported = __import__(module)
    version = getattr(imported, "__version__", "unknown")
    print(f"{module}: ok ({version})")
PY

echo "[awq-env] Ready. Use: conda run -n ${TARGET_ENV} python scripts/quantize_vla0_awq.py --dry-run-calibration"
