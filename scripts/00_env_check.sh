#!/usr/bin/env bash
set -euo pipefail

mkdir -p runs/_env
OUT="runs/_env/env_$(date +%Y%m%d_%H%M%S).txt"

{
  echo "### DATE"
  date
  echo
  echo "### OS"
  uname -a
  echo
  echo "### GPU"
  nvidia-smi || true
  echo
  echo "### Python"
  python -V
  echo
  echo "### Pip freeze (top)"
  pip list | head -n 50 || true
  echo
  echo "### Torch"
  python - <<'PY'
import torch, sys
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("cudnn:", torch.backends.cudnn.version())
print("device count:", torch.cuda.device_count())
if torch.cuda.device_count():
    print("device 0:", torch.cuda.get_device_name(0))
PY
} | tee "$OUT"

echo "[OK] Saved env snapshot to: $OUT"
