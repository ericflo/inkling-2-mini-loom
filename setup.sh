#!/usr/bin/env bash
set -euo pipefail

# Reproducible setup script for Inkling-2-Mini (official weights) on AMD Loom (gfx1151).
# Keeps all large files (HF cache, official safetensors, converted GGUFs, venv)
# inside this directory (all .gitignored).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> [1/3] Setting up Python virtualenv & dependencies in .venv ..."
if [[ ! -x ".venv/bin/python" ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python \
      "hrx-loom==0.3.0" "huggingface-hub" "hf-xet" "numpy" "requests"
  else
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install \
      "hrx-loom==0.3.0" "huggingface-hub" "hf-xet" "numpy" "requests"
  fi
fi

echo "==> [2/3] Downloading official thinkingmachines/Inkling-2-Mini weights ..."
mkdir -p models/Inkling-2-Mini .hf_cache weights
if [[ ! -f "models/Inkling-2-Mini/model-00001-of-00001.safetensors" || ! -f "models/Inkling-2-Mini/mtp.safetensors" ]]; then
  HF_HOME="$SCRIPT_DIR/.hf_cache" \
  HF_XET_HIGH_PERFORMANCE=1 \
  .venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='thinkingmachines/Inkling-2-Mini',
    local_dir='$SCRIPT_DIR/models/Inkling-2-Mini',
)
"
else
  echo "    Official safetensors already present in models/Inkling-2-Mini/ (skipping download)."
fi

echo "==> [3/3] Converting official safetensors -> Loom GGUFs (NVFP4 + Q8_0 + DSpark) & tokenizer ..."
if [[ ! -f "weights/Inkling-2-Mini-official-nvfp4-q8.gguf" || ! -f "weights/dspark-Inkling-2-Mini-official-q8.gguf" ]]; then
  .venv/bin/python scripts/convert_official.py \
    --model-dir models/Inkling-2-Mini \
    --out-dir weights
elif [[ ! -f "weights/tokenizer.json" ]]; then
  .venv/bin/python scripts/convert_official.py \
    --model-dir models/Inkling-2-Mini \
    --out-dir weights \
    --tokenizer-only
else
  echo "    Converted GGUFs and tokenizer already present in weights/ (delete them to reconvert)."
fi

echo ""
echo "Setup complete! Start the server with:"
echo "  ./run.sh"
