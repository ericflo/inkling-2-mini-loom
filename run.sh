#!/usr/bin/env bash
set -euo pipefail

# Serve text and image chat with Inkling-2-Mini (NVFP4 + Q8_0 + DSpark) on AMD Loom.
#
# Usage:
#   ./run.sh                         # Start on 127.0.0.1:18422 (262k context, adaptive DSpark d1<->d3)
#   PORT=19000 ./run.sh              # Custom port
#   HOST=0.0.0.0 PORT=18422 ./run.sh # Bind all interfaces (LAN)
#   DRAFT_DEPTH=0 ./run.sh           # Disable speculative decoding (undrafted baseline)
#   CONTEXT_CAPACITY=131072 ./run.sh # Override KV pool size (e.g. for 64 GB machines)
#   POOL_BACKING=elastic ./run.sh    # Demand-committed memory instead of fixed eager pool
#
# Endpoints once ready:
#   OpenAI Chat API:  http://$HOST:$PORT/v1/chat/completions
#   Model status:     http://$HOST:$PORT/v1/status
#   Capabilities:     http://$HOST:$PORT/v1/models
#   Live dashboard:   http://$HOST:$PORT/dashboard/
#   Metrics:          http://$HOST:$PORT/metrics

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '3,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18422}"
DEVICE="${DEVICE:-amdgpu://?completion_spin_ns=4294967295}"
POOL_BACKING="${POOL_BACKING:-fixed}"
WAIT_MODE="${WAIT_MODE:-active}"
WORKERS="${WORKERS:-8}"
CONTEXT_CAPACITY="${CONTEXT_CAPACITY:-262144}"
DRAFT_DEPTH="${DRAFT_DEPTH:-3}"
ROWS="${ROWS:-4}"

if [[ ! -x ".venv/bin/loom-serve" ]]; then
  echo "ERROR: .venv/bin/loom-serve not found. Run ./setup.sh first." >&2
  exit 1
fi

for f in \
  "weights/Inkling-2-Mini-official-nvfp4-q8.gguf" \
  "weights/dspark-Inkling-2-Mini-official-q8.gguf" \
  "weights/tokenizer.json"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: Missing $f. Run ./setup.sh first." >&2
    exit 1
  fi
done

if [[ ! -f "weights/mmproj-Inkling-2-Mini-official-f16.gguf" ]]; then
  if [[ ! -f "models/Inkling-2-Mini/model-00001-of-00001.safetensors" ]]; then
    echo "ERROR: Vision weights are missing. Run ./setup.sh to build the image projector." >&2
    exit 1
  fi
  .venv/bin/python scripts/convert_official.py --only-missing
fi

mkdir -p "$SCRIPT_DIR/.loom_cache"
export XDG_CACHE_HOME="$SCRIPT_DIR/.loom_cache"

MODEL_ARG="$SCRIPT_DIR/model.json"
if [[ "$CONTEXT_CAPACITY" != "262144" || "$DRAFT_DEPTH" != "3" || "$ROWS" != "4" ]]; then
  MODEL_ARG="$SCRIPT_DIR/.loom_cache/deployment.json"
  cat > "$MODEL_ARG" <<EOF
{
  "name": "Inkling-2-Mini",
  "model": {
    "path": "$SCRIPT_DIR/model.json"
  },
  "definition": {
    "kind": "text",
    "rows": $ROWS,
    "context_capacity": $CONTEXT_CAPACITY,
    "draft_depth": $DRAFT_DEPTH
  }
}
EOF
fi

echo "==> Starting AMD Loom server for Inkling-2-Mini (text and image input, official NVFP4 + Q8_0)"
echo "    Listen:      http://$HOST:$PORT"
echo "    Dashboard:   http://$HOST:$PORT/dashboard/"
echo "    Device:      $DEVICE"
echo "    Backing:     $POOL_BACKING  |  Wait mode: $WAIT_MODE"
echo "    Context:     $CONTEXT_CAPACITY tokens  |  Rows: $ROWS  |  DSpark depth: $DRAFT_DEPTH"
echo ""

exec "$SCRIPT_DIR/.venv/bin/loom-serve" \
  --model="$MODEL_ARG" \
  --device="$DEVICE" \
  --pool_backing="$POOL_BACKING" \
  --wait_mode="$WAIT_MODE" \
  --workers="$WORKERS" \
  --listen="$HOST:$PORT" \
  "$@"
