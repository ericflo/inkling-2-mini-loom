# Inkling-2-Mini on AMD Loom (Strix Halo / `gfx1151`)

Fastest local inference setup for Thinking Machines Lab's **[`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** on **AMD Strix Halo (`gfx1151` / Radeon 8060S)** using **[AMD Loom (`hrx-loom`)](https://github.com/ROCm/hrx-system)** — converted **directly from the official Hugging Face safetensors weights** with **zero Q4 downgrade** and **DSpark speculative decoding** enabled.

## Why This Setup

- **100% official weight provenance:** Built directly from [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (`model-00001-of-00001.safetensors` + `mtp.safetensors` + `tokenizer.json`) via our auditable pure-NumPy converter ([`scripts/convert_official.py`](scripts/convert_official.py)). No third-party quantized weight uploads are downloaded or trusted.
- **No Q4 downgrade:** Official `Inkling-2-Mini` was trained and released natively in mixed precision (`NVFP4` on layers 0–19 routed experts, `MXFP8` on layers 20–23 routed experts, `BF16`/`F32` everywhere else). This conversion preserves layers 0–19 as **bit-exact native NVFP4** (`0.0` max reconstruction error, verified by built-in self-test) and keeps every other weight at **8-bit (`Q8_0`) or higher (`BF16`/`F32`)**.
- **3–6× faster than `llama.cpp` on Strix Halo:** Uses AMD Loom's handwritten `gfx11` RDNA3.5 AMDGPU assembly kernels for NVFP4 dequantization, INT8 `dot4` matvecs, FP16 WMMA prefill, and DSpark MTP speculative decoding.
- **Self-contained & git-clean:** All large files (`models/`, `weights/`, `.hf_cache/`, `.loom_cache/`, `.venv/`) stay inside this directory and are `.gitignored`.

---

## Requirements

- **Hardware:** AMD Strix Halo (`gfx1151` / Radeon 8060S) or supported RDNA3/RDNA4 Linux GPU with `/dev/kfd` and `/dev/dri` access
- **Memory:** ≥64 GB unified RAM/VRAM (~30.5 GiB fixed pool at 262k context; use `CONTEXT_CAPACITY=131072` or `POOL_BACKING=elastic` for tighter footprints)
- **Disk:** ~40 GB free (~19 GB official safetensors + ~17 GB converted Loom GGUFs; once converted, `models/` can optionally be deleted if you don't need to reconvert)
- **Software:** Linux, Python 3.12+ (`uv` optional, auto-detected by `setup.sh`)

---

## Quick Start

```bash
git clone https://github.com/ericflo/inkling-2-mini-loom.git
cd inkling-2-mini-loom
./setup.sh   # Creates .venv (hrx-loom 0.3.0), downloads official HF weights, converts to Loom GGUFs
./run.sh     # Starts the OpenAI-compatible server on http://127.0.0.1:18422
```

After the one-time `./setup.sh` step, just run `./run.sh` to start the server. It compiles/loads on `AMD Radeon 8060S (gfx1151)` in ~18 seconds and serves:

- **OpenAI Chat API:** `http://127.0.0.1:18422/v1/chat/completions`
- **Live Dashboard:** `http://127.0.0.1:18422/dashboard/`
- **Readiness / Status:** `http://127.0.0.1:18422/v1/status`
- **Prometheus Metrics:** `http://127.0.0.1:18422/metrics`

---

## Running with Custom Options (`run.sh`)

All options can be overridden via environment variables (run `./run.sh --help` anytime):

```bash
# Default: 127.0.0.1:18422, 262k context, 4 concurrent rows, adaptive DSpark (d1 <-> d3)
./run.sh

# Bind on LAN / custom port
HOST=0.0.0.0 PORT=19000 ./run.sh

# Disable speculative decoding (undrafted baseline)
DRAFT_DEPTH=0 ./run.sh

# Reduce KV pool size (e.g., 131k tokens on a 64 GB Strix Halo)
CONTEXT_CAPACITY=131072 ./run.sh

# Use demand-committed elastic memory instead of fixed eager allocation (~30.5 GB fixed)
POOL_BACKING=elastic ./run.sh
```

| Env Var | Default | Description |
|---|---|---|
| `HOST` | `127.0.0.1` | Bind host |
| `PORT` | `18422` | Bind port |
| `CONTEXT_CAPACITY` | `262144` | Shared KV token pool capacity (up to 1M native) |
| `ROWS` | `4` | Max concurrent request slots |
| `DRAFT_DEPTH` | `3` | DSpark speculative depth (`0` = off, `1`–`3` = adaptive `d1↔d3`) |
| `POOL_BACKING` | `fixed` | `fixed` (eager allocation, lowest latency) or `elastic` (demand-committed) |
| `WAIT_MODE` | `active` | `active` (low-latency spin wait) or `blocking` |
| `DEVICE` | `amdgpu://?completion_spin_ns=4294967295` | Loom HAL device URI |
| `WORKERS` | `8` | CPU workers for compilation & scheduling |

---

## API Examples

### Standard chat (recommended sampling: `temperature=1.0`, `top_p=1.0`)

```bash
curl http://127.0.0.1:18422/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Inkling-2-Mini",
    "messages": [{"role": "user", "content": "Explain how looped transformers work in 3 crisp bullet points."}],
    "temperature": 1.0,
    "top_p": 1.0
  }'
```

### Control thinking / reasoning effort

Inkling-2-Mini supports `reasoning_effort`: `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"` (default), `"xhigh"`, `"max"`, or `chat_template_kwargs: {"enable_thinking": false}`. Responses cleanly separate `reasoning_content` and `content`:

```bash
curl http://127.0.0.1:18422/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Inkling-2-Mini",
    "messages": [{"role": "user", "content": "What is 17 * 23 + 11?"}],
    "reasoning_effort": "low"
  }'
```

### OpenAI-style tool calling

```bash
curl http://127.0.0.1:18422/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Inkling-2-Mini",
    "messages": [{"role": "user", "content": "What is the weather in Seattle?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }]
  }'
```

### Per-request DSpark depth override

On a drafting-enabled server (`DRAFT_DEPTH=3`), you can pass `"draft_depth"` per request:
- omitted or `-1`: **adaptive** `d1 ↔ d3` (automatically disables drafting at `>= 2` concurrent streams)
- `1`: fixed depth 1
- `3`: fixed depth 3

> **Sampling tip (Thinking Machines documented note):** Keep the model's recommended `temperature: 1.0, top_p: 1.0` for general/long reasoning use. Pure greedy (`temperature: 0`) on long reasoning traces can trigger repetitive verification loops.

---

## Measured Benchmarks (Strix Halo `gfx1151`)

Measured on an `AMD Ryzen AI Max+ 395 w/ Radeon 8060S` (`gfx1151`, 128 GB unified LPDDR5X, Arch Linux, `hrx-loom 0.3.0`, FP16 KV cache, `T=1.0`, `top_p=1.0`) via [`scripts/verify_and_bench.py`](scripts/verify_and_bench.py):

| Workload (`T=1.0`) | Loom Undrafted (`DRAFT_DEPTH=0`) | Loom Adaptive DSpark (`DRAFT_DEPTH=3`) | DSpark Stats |
|---|---:|---:|---|
| Short prose decode (256 tok) | **44.2 tok/s** | **45.9 tok/s** | 36.5% of gen from drafts (`1.57 tok/epoch`) |
| Python code decode (LRU cache, 320 tok) | **43.5 tok/s** | **48.3 tok/s** | 44.5% of gen from drafts (`1.80 tok/epoch`) |
| Math / reasoning decode (prime proof, thinking on) | **43.0 tok/s** | **55.3 tok/s** | 55.6% of gen from drafts (`2.25 tok/epoch`, **+29%**) |
| ~2.9k context decode | **38.9 tok/s** | **44.5 tok/s** | 51.8% of gen from drafts (`2.07 tok/epoch`, **+14%**) |
| ~2.9k context prefill | **1,469.5 tok/s** | **1,332.1 tok/s** | — |
| ~11.5k context prefill | — | **1,390.7 tok/s** | — |
| 4 concurrent streams (aggregate decode) | — | **88.4 tok/s** | occupancy gate auto-disables drafting |
| Greedy (`T=0`) exactness (`DRAFT_DEPTH=3` vs `0`) | — | **PASS (3/3)** | 100% character-for-character identical |

Run the verification and benchmark suite yourself against a running server:

```bash
.venv/bin/python scripts/verify_and_bench.py --url http://127.0.0.1:18422
```

---

## Weight Precision Breakdown

`Inkling-2-Mini` is a 26.4B-parameter looped Mixture-of-Experts model (24 physical layers executed across 48 logical passes via 3× looping on core layers 6–17; 128 routed experts, top-8 active per pass).

| Component | Official Safetensors | Local Loom GGUF | GPU Runtime | Notes |
|---|---|---|---|---|
| Layers 0–19 MoE experts (`w13`, `w2`) | Native `NVFP4` (`E2M1` + `UE4M3` + `F32` outer scale) | **`NVFP4`** (`11.46 GiB`) | **`NVFP4`** | **Bit-exact repack** (`0.0` max abs error); `w13` rows de-interleaved `[gate; up]`, nibbles repacked to GGML split-half |
| Layers 20–23 MoE experts (`w13`, `w2`) | Native `MXFP8` (`E4M3` + `UE8M0`) | **`Q8_0`** (`4.03 GiB`) | **`Q8_0`** | 8-bit → 8-bit (`w13` rows de-interleaved `[gate; up]`) |
| Attention projections (`q, k, v, r, gate, o`) | `BF16` | **`Q8_0`** | **`Q8_0`** | All 24 physical layers |
| Embeddings (`token_embd`) & Target LM head (`output`) | `BF16` | **`Q8_0`** (`939 MiB`) | **`Q8_0`** | Verified target logits always use `Q8_0` |
| MoE routers (`ffn_gate_inp`) & loop controllers (`loop_*`) | `BF16` | `F32` | **`BF16`** | Losslessly staged `F32 → BF16` at load by `weights.loom` |
| Attention sinks (`attn_sink_proj`) & dynamic sconv projs | `BF16` | **`BF16`** | **`BF16`** | Exact raw `BF16` copy |
| Norms, biases, clips, rel bias, static sconv, res queries | `BF16` / `F32` | **`F32`** | **`F32`** | Exact `F32`; `rel_logits_proj` transposed, full-layer `static_w` flipped + transposed |
| DSpark MTP drafter (3 layers) | `BF16` (`mtp.safetensors`) | **`Q8_0`** | **`Q8_0`** | `dspark-Inkling-2-Mini-official-q8.gguf` (`0.36 GiB`) |
| DSpark Markov head (`markov_w1`, `joint_proj`, `hidden_out`) | `BF16` | **`BF16`** | **`BF16`** | Exact raw `BF16` copy; `fuse_weight`/`mask_embd`/norms in `F32` |
| DSpark *proposal-only* LM head (`%w_phead`) | Derived from `output.weight` | — | `Q4_0s` | Built at load by `head_weights.loom` **only** for draft proposals; never alters verified output |

Total converted weight footprint: **16.41 GiB** (main) + **0.36 GiB** (DSpark drafter) = **16.77 GiB**.
Total fixed GPU/unified memory pool at 262k context: **~30.5 GiB** (fits comfortably in 64 GB or 128 GB Strix Halo).

---

## Project Layout

```text
inkling-2-mini-loom/
├── README.md                 # This documentation
├── run.sh                    # Start the AMD Loom server on Strix Halo
├── setup.sh                  # Reproducible install + download + conversion
├── model.json                # AMD Loom model card pointing to local official GGUFs
├── llm_config.json           # Inkling-2-Mini model & chat configuration
├── weights.loom              # Load-time F32 -> BF16 weight policy (routers & loop controllers)
├── head_weights.loom         # Load-time Q8_0 -> Q4_0s proposal-only LM head policy
├── program/                  # AMD Loom source programs (compiled JIT for gfx1151)
│   ├── control.loom          # Inkling-2 chat template, thinking & tool-call parser, scheduler
│   ├── llm.loom              # gfx1151 NVFP4 / Q8_0 / WMMA / DSpark compute kernels
│   ├── prepare.loom          # Stage & execution shape planner
│   ├── service.loom          # OpenAI text service ABI
│   └── storage.loom          # Paged KV + SWA ring + short-conv state arena layout
├── scripts/
│   ├── convert_official.py   # Pure-NumPy official safetensors -> Loom GGUF & tokenizer converter
│   └── verify_and_bench.py   # Coherence, tool-calling, DSpark acceptance & throughput benchmarks
├── models/                   # (.gitignored) Official thinkingmachines/Inkling-2-Mini download (~19 GB)
├── weights/                  # (.gitignored) Converted NVFP4+Q8_0 GGUFs & Loom tokenizer (~17 GB)
├── .hf_cache/                # (.gitignored) Hugging Face download cache
├── .loom_cache/              # (.gitignored) Loom runtime/compiler cache
└── .venv/                    # (.gitignored) Local Python 3.12 venv with hrx-loom 0.3.0
```

---

## Provenance & Credits

- **Model weights & architecture:** Thinking Machines Lab — [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Apache-2.0)
- **Inference engine & `.loom` programs:** AMD — [`ROCm/hrx-system`](https://github.com/ROCm/hrx-system) (`hrx-loom 0.3.0`) and [`loom-fun/unsloth-Inkling-2-Mini-loom`](https://huggingface.co/loom-fun/unsloth-Inkling-2-Mini-loom) (Apache-2.0)
- The native image encoder, image component and mixed-input embedding code are adapted from the [Loom port at revision `53127fe`](https://huggingface.co/loom-fun/unsloth-Inkling-2-Mini-loom/tree/53127fe1faac17eba76e1de3d608d41fbbadde99/program) (Apache-2.0). Setup builds their weights from the official Thinking Machines checkpoint.
