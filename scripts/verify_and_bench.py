#!/usr/bin/env python3
"""
Correctness, DSpark acceptance, and throughput verification suite for
Inkling-2-Mini on AMD Loom (Strix Halo gfx1151).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import time
from typing import Any

import requests


def wait_ready(base_url: str, timeout_s: int = 300) -> dict[str, Any]:
    t0 = time.monotonic()
    last_state = None
    while time.monotonic() - t0 < timeout_s:
        try:
            r = requests.get(f"{base_url}/v1/status", timeout=5)
            if r.status_code == 200:
                st = r.json()
                models = st.get("data") or st.get("models") or []
                if models:
                    m = models[0]
                    loom = m.get("loom") or {}
                    state = loom.get("registration") or m.get("state") or m.get("status")
                    if state != last_state:
                        print(f"[{time.monotonic() - t0:5.1f}s] model={m.get('id')} registration={state}", flush=True)
                        last_state = state
                    if state in ("ready", "READY"):
                        return st
                    if state in ("failed", "FAILED", "error", "ERROR"):
                        raise RuntimeError(f"Model failed to load: {json.dumps(m, indent=2)}")
        except requests.RequestException:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"Server at {base_url} not ready after {timeout_s}s")


def read_metrics(base_url: str) -> dict[str, float]:
    r = requests.get(f"{base_url}/metrics", timeout=5)
    out: dict[str, float] = {}
    if r.status_code != 200:
        return out
    for line in r.text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+([-+0-9.eE]+)$", line.strip())
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def stream_chat(
    base_url: str,
    messages: list[dict[str, Any]],
    *,
    model: str = "Inkling-2-Mini",
    max_tokens: int = 256,
    temperature: float | None = 1.0,
    top_p: float | None = 1.0,
    enable_thinking: bool = False,
    draft_depth: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if draft_depth is not None:
        payload["draft_depth"] = draft_depth
    if tools is not None:
        payload["tools"] = tools
    if seed is not None:
        payload["seed"] = seed

    t0 = time.monotonic()
    t_first: float | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}
    chunks = 0

    with requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=300,
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data_str = raw[5:].strip()
            if data_str == "[DONE]":
                break
            obj = json.loads(data_str)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            txt = delta.get("content") or ""
            rsn = delta.get("reasoning_content") or delta.get("reasoning") or ""
            tc = delta.get("tool_calls")
            if (txt or rsn or tc) and t_first is None:
                t_first = time.monotonic()
            if txt:
                content_parts.append(txt)
                chunks += 1
            if rsn:
                reasoning_parts.append(rsn)
                chunks += 1
            if tc:
                tool_calls.extend(tc)
                chunks += 1

    t_end = time.monotonic()
    total_s = t_end - t0
    ttft_s = (t_first - t0) if t_first is not None else total_s
    gen_s = (t_end - t_first) if t_first is not None else 0.0
    comp_toks = usage.get("completion_tokens") or chunks
    prompt_toks = usage.get("prompt_tokens") or 0
    decode_tps = (comp_toks - 1) / gen_s if (gen_s > 0 and comp_toks > 1) else (comp_toks / total_s if total_s > 0 else 0.0)
    prefill_tps = prompt_toks / ttft_s if (ttft_s > 0 and prompt_toks > 0) else 0.0

    return {
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "tool_calls": tool_calls,
        "usage": usage,
        "prompt_tokens": prompt_toks,
        "completion_tokens": comp_toks,
        "ttft_s": ttft_s,
        "gen_s": gen_s,
        "total_s": total_s,
        "decode_tps": decode_tps,
        "prefill_tps": prefill_tps,
    }


def make_filler_doc(approx_tokens: int) -> str:
    para = (
        "The Strix Halo unified memory subsystem combines 256-bit LPDDR5X channels with "
        "an integrated RDNA 3.5 graphics architecture (gfx1151), allowing large mixture-of-experts "
        "language models to stream active parameters without PCIe transfers. In a looped transformer "
        "design, physical layers in the middle cycle execute multiple passes with cross-pass residual "
        "controllers, increasing effective reasoning depth while preserving a compact static footprint. "
    )
    # ~72 tokens per paragraph
    n = max(1, approx_tokens // 70)
    return "\n\n".join(f"[Section {i + 1}] {para}" for i in range(n))


def extract_draft_stats(m_before: dict[str, float], m_after: dict[str, float]) -> str:
    accepted = 0.0
    decoded = 0.0
    epochs = 0.0
    for k, v in m_after.items():
        dv = v - m_before.get(k, 0.0)
        if 'field="draft_accepted_inputs"' in k:
            accepted += dv
        elif 'field="decode_tokens"' in k:
            decoded += dv
        elif 'field="completed_epochs"' in k:
            epochs += dv
    parts = []
    if decoded > 0:
        parts.append(f"draft_acc={int(accepted)}/{int(decoded)} ({accepted / decoded:.1%} of gen)")
    if epochs > 1:
        parts.append(f"tok/epoch={decoded / max(epochs - 1, 1):.2f}")
    return ", ".join(parts) if parts else "draft=n/a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18422")
    ap.add_argument("--quick", action="store_true", help="Skip 8k prefill and 4-stream tests")
    args = ap.parse_args()

    print(f"Waiting for server at {args.url} ...")
    wait_ready(args.url)
    print("Server READY!\n")

    # 1. Sanity / Coherence checks
    print("=== 1. COHERENCE & FEATURE CHECKS ===")
    r1 = stream_chat(
        args.url,
        [{"role": "user", "content": "Explain in 2 crisp sentences why unified memory helps local MoE LLM inference."}],
        max_tokens=256,
        temperature=0.0,
        enable_thinking=False,
    )
    print(f"[Greedy, effort=none] ({r1['completion_tokens']} tok, {r1['decode_tps']:.1f} tok/s):")
    if r1["reasoning"]:
        print("  reasoning:", repr(r1["reasoning"][:120]))
    print("  content:  ", r1["content"].strip())

    r2 = stream_chat(
        args.url,
        [{"role": "user", "content": "What is 17 * 23 + 11? Give the final number clearly."}],
        max_tokens=384,
        temperature=1.0,
        enable_thinking=True,
    )
    print(f"\n[T=1.0, think=on] ({r2['completion_tokens']} tok, {r2['decode_tps']:.1f} tok/s):")
    print("  reasoning:", repr(r2["reasoning"][:160]))
    print("  content:  ", r2["content"].strip()[:200])

    # Tool-calling check
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    r_tool = stream_chat(
        args.url,
        [{"role": "user", "content": "What's the weather in Seattle right now?"}],
        max_tokens=160,
        temperature=0.0,
        enable_thinking=False,
        tools=tools,
    )
    print(f"\n[Tool call] tool_calls={r_tool['tool_calls']} content={r_tool['content'][:120]!r}")

    # 2. Throughput & DSpark Acceptance Benchmarks (T=1.0 recommended sampling)
    print("\n=== 2. BENCHMARKS (T=1.0, adaptive DSpark) ===")

    # Warmup done above; now measure clean benches
    benches = [
        (
            "decode_prose",
            [{"role": "user", "content": "Write a clear, detailed explanation of how B-trees work in databases, covering node splits, balancing, and disk I/O."}],
            256,
            False,
        ),
        (
            "decode_code",
            [{"role": "user", "content": "Implement a clean, thread-safe LRU cache class in Python using a doubly linked list and dictionary, with full type hints and brief docstrings."}],
            320,
            False,
        ),
        (
            "decode_math_think",
            [{"role": "user", "content": "Prove that there are infinitely many prime numbers of the form 4k + 3."}],
            384,
            True,
        ),
    ]
    for name, msgs, mtok, think in benches:
        mb = read_metrics(args.url)
        res = stream_chat(args.url, msgs, max_tokens=mtok, temperature=1.0, enable_thinking=think)
        ma = read_metrics(args.url)
        dstat = extract_draft_stats(mb, ma)
        print(
            f"  {name:18s} | prompt={res['prompt_tokens']:5d} | gen={res['completion_tokens']:4d} tok "
            f"| ttft={res['ttft_s']*1000:6.1f} ms | decode={res['decode_tps']:5.1f} tok/s | {dstat}"
        )

    # 2k context prefill + decode
    doc2k = make_filler_doc(2048)
    mb = read_metrics(args.url)
    r2k = stream_chat(
        args.url,
        [{"role": "user", "content": f"Read the following technical notes carefully and then summarize the key points in 150 words:\n\n{doc2k}"}],
        max_tokens=200,
        temperature=1.0,
        enable_thinking=False,
    )
    ma = read_metrics(args.url)
    print(
        f"  {'ctx_2k':18s} | prompt={r2k['prompt_tokens']:5d} | gen={r2k['completion_tokens']:4d} tok "
        f"| prefill={r2k['prefill_tps']:6.1f} tok/s | decode={r2k['decode_tps']:5.1f} tok/s | {extract_draft_stats(mb, ma)}"
    )

    if not args.quick:
        doc8k = make_filler_doc(8192)
        r8k = stream_chat(
            args.url,
            [{"role": "user", "content": f"Summarize the main theme of this document in one sentence:\n\n{doc8k}"}],
            max_tokens=64,
            temperature=1.0,
            enable_thinking=False,
        )
        print(
            f"  {'prefill_8k':18s} | prompt={r8k['prompt_tokens']:5d} | gen={r8k['completion_tokens']:4d} tok "
            f"| prefill={r8k['prefill_tps']:6.1f} tok/s | decode={r8k['decode_tps']:5.1f} tok/s"
        )

        # 4 concurrent streams
        prompts = [
            f"Write a clean technical guide #{i+1} explaining how Linux virtual memory paging and TLBs work."
            for i in range(4)
        ]
        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futs = [
                ex.submit(
                    stream_chat,
                    args.url,
                    [{"role": "user", "content": p}],
                    max_tokens=200,
                    temperature=1.0,
                    enable_thinking=False,
                )
                for p in prompts
            ]
            results = [f.result() for f in futs]
        wall = time.monotonic() - t0
        tot_toks = sum(r["completion_tokens"] for r in results)
        print(
            f"  {'streams_4':18s} | total_gen={tot_toks:4d} tok | wall={wall:5.2f}s | aggregate={tot_toks / wall:5.1f} tok/s"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
