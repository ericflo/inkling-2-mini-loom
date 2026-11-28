#!/usr/bin/env python3
"""
Convert official thinkingmachines/Inkling-2-Mini safetensors directly into
AMD Loom-compatible GGUF files (main model, DSpark MTP drafter and vision encoder) with ZERO Q4
downgrade:

  - Layers 0-19 routed MoE experts:  bit-exact native NVFP4 (repacked from
    official safetensors E2M1 nibbles + UE4M3 scales + F32 outer scales,
    zero dequant/requant loss)
  - Layers 20-23 routed MoE experts: MXFP8 -> Q8_0 (8-bit lossless-grade)
  - Attention, embeddings, LM head:  BF16  -> Q8_0 (8-bit)
  - Routers & loop controllers:      BF16  -> F32 (converted to exact BF16 at
    load time by weights.loom)
  - Sinks & dynamic sconv projs:     BF16  -> BF16 (exact raw copy)
  - Norms, biases, clips, sconv,
    attention-residual queries:      F32   (exact)
  - DSpark MTP drafter:              BF16  -> Q8_0 (3 layers) + BF16 (Markov
    head) + F32 (norms/biases/fuse/mask)
  - Vision encoder and projector:    BF16  -> F16 matrices + F32 positions,
    normalization, layer scales and biases

Requires only Python stdlib + numpy. No PyTorch, no third-party weight files.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

# GGML quantization type IDs (GGUF v3)
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_BF16 = 30
GGML_TYPE_NVFP4 = 40

# GGUF metadata value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8

GGUF_ALIGNMENT = 32

FULL_LAYERS = {5, 11, 17, 23}
LOOP_LAYERS = set(range(6, 18))
NVFP4_LAYERS = set(range(0, 20))
MXFP8_LAYERS = set(range(20, 24))


def _make_e4m3fn_table() -> np.ndarray:
    """Exact lookup table for IEEE/OCP float8_e4m3fn (bias=7, 0x7F/0xFF=NaN)."""
    codes = np.arange(256, dtype=np.uint8)
    exp = ((codes & 0x7F) >> 3).astype(np.int32)
    man = (codes & 0x07).astype(np.float32)
    vals = np.where(
        exp == 0,
        np.ldexp(man, -9),
        np.ldexp(1.0 + man / 8.0, exp - 7),
    )
    vals = np.where((codes & 0x80) != 0, -vals, vals)
    vals[(codes & 0x7F) == 0x7F] = np.nan
    return vals.astype(np.float32)


E4M3FN_LUT = _make_e4m3fn_table()
E2M1_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


class SafetensorsReader:
    """Minimal zero-dependency mmap/seek safetensors reader supporting BF16/F32/U8/F8_E4M3."""

    def __init__(self, path: Path):
        self.path = path
        self.fp = open(path, "rb")
        hlen = struct.unpack("<Q", self.fp.read(8))[0]
        self.header = json.loads(self.fp.read(hlen))
        self.data_base = 8 + hlen

    def close(self) -> None:
        self.fp.close()

    def __enter__(self) -> "SafetensorsReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def shape(self, key: str) -> tuple[int, ...]:
        return tuple(self.header[key]["shape"])

    def dtype(self, key: str) -> str:
        return self.header[key]["dtype"]

    def read_raw(self, key: str) -> bytes:
        s, e = self.header[key]["data_offsets"]
        self.fp.seek(self.data_base + s)
        buf = self.fp.read(e - s)
        if len(buf) != e - s:
            raise EOFError(f"Short read for {key}: got {len(buf)}, expected {e - s}")
        return buf

    def read_raw_slice(self, key: str, byte_offset: int, byte_len: int) -> bytes:
        s, e = self.header[key]["data_offsets"]
        if byte_offset < 0 or byte_offset + byte_len > e - s:
            raise ValueError(f"Slice out of range for {key}")
        self.fp.seek(self.data_base + s + byte_offset)
        buf = self.fp.read(byte_len)
        if len(buf) != byte_len:
            raise EOFError(f"Short slice read for {key}")
        return buf

    def read_u8(self, key: str) -> np.ndarray:
        return np.frombuffer(self.read_raw(key), dtype=np.uint8).reshape(self.shape(key))

    def read_u16_bf16(self, key: str) -> np.ndarray:
        if self.dtype(key) != "BF16":
            raise TypeError(f"Expected BF16 for {key}, got {self.dtype(key)}")
        return np.frombuffer(self.read_raw(key), dtype=np.uint16).reshape(self.shape(key))

    def read_f32(self, key: str) -> np.ndarray:
        dt = self.dtype(key)
        raw = self.read_raw(key)
        sh = self.shape(key)
        if dt == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(sh).copy()
        if dt == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16).reshape(sh)
            return (u16.astype(np.uint32) << 16).view(np.float32)
        raise TypeError(f"Unsupported dtype {dt} for read_f32({key})")


def quantize_q8_0(arr_f32: np.ndarray) -> bytes:
    """
    Quantize a contiguous float32 array (last dim multiple of 32) into GGML Q8_0
    blocks: 34 bytes per 32 values = [f16 scale (2B), int8[32] (32B)].
    """
    if arr_f32.shape[-1] % 32 != 0:
        raise ValueError(f"Q8_0 requires last dim % 32 == 0, got {arr_f32.shape}")
    blocks = np.ascontiguousarray(arr_f32, dtype=np.float32).reshape(-1, 32)
    n_blocks = blocks.shape[0]
    amax = np.max(np.abs(blocks), axis=1)
    d_f32 = amax / np.float32(127.0)
    d_f16 = d_f32.astype(np.float16)
    # Use the actual stored f16 scale when rounding quants to minimize error
    d_stored = d_f16.astype(np.float32)
    inv_d = np.where(d_stored != 0.0, np.float32(1.0) / d_stored, np.float32(0.0))
    scaled = blocks * inv_d[:, None]
    # Round half away from zero to match C roundf()
    q = np.copysign(np.floor(np.abs(scaled) + np.float32(0.5)), scaled)
    q_i8 = np.clip(q, -127.0, 127.0).astype(np.int8)

    out = np.empty((n_blocks, 34), dtype=np.uint8)
    out[:, :2] = d_f16.view(np.uint8).reshape(n_blocks, 2)
    out[:, 2:] = q_i8.view(np.uint8)
    return out.tobytes()


def quantize_bf16_tensor_to_q8_0_chunks(
    reader: SafetensorsReader, key: str, chunk_rows: int = 8192
) -> Iterable[bytes]:
    """Stream-quantize a 2D BF16 safetensors tensor [rows, cols] into Q8_0 bytes."""
    shape = reader.shape(key)
    if reader.dtype(key) != "BF16":
        raise TypeError(f"Expected BF16 for {key}, got {reader.dtype(key)}")
    rows, cols = shape
    row_bytes = cols * 2
    for r0 in range(0, rows, chunk_rows):
        r1 = min(r0 + chunk_rows, rows)
        raw = reader.read_raw_slice(key, r0 * row_bytes, (r1 - r0) * row_bytes)
        u16 = np.frombuffer(raw, dtype=np.uint16).reshape(r1 - r0, cols)
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        yield quantize_q8_0(f32)


def repack_nvfp4_experts_chunks(
    reader: SafetensorsReader,
    w_key: str,
    s_key: str,
    deinterleave_w13: bool,
    chunk_experts: int = 8,
) -> Iterable[bytes]:
    """
    Bit-exact repack of official Inkling-2-Mini NVFP4 experts into GGML_TYPE_NVFP4
    (36 bytes per 64 values: 4 UE4M3 scale bytes + 32 split-half nibble bytes).

    Safetensors layout:
      - w: [E, R, K/2] uint8, interleaved nibbles: lo = elem 2j, hi = elem 2j+1
      - s: [E, R, K/16] float8_e4m3fn (positive UE4M3 in [0..0x7E])
      - For w13_weight, rows are interleaved [gate_0, up_0, gate_1, up_1, ...];
        Loom / GGML expects [gate_0..R/2-1, up_0..R/2-1].

    GGML_TYPE_NVFP4 layout (block_nvfp4, QK=64, 4 sub-blocks of 16):
      - d[4]:  4 UE4M3 scale bytes verbatim from safetensors
      - qs[32]: 4 sub-blocks of 8 bytes; within each 8-byte sub-block,
        lo nibbles of bytes 0..7 = elems 0..7, hi nibbles = elems 8..15.
    """
    e_cnt, rows, half_k = reader.shape(w_key)
    se_cnt, s_rows, k16 = reader.shape(s_key)
    k = half_k * 2
    assert (e_cnt, rows, k // 16) == (se_cnt, s_rows, k16)
    assert k % 64 == 0

    w_exp_bytes = rows * half_k
    s_exp_bytes = rows * k16
    n_super = k // 64

    for e0 in range(0, e_cnt, chunk_experts):
        e1 = min(e0 + chunk_experts, e_cnt)
        ce = e1 - e0
        w = np.frombuffer(
            reader.read_raw_slice(w_key, e0 * w_exp_bytes, ce * w_exp_bytes),
            dtype=np.uint8,
        ).reshape(ce, rows, half_k)
        s = np.frombuffer(
            reader.read_raw_slice(s_key, e0 * s_exp_bytes, ce * s_exp_bytes),
            dtype=np.uint8,
        ).reshape(ce, rows, k16)

        if np.any((s >= 0x80) | (s == 0x7F)):
            raise ValueError(f"Invalid UE4M3 scale byte in {s_key} experts {e0}..{e1}")

        if deinterleave_w13:
            # [ce, 2*inter, ...] -> even rows (gate) followed by odd rows (up)
            w = np.concatenate([w[:, 0::2, :], w[:, 1::2, :]], axis=1)
            s = np.concatenate([s[:, 0::2, :], s[:, 1::2, :]], axis=1)

        # Reshape into super-blocks of 64 values (4 sub-blocks of 16 values = 8 bytes each)
        # w_sub: (ce, rows, n_super, 4, 8)
        w_sub = w.reshape(ce, rows, n_super, 4, 8)
        # Unpack interleaved nibbles (lo=even, hi=odd) -> elems 0..15:
        # elems 0..7 come from bytes 0..3 (lo=0,2,4,6; hi=1,3,5,7)
        # elems 8..15 come from bytes 4..7 (lo=8,10,12,14; hi=9,11,13,15)
        lo_in = w_sub & np.uint8(0x0F)
        hi_in = w_sub >> np.uint8(4)
        elems = np.empty((ce, rows, n_super, 4, 16), dtype=np.uint8)
        elems[..., 0::2] = lo_in
        elems[..., 1::2] = hi_in
        # Repack in GGML split-half order: lo=elems 0..7, hi=elems 8..15
        qs = (elems[..., :8] | (elems[..., 8:] << np.uint8(4))).reshape(ce, rows, n_super, 32)
        ds = s.reshape(ce, rows, n_super, 4)

        out = np.empty((ce, rows, n_super, 36), dtype=np.uint8)
        out[..., :4] = ds
        out[..., 4:] = qs
        yield out.tobytes()


def convert_mxfp8_experts_to_q8_0_chunks(
    reader: SafetensorsReader,
    w_key: str,
    s_key: str,
    deinterleave_w13: bool,
    chunk_experts: int = 4,
) -> Iterable[bytes]:
    """
    Dequantize official MXFP8 experts (layers 20-23: F8_E4M3 weights + UE8M0
    per-32 scales) and quantize to Q8_0.
    """
    e_cnt, rows, k = reader.shape(w_key)
    se_cnt, s_rows, k32 = reader.shape(s_key)
    assert (e_cnt, rows, k // 32) == (se_cnt, s_rows, k32)
    assert k % 32 == 0

    w_exp_bytes = rows * k
    s_exp_bytes = rows * k32

    for e0 in range(0, e_cnt, chunk_experts):
        e1 = min(e0 + chunk_experts, e_cnt)
        ce = e1 - e0
        w_u8 = np.frombuffer(
            reader.read_raw_slice(w_key, e0 * w_exp_bytes, ce * w_exp_bytes),
            dtype=np.uint8,
        ).reshape(ce, rows, k)
        s_u8 = np.frombuffer(
            reader.read_raw_slice(s_key, e0 * s_exp_bytes, ce * s_exp_bytes),
            dtype=np.uint8,
        ).reshape(ce, rows, k32)

        if np.any(s_u8 == 255):
            raise ValueError(f"MXFP8 UE8M0 NaN scale in {s_key} experts {e0}..{e1}")

        if deinterleave_w13:
            w_u8 = np.concatenate([w_u8[:, 0::2, :], w_u8[:, 1::2, :]], axis=1)
            s_u8 = np.concatenate([s_u8[:, 0::2, :], s_u8[:, 1::2, :]], axis=1)

        vals = E4M3FN_LUT[w_u8]
        if np.isnan(vals).any():
            raise ValueError(f"MXFP8 E4M3FN NaN weight in {w_key} experts {e0}..{e1}")
        scales = np.ldexp(np.float32(1.0), s_u8.astype(np.int32) - 127).astype(np.float32)
        vals = vals.reshape(ce, rows, k32, 32) * scales[..., None]
        yield quantize_q8_0(vals.reshape(ce, rows, k))


def _gguf_byte_size(shape_rowmajor: tuple[int, ...], ggml_type: int) -> int:
    ne = 1
    for d in shape_rowmajor:
        ne *= d
    if ggml_type == GGML_TYPE_F32:
        return ne * 4
    if ggml_type in (GGML_TYPE_F16, GGML_TYPE_BF16):
        return ne * 2
    if ggml_type == GGML_TYPE_Q8_0:
        assert shape_rowmajor[-1] % 32 == 0
        return (ne // 32) * 34
    if ggml_type == GGML_TYPE_NVFP4:
        assert shape_rowmajor[-1] % 64 == 0
        return (ne // 64) * 36
    raise ValueError(f"Unknown ggml_type {ggml_type}")


@dataclass
class TensorSpec:
    name: str
    shape: tuple[int, ...]  # row-major logical shape (outermost .. innermost)
    ggml_type: int
    writer_fn: Callable[[], Iterable[bytes]]

    @property
    def nbytes(self) -> int:
        return _gguf_byte_size(self.shape, self.ggml_type)


def _pack_gguf_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def write_gguf(
    out_path: Path,
    metadata: list[tuple[str, int, object]],
    tensors: list[TensorSpec],
) -> None:
    """Stream-write a valid GGUF v3 file with 32-byte aligned tensor payloads."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute relative data offsets (32-byte aligned)
    offsets: list[int] = []
    cur = 0
    for t in tensors:
        rem = cur % GGUF_ALIGNMENT
        if rem:
            cur += GGUF_ALIGNMENT - rem
        offsets.append(cur)
        cur += t.nbytes
    total_data_bytes = cur

    # Serialize header + KV metadata + tensor info table
    hdr = bytearray()
    hdr += b"GGUF"
    hdr += struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, vtype, val in metadata:
        hdr += _pack_gguf_str(key)
        hdr += struct.pack("<I", vtype)
        if vtype == GGUF_TYPE_STRING:
            hdr += _pack_gguf_str(str(val))
        elif vtype == GGUF_TYPE_UINT32:
            hdr += struct.pack("<I", int(val))
        else:
            raise ValueError(f"Unsupported metadata type {vtype}")

    for t, off in zip(tensors, offsets):
        hdr += _pack_gguf_str(t.name)
        # GGML stores dimensions innermost-first (reversed relative to NumPy/PyTorch)
        ggml_dims = tuple(reversed(t.shape))
        hdr += struct.pack("<I", len(ggml_dims))
        for d in ggml_dims:
            hdr += struct.pack("<Q", d)
        hdr += struct.pack("<IQ", t.ggml_type, off)

    pad_hdr = (GGUF_ALIGNMENT - (len(hdr) % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT
    hdr += b"\x00" * pad_hdr
    data_start = len(hdr)

    t0 = time.monotonic()
    written_data = 0
    # Simultaneous setup/first launches must never share a partial output.
    with (
        tempfile.TemporaryDirectory(dir=out_path.parent) as directory,
        (Path(directory) / out_path.name).open("wb") as fp,
    ):
        fp.write(hdr)
        for idx, (t, off) in enumerate(zip(tensors, offsets)):
            cur_pos = fp.tell() - data_start
            if cur_pos < off:
                fp.write(b"\x00" * (off - cur_pos))
            elif cur_pos != off:
                raise RuntimeError(f"Offset mismatch before {t.name}: {cur_pos} != {off}")
            t_written = 0
            for chunk in t.writer_fn():
                fp.write(chunk)
                t_written += len(chunk)
            if t_written != t.nbytes:
                raise RuntimeError(
                    f"Size mismatch for {t.name}: wrote {t_written}, expected {t.nbytes}"
                )
            written_data += t_written
            if (idx + 1) % 50 == 0 or idx + 1 == len(tensors):
                elapsed = max(time.monotonic() - t0, 1e-6)
                gb = written_data / (1024**3)
                tot_gb = total_data_bytes / (1024**3)
                print(
                    f"  [{idx + 1:3d}/{len(tensors)}] {gb:5.2f} / {tot_gb:5.2f} GiB "
                    f"({gb / elapsed:4.2f} GiB/s)  last={t.name}",
                    flush=True,
                )

        fp.flush()
        os.replace(fp.name, out_path)
    final_sz = out_path.stat().st_size
    print(
        f"Wrote {out_path} ({final_sz / (1024**3):.2f} GiB, {len(tensors)} tensors) "
        f"in {time.monotonic() - t0:.1f}s"
    )


def build_main_specs(reader: SafetensorsReader) -> list[TensorSpec]:
    specs: list[TensorSpec] = []

    def add_f32(gguf_name: str, st_key: str, transform: Callable[[np.ndarray], np.ndarray] | None = None):
        sh = reader.shape(st_key)
        if transform is not None:
            # Probe transformed shape without reading huge data (only used on small tensors)
            arr = transform(reader.read_f32(st_key))
            sh = arr.shape
            specs.append(
                TensorSpec(
                    gguf_name,
                    sh,
                    GGML_TYPE_F32,
                    lambda k=st_key, tf=transform: [np.ascontiguousarray(tf(reader.read_f32(k)), dtype=np.float32).tobytes()],
                )
            )
        else:
            specs.append(
                TensorSpec(
                    gguf_name,
                    sh,
                    GGML_TYPE_F32,
                    lambda k=st_key: [np.ascontiguousarray(reader.read_f32(k), dtype=np.float32).tobytes()],
                )
            )

    def add_bf16_raw(gguf_name: str, st_key: str):
        assert reader.dtype(st_key) == "BF16", f"Expected BF16 for {st_key}"
        sh = reader.shape(st_key)
        specs.append(
            TensorSpec(
                gguf_name,
                sh,
                GGML_TYPE_BF16,
                lambda k=st_key: [reader.read_raw(k)],
            )
        )

    def add_q8_from_bf16(gguf_name: str, st_key: str):
        sh = reader.shape(st_key)
        specs.append(
            TensorSpec(
                gguf_name,
                sh,
                GGML_TYPE_Q8_0,
                lambda k=st_key: quantize_bf16_tensor_to_q8_0_chunks(reader, k),
            )
        )

    # Global tensors
    add_q8_from_bf16("token_embd.weight", "model.llm.embed.weight")
    add_f32("token_embd_norm.weight", "model.llm.embed_norm.weight")
    add_f32("output_norm.weight", "model.llm.norm.weight")
    add_f32("output_res_query.weight", "model.llm.attn_res.output_query")
    add_q8_from_bf16("output.weight", "model.llm.unembed.weight")
    add_f32("loop_res_query.weight", "model.llm.middle_cycle_cross_pass_queries")
    add_f32("loop_kv_res_query.weight", "model.llm.middle_cycle_cross_pass_kv_queries")

    # Attention-residual queries: attn_queries.{i} (i=0..22) -> blk.{i+1}.attn_res_query.weight
    for i in range(23):
        add_f32(f"blk.{i + 1}.attn_res_query.weight", f"model.llm.attn_res.attn_queries.{i}")
    for i in range(24):
        add_f32(f"blk.{i}.ffn_res_query.weight", f"model.llm.attn_res.mlp_queries.{i}")

    # Physical layers 0..23
    for layer in range(24):
        lp = f"model.llm.layers.{layer}"
        gp = f"blk.{layer}"

        # Norms
        if layer in FULL_LAYERS:
            add_f32(f"{gp}.attn_q_inp_norm.weight", f"{lp}.q_input_norm.weight")
            add_f32(f"{gp}.attn_k_inp_norm.weight", f"{lp}.k_input_norm.weight")
            add_f32(f"{gp}.attn_v_inp_norm.weight", f"{lp}.v_input_norm.weight")
        else:
            add_f32(f"{gp}.attn_norm.weight", f"{lp}.attn_norm.weight")
        add_f32(f"{gp}.ffn_norm.weight", f"{lp}.mlp_norm.weight")

        # Attention projections (BF16 -> Q8_0)
        add_q8_from_bf16(f"{gp}.attn_q.weight", f"{lp}.attn.wq_du.weight")
        add_q8_from_bf16(f"{gp}.attn_k.weight", f"{lp}.attn.wk_dv.weight")
        add_q8_from_bf16(f"{gp}.attn_v.weight", f"{lp}.attn.wv_dv.weight")
        add_q8_from_bf16(f"{gp}.attn_rel.weight", f"{lp}.attn.wr_du.weight")
        add_q8_from_bf16(f"{gp}.attn_gate.weight", f"{lp}.attn.gate_attn.weight")
        add_q8_from_bf16(f"{gp}.attn_output.weight", f"{lp}.attn.wo_ud.weight")

        # Attention sink projection (BF16 raw) & clips/rel bias (F32)
        add_bf16_raw(f"{gp}.attn_sink_proj.weight", f"{lp}.attn.attn_sink_proj.weight")
        add_f32(f"{gp}.attn_q_clip.weight", f"{lp}.attn.q_l2_clip.scale")
        add_f32(f"{gp}.attn_rel_clip.weight", f"{lp}.attn.rel_l2_clip.scale")
        # Transpose rel_logits_proj.proj (16, 1024) -> (1024, 16) F32
        add_f32(
            f"{gp}.attn_rel_bias.weight",
            f"{lp}.attn.rel_logits_proj.proj",
            transform=lambda a: np.ascontiguousarray(a.T),
        )

        # Short convolutions
        if layer in FULL_LAYERS:
            # static_w is (4, 1024) with newest tap at index 0 -> flip axis 0 & transpose to (1024, 4)
            add_f32(
                f"{gp}.attn_k_conv.weight",
                f"{lp}.attn.k_sconv.static_w",
                transform=lambda a: np.ascontiguousarray(a[::-1, :].T),
            )
            add_f32(
                f"{gp}.attn_v_conv.weight",
                f"{lp}.attn.v_sconv.static_w",
                transform=lambda a: np.ascontiguousarray(a[::-1, :].T),
            )
            add_bf16_raw(f"{gp}.attn_k_conv_proj.weight", f"{lp}.attn.k_sconv.weight_proj")
            add_bf16_raw(f"{gp}.attn_v_conv_proj.weight", f"{lp}.attn.v_sconv.weight_proj")
        else:
            add_f32(
                f"{gp}.attn_k_conv.weight",
                f"{lp}.attn.k_sconv.weight",
                transform=lambda a: np.ascontiguousarray(a.squeeze(1)),
            )
            add_f32(
                f"{gp}.attn_v_conv.weight",
                f"{lp}.attn.v_sconv.weight",
                transform=lambda a: np.ascontiguousarray(a.squeeze(1)),
            )
        add_f32(
            f"{gp}.attn_post_conv.weight",
            f"{lp}.attn_sconv.weight",
            transform=lambda a: np.ascontiguousarray(a.squeeze(1)),
        )
        add_f32(
            f"{gp}.ffn_post_conv.weight",
            f"{lp}.mlp_sconv.weight",
            transform=lambda a: np.ascontiguousarray(a.squeeze(1)),
        )

        # Loop controllers (layers 6..17): stored as F32 in GGUF so weights.loom
        # converts them cleanly to exact BF16 at load time without editing weights.loom
        if layer in LOOP_LAYERS:
            for ctrl in ("attn", "k", "v", "mlp"):
                add_f32(f"{gp}.loop_{ctrl}_down.weight", f"{lp}.controllers.{ctrl}.down_weight")
                add_f32(f"{gp}.loop_{ctrl}_gain.weight", f"{lp}.controllers.{ctrl}.gain_weight")
                add_f32(f"{gp}.loop_{ctrl}_forget.weight", f"{lp}.controllers.{ctrl}.forget_weight")

        # MoE router (stored as F32 in GGUF for weights.loom F32->BF16 stage) & bias (F32)
        add_f32(f"{gp}.ffn_gate_inp.weight", f"{lp}.mlp.gate.weight")
        add_f32(f"{gp}.exp_probs_b.bias", f"{lp}.mlp.gate.bias")

        # MoE experts
        w13_key = f"{lp}.mlp.experts.w13_weight"
        w2_key = f"{lp}.mlp.experts.w2_weight"
        if layer in NVFP4_LAYERS:
            # Bit-exact NVFP4 repacking + per-expert F32 outer scale = amax / denom
            e_cnt, rows13, half_k13 = reader.shape(w13_key)
            e_cnt2, rows2, half_k2 = reader.shape(w2_key)
            specs.append(
                TensorSpec(
                    f"{gp}.ffn_gate_up_exps.weight",
                    (e_cnt, rows13, half_k13 * 2),
                    GGML_TYPE_NVFP4,
                    lambda wk=w13_key, sk=w13_key + "_scale": repack_nvfp4_experts_chunks(
                        reader, wk, sk, deinterleave_w13=True
                    ),
                )
            )
            specs.append(
                TensorSpec(
                    f"{gp}.ffn_gate_up_exps.scale",
                    (e_cnt,),
                    GGML_TYPE_F32,
                    lambda ak=w13_key + "_amax", dk=w13_key + "_outer_scale_denominator": [
                        (reader.read_f32(ak) / reader.read_f32(dk)[0]).astype(np.float32).tobytes()
                    ],
                )
            )
            specs.append(
                TensorSpec(
                    f"{gp}.ffn_down_exps.weight",
                    (e_cnt2, rows2, half_k2 * 2),
                    GGML_TYPE_NVFP4,
                    lambda wk=w2_key, sk=w2_key + "_scale": repack_nvfp4_experts_chunks(
                        reader, wk, sk, deinterleave_w13=False
                    ),
                )
            )
            specs.append(
                TensorSpec(
                    f"{gp}.ffn_down_exps.scale",
                    (e_cnt2,),
                    GGML_TYPE_F32,
                    lambda ak=w2_key + "_amax", dk=w2_key + "_outer_scale_denominator": [
                        (reader.read_f32(ak) / reader.read_f32(dk)[0]).astype(np.float32).tobytes()
                    ],
                )
            )
        else:
            # Layers 20..23: MXFP8 -> Q8_0
            sh13 = reader.shape(w13_key)
            sh2 = reader.shape(w2_key)
            specs.append(
                TensorSpec(
                    f"{gp}.ffn_gate_up_exps.weight",
                    sh13,
                    GGML_TYPE_Q8_0,
                    lambda wk=w13_key, sk=w13_key + "_scale": convert_mxfp8_experts_to_q8_0_chunks(
                        reader, wk, sk, deinterleave_w13=True
                    ),
                )
            )
            specs.append(
                TensorSpec(
                    f"{gp}.ffn_down_exps.weight",
                    sh2,
                    GGML_TYPE_Q8_0,
                    lambda wk=w2_key, sk=w2_key + "_scale": convert_mxfp8_experts_to_q8_0_chunks(
                        reader, wk, sk, deinterleave_w13=False
                    ),
                )
            )

    assert len(specs) == 734, f"Expected 734 main tensors, got {len(specs)}"
    return specs


def build_vision_specs(reader: SafetensorsReader) -> list[TensorSpec]:
    """Map the official visual tower to the tensors consumed by vision_encode."""
    specs: list[TensorSpec] = []

    def add(source: str, name: str, ggml_type: int) -> None:
        key = "model.visual." + source
        shape = reader.shape(key)
        is_patch = name == "v.patch_embd.weight"
        if is_patch:
            # The checkpoint flattens each 20x20 RGB patch in HWC order.
            shape = (shape[0], 3, 20, 20)

        def write() -> Iterable[bytes]:
            values = reader.read_f32(key)
            if is_patch:
                values = values.reshape(-1, 20, 20, 3).transpose(0, 3, 1, 2)
            dtype = "<f2" if ggml_type == GGML_TYPE_F16 else "<f4"
            yield np.ascontiguousarray(values, dtype=dtype).tobytes()

        specs.append(TensorSpec(name, shape, ggml_type, write))

    for source, name in (
        ("class_embedding", "class_embd"),
        ("cls_position_embedding", "class_position_embd.weight"),
        ("position_embedding", "position_embd.weight"),
    ):
        add("backbone.embeddings." + source, "v." + name, GGML_TYPE_F32)
    add("backbone.embeddings.patch_embedding.weight", "v.patch_embd.weight", GGML_TYPE_F16)
    for suffix in ("weight", "bias"):
        add("backbone.ln_pre." + suffix, "v.pre_ln." + suffix, GGML_TYPE_F32)

    for i in range(23):
        source = f"backbone.layers.{i}."
        dest = f"v.blk.{i}."
        for n in (1, 2):
            add(source + f"ls_{n}.gamma", dest + f"ls{n}.weight", GGML_TYPE_F32)
            for suffix in ("weight", "bias"):
                add(source + f"ln_{n}.{suffix}", dest + f"ln{n}.{suffix}", GGML_TYPE_F32)
        for layer, name in (
            ("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"),
            ("self_attn.v_proj", "attn_v"), ("self_attn.out_proj", "attn_out"),
            ("mlp.c_fc", "ffn_up"), ("mlp.c_proj", "ffn_down"),
        ):
            add(source + layer + ".weight", dest + name + ".weight", GGML_TYPE_F16)
            add(source + layer + ".bias", dest + name + ".bias", GGML_TYPE_F32)
    add("proj_in.weight", "mm.1.weight", GGML_TYPE_F16)
    add("proj_out.weight", "mm.2.weight", GGML_TYPE_F16)
    return specs


def build_dspark_specs(reader: SafetensorsReader) -> list[TensorSpec]:
    specs: list[TensorSpec] = []

    def add_f32(gguf_name: str, st_key: str):
        sh = reader.shape(st_key)
        specs.append(
            TensorSpec(
                gguf_name,
                sh,
                GGML_TYPE_F32,
                lambda k=st_key: [np.ascontiguousarray(reader.read_f32(k), dtype=np.float32).tobytes()],
            )
        )

    def add_bf16_raw(gguf_name: str, st_key: str):
        assert reader.dtype(st_key) == "BF16", f"Expected BF16 for {st_key}"
        sh = reader.shape(st_key)
        specs.append(
            TensorSpec(
                gguf_name,
                sh,
                GGML_TYPE_BF16,
                lambda k=st_key: [reader.read_raw(k)],
            )
        )

    def add_q8_from_bf16(gguf_name: str, st_key: str):
        sh = reader.shape(st_key)
        specs.append(
            TensorSpec(
                gguf_name,
                sh,
                GGML_TYPE_Q8_0,
                lambda k=st_key: quantize_bf16_tensor_to_q8_0_chunks(reader, k),
            )
        )

    # Top-level + Markov head
    add_f32("dspark_fuse.weight", "fuse_weight")
    add_f32("mask_embd.weight", "mask_embed")
    add_f32("output_norm.weight", "norm.weight")
    add_f32("dspark_norm.weight", "markov_head.dflare_norm.weight")
    add_f32("dspark_hidden_out.bias", "markov_head.hidden_out.bias")
    add_bf16_raw("dspark_hidden_out.weight", "markov_head.hidden_out.weight")
    add_f32("joint_proj.bias", "markov_head.rnn_proj.bias")
    add_bf16_raw("joint_proj.weight", "markov_head.rnn_proj.weight")
    add_bf16_raw("markov_w1.weight", "markov_head.markov_w1.weight")

    # 3 DSpark transformer layers
    for i in range(3):
        lp = f"layers.{i}"
        gp = f"blk.{i}"
        add_f32(f"{gp}.dflare_norm.weight", f"{lp}.dflare_norm.weight")
        add_f32(f"{gp}.attn_norm.weight", f"{lp}.input_layernorm.weight")
        add_f32(f"{gp}.ffn_norm.weight", f"{lp}.post_attention_layernorm.weight")
        add_f32(f"{gp}.attn_q_norm.weight", f"{lp}.self_attn.q_norm.weight")
        add_f32(f"{gp}.attn_k_norm.weight", f"{lp}.self_attn.k_norm.weight")
        add_f32(f"{gp}.attn_k_ctx_norm.weight", f"{lp}.self_attn.k_norm_ctx.weight")
        add_q8_from_bf16(f"{gp}.attn_q.weight", f"{lp}.self_attn.q_proj.weight")
        add_q8_from_bf16(f"{gp}.attn_k.weight", f"{lp}.self_attn.k_proj.weight")
        add_q8_from_bf16(f"{gp}.attn_v.weight", f"{lp}.self_attn.v_proj.weight")
        add_q8_from_bf16(f"{gp}.attn_k_ctx.weight", f"{lp}.self_attn.k_proj_ctx.weight")
        add_q8_from_bf16(f"{gp}.attn_v_ctx.weight", f"{lp}.self_attn.v_proj_ctx.weight")
        add_q8_from_bf16(f"{gp}.attn_output.weight", f"{lp}.self_attn.o_proj.weight")
        add_q8_from_bf16(f"{gp}.ffn_gate.weight", f"{lp}.mlp.gate_proj.weight")
        add_q8_from_bf16(f"{gp}.ffn_up.weight", f"{lp}.mlp.up_proj.weight")
        add_q8_from_bf16(f"{gp}.ffn_down.weight", f"{lp}.mlp.down_proj.weight")

    assert len(specs) == 54, f"Expected 54 DSpark tensors, got {len(specs)}"
    return specs


def verify_nvfp4_roundtrip(model_st: Path) -> None:
    """Self-test: prove our NVFP4 repacking matches reference dequant to 0.00000000 max abs error."""
    with SafetensorsReader(model_st) as r:
        w_key = "model.llm.layers.0.mlp.experts.w13_weight"
        s_key = w_key + "_scale"
        a_key = w_key + "_amax"
        d_key = w_key + "_outer_scale_denominator"

        # Reference dequant of expert 0 (matches llama.cpp conversion/inkling2.py)
        e_cnt, rows, half_k = r.shape(w_key)
        k = half_k * 2
        w0 = np.frombuffer(r.read_raw_slice(w_key, 0, rows * half_k), dtype=np.uint8).reshape(rows, half_k)
        s0 = np.frombuffer(r.read_raw_slice(s_key, 0, rows * (k // 16)), dtype=np.uint8).reshape(rows, k // 16)
        outer = r.read_f32(a_key)[0] / r.read_f32(d_key)[0]

        ref = np.empty((rows, k), dtype=np.float32)
        ref[:, 0::2] = E2M1_LUT[w0 & 0xF]
        ref[:, 1::2] = E2M1_LUT[w0 >> 4]
        ref.reshape(rows, k // 16, 16)[:] *= E4M3FN_LUT[s0][:, :, None]
        ref *= outer
        ref_deint = np.concatenate([ref[0::2], ref[1::2]], axis=0)

        # Repack expert 0 via our converter and dequant via GGML/Loom block_nvfp4 rules
        first_chunk = next(iter(repack_nvfp4_experts_chunks(r, w_key, s_key, deinterleave_w13=True, chunk_experts=1)))
        blocks = np.frombuffer(first_chunk, dtype=np.uint8).reshape(rows, k // 64, 36)
        d_bytes = blocks[:, :, :4]  # UE4M3
        qs = blocks[:, :, 4:].reshape(rows, k // 64, 4, 8)
        lo = qs & 0xF
        hi = qs >> 4
        nib = np.concatenate([lo, hi], axis=-1)  # (rows, k//64, 4, 16)
        kvals = (E2M1_LUT * 2.0).astype(np.float32)  # doubled codebook (0,1,2,3,4,6,8,12,...)
        # Loom @g_ue4m3_to_f32 decodes UE4M3 with 0.5 factor to cancel 2x codebook
        d_f32 = E4M3FN_LUT[d_bytes] * np.float32(0.5)
        got = (kvals[nib] * d_f32[:, :, :, None]).reshape(rows, k) * outer

        max_err = float(np.max(np.abs(ref_deint - got)))
        if max_err != 0.0:
            raise AssertionError(f"NVFP4 round-trip error {max_err} != 0.0!")
        print(f"Self-test PASSED: NVFP4 expert 0 round-trip max_abs_err = {max_err:.1e} (bit-exact!)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert official Inkling-2-Mini safetensors to AMD Loom GGUFs")
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/Inkling-2-Mini"),
        help="Directory containing official model-00001-of-00001.safetensors and mtp.safetensors",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("weights"),
        help="Output directory for generated GGUF files",
    )
    ap.add_argument("--skip-main", action="store_true", help="Skip main model conversion")
    ap.add_argument("--skip-draft", action="store_true", help="Skip DSpark drafter conversion")
    ap.add_argument("--skip-vision", action="store_true", help="Skip vision encoder and projector conversion")
    ap.add_argument("--only-missing", action="store_true", help="Reuse existing GGUFs and tokenizer")
    ap.add_argument("--tokenizer-only", action="store_true", help="Only generate Loom-patched tokenizer.json")
    ap.add_argument("--llm-config", type=Path, default=Path("llm_config.json"), help="Path to llm_config.json")
    ap.add_argument("--verify-only", action="store_true", help="Run NVFP4 bit-exact round-trip self-test only")
    args = ap.parse_args()

    src_tok = args.model_dir / "tokenizer.json"
    dst_tok = args.out_dir / "tokenizer.json"
    if args.tokenizer_only:
        patch_tokenizer_for_loom(src_tok, dst_tok, args.llm_config)
        return 0

    model_st = args.model_dir / "model-00001-of-00001.safetensors"
    mtp_st = args.model_dir / "mtp.safetensors"
    out_main = args.out_dir / "Inkling-2-Mini-official-nvfp4-q8.gguf"
    out_draft = args.out_dir / "dspark-Inkling-2-Mini-official-q8.gguf"
    out_vision = args.out_dir / "mmproj-Inkling-2-Mini-official-f16.gguf"

    def needed(path: Path) -> bool:
        return not (args.only_missing and path.is_file())

    if args.verify_only:
        verify_nvfp4_roundtrip(model_st)
        return 0

    if needed(dst_tok) and src_tok.is_file() and args.llm_config.is_file():
        patch_tokenizer_for_loom(src_tok, dst_tok, args.llm_config)

    if not args.skip_main and needed(out_main):
        verify_nvfp4_roundtrip(model_st)
        print(f"\nConverting main model: {model_st} -> {out_main}")
        with SafetensorsReader(model_st) as r:
            specs = build_main_specs(r)
            meta = [
                ("general.architecture", GGUF_TYPE_STRING, "inkling2_loop"),
                ("general.name", GGUF_TYPE_STRING, "Inkling-2-Mini (official NVFP4 + Q8_0)"),
                ("general.alignment", GGUF_TYPE_UINT32, GGUF_ALIGNMENT),
                ("general.quantization_version", GGUF_TYPE_UINT32, 2),
            ]
            write_gguf(out_main, meta, specs)

    if not args.skip_draft and needed(out_draft):
        if not mtp_st.is_file():
            print(f"ERROR: missing {mtp_st}", file=sys.stderr)
            return 1
        print(f"\nConverting DSpark drafter: {mtp_st} -> {out_draft}")
        with SafetensorsReader(mtp_st) as r:
            specs = build_dspark_specs(r)
            meta = [
                ("general.architecture", GGUF_TYPE_STRING, "inkling2_dspark"),
                ("general.name", GGUF_TYPE_STRING, "DSpark Inkling-2-Mini (official Q8_0 + BF16)"),
                ("general.alignment", GGUF_TYPE_UINT32, GGUF_ALIGNMENT),
                ("general.quantization_version", GGUF_TYPE_UINT32, 2),
            ]
            write_gguf(out_draft, meta, specs)

    if not args.skip_vision and needed(out_vision):
        print(f"\nConverting vision encoder and projector: {model_st} -> {out_vision}")
        with SafetensorsReader(model_st) as reader:
            meta = [
                ("general.architecture", GGUF_TYPE_STRING, "clip"),
                ("general.name", GGUF_TYPE_STRING, "Inkling-2-Mini (official vision F16 + F32)"),
                ("general.alignment", GGUF_TYPE_UINT32, GGUF_ALIGNMENT),
                ("clip.vision.projector_type", GGUF_TYPE_STRING, "inkling2v"),
            ]
            write_gguf(out_vision, meta, build_vision_specs(reader))

    print("\nAll conversions complete!")
    return 0


def patch_tokenizer_for_loom(src_tokenizer: Path, dst_tokenizer: Path, llm_config: Path) -> None:
    """
    Create Loom-ready tokenizer.json from official thinkingmachines/Inkling-2-Mini
    tokenizer.json. loom-serve strips `special: true` added tokens from decoded text,
    so the 5 structural channel/tool tags listed in llm_config.json chat.nonspecial_tokens
    must be marked `special: false` so control.loom can split reasoning_content, content,
    and tool_calls cleanly (exact match to AMD loom-skills build.py).
    """
    with open(llm_config, encoding="utf-8") as f:
        chat = json.load(f).get("chat", {})
    nonspecial = set(chat.get("nonspecial_tokens", []))
    terminal = chat.get("terminal")

    with open(src_tokenizer, encoding="utf-8") as f:
        tok = json.load(f)

    patched = 0
    for entry in tok.get("added_tokens", []):
        if entry.get("content") in nonspecial:
            entry["special"] = False
            patched += 1
        elif terminal and entry.get("content") == terminal:
            entry["special"] = True

    dst_tokenizer.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dst_tokenizer.parent) as directory:
        tmp = Path(directory) / dst_tokenizer.name
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(tok, f, ensure_ascii=False, separators=(",", ":"))
        tmp.replace(dst_tokenizer)
    print(f"Wrote {dst_tokenizer} (marked {patched} structural channel tokens non-special for Loom)")


if __name__ == "__main__":
    raise SystemExit(main())
