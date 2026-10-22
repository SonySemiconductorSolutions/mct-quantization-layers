# Copyright 2023 Sony Semiconductor Solutions, Inc. All rights reserved.
# Portions Copyright 2026 8b-is (https://8b.is).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""
MCT Quantization Layers × 8b-is (`https://8b.is`):
Omni `BFP-64`, Quantum 64-Trit Bitplane (`b1.58`), and `p/q` ("Counting Divided")
Inferable Quantizers for Constrained Edge Silicon & AI Sensors (IMX500 / NPU)
================================================================================

Brought to you by **Chris, Hue & the 8b-is team** (https://8b.is · WhatsApp: `+1 757-614-7542`):
  - `8b-is/alexiai` — Sovereign offline AI substrate (`GAIA-MLX-QUANT`, ternary b1.58,
    Quantum 64-Trit bitplanes, `BFP-64`, `wupz`)
  - `8b-is/Engram` — Conditional Memory via Scalable Lookup with `BFP-64` & `p/q`
  - `8b-is/deepsiper-enthea` — Hardware-aligned `BFP-64` & 64-trit bitplane engine

Overview
--------
This module extends Sony Semiconductor Solutions' `mct_quantizers` (`BaseInferableQuantizer`
and `@mark_quantizer`) with two hardware-friendly quantizer families that operate with
zero runtime dependencies (pure Python 3 stdlib + optional NumPy / PyTorch):

1. **`Bfp64BlockInferableQuantizer` (`QuantizationMethod.BFP64_BLOCK`)**:
   - Partitions weights or activations into 64-lane blocks (`BFP-64`) sharing a single
     8-bit base-2 exponent `shared_exp` plus signed `n_bits` (`i8` or `i16`) mantissas:
       $$x_i \approx m_i \cdot 2^{\text{shared\_exp} - (n_{\text{bits}} - 2)}$$
   - Compresses FP32 tensors by **`3.94x`** (`i8`, `65 B / 64 lanes`) while preserving
     full dynamic range without per-tensor clipping saturation.

2. **`QuantumTritBitplaneInferableQuantizer` (`QuantizationMethod.QUANTUM_TRIT_B158`)**:
   - Quantizes weights into ternary trits $\{-1, 0, +1\}$ packed into dual 64-bit
     bitplanes `(pos_mask: u64, neg_mask: u64)` per 64 weights (`16 bytes` bitmasks +
     `4 bytes` rational `Q16` scale = `20 bytes / 64 weights` = **`12.8x - 16x` compression**).
   - Replaces floating-point MACs on edge NPUs with pure bitwise `AND`/`OR`, `POPCNT`,
     and integer add/sub accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Sequence, Tuple

try:
    from mct_quantizers.common.base_inferable_quantizer import (
        BaseInferableQuantizer,
        QuantizationTarget,
        mark_quantizer,
    )
    from mct_quantizers.common.quant_info import QuantizationMethod
except ImportError:
    from base_inferable_quantizer import (
        BaseInferableQuantizer,
        QuantizationTarget,
        mark_quantizer,
    )
    from quant_info import QuantizationMethod

MASK64 = (1 << 64) - 1
Q16_ONE = 1 << 16


def _popcnt64(x: int) -> int:
    return (x & MASK64).bit_count()


def isqrt_u128(n: int) -> int:
    """Exact integer floor(sqrt(n)) via hardware shifts/adds."""
    return math.isqrt(max(0, int(n)))


@dataclass(frozen=True)
class PackedBfp64Block:
    """
    Hardware packet for a 64-lane `BFP-64` block:
      - `shared_exp`: 8-bit signed base-2 exponent
      - `n_bits`: bit-width of each signed integer mantissa (`8` or `16`)
      - `mantissas`: up to 64 signed integer mantissas
    """

    shared_exp: int
    n_bits: int
    mantissas: Tuple[int, ...]

    @property
    def frac_bits(self) -> int:
        return self.n_bits - 2

    @property
    def max_quant(self) -> int:
        return (1 << (self.n_bits - 1)) - 1

    @property
    def byte_size(self) -> int:
        bytes_per_elem = 1 if self.n_bits <= 8 else 2
        return 1 + len(self.mantissas) * bytes_per_elem

    def dequantize(self) -> List[float]:
        shift = self.shared_exp - self.frac_bits
        return [math.ldexp(float(m), shift) for m in self.mantissas]


@dataclass(frozen=True)
class PackedQuantumTritBlock64:
    """
    Hardware packet for 64 ternary (`b1.58`) weights:
      - `pos_mask`: 64-bit mask where bit `k` is 1 iff weight `k` is `+1`
      - `neg_mask`: 64-bit mask where bit `k` is 1 iff weight `k` is `-1`
      - `scale_q16`: rational `p/q` scale numerator over `q = 2^16`
      - `length`: number of valid elements (`<= 64`)
    """

    pos_mask: int
    neg_mask: int
    scale_q16: int
    length: int = 64

    @property
    def active_trits(self) -> int:
        """Exact L1 and L2-squared norm of the unscaled trit block via POPCNT."""
        return _popcnt64(self.pos_mask | self.neg_mask)

    def dequantize(self) -> List[float]:
        scale = self.scale_q16 / float(Q16_ONE)
        out: List[float] = []
        for k in range(self.length):
            bit = 1 << k
            if self.pos_mask & bit:
                out.append(scale)
            elif self.neg_mask & bit:
                out.append(-scale)
            else:
                out.append(0.0)
        return out

    def dot_bfp64(self, act_block: PackedBfp64Block) -> float:
        """
        Multiplier-free integer dot product between this 64-trit weight block
        and a 64-lane `BFP-64` activation block!
        """
        acc_int = 0
        n = min(self.length, len(act_block.mantissas))
        p = self.pos_mask & MASK64
        while p:
            lsb = p & -p
            idx = lsb.bit_length() - 1
            if idx < n:
                acc_int += act_block.mantissas[idx]
            p ^= lsb
        q = self.neg_mask & MASK64
        while q:
            lsb = q & -q
            idx = lsb.bit_length() - 1
            if idx < n:
                acc_int -= act_block.mantissas[idx]
            q ^= lsb
        combined_int = acc_int * self.scale_q16
        return math.ldexp(float(combined_int), act_block.shared_exp - act_block.frac_bits - 16)


@mark_quantizer(
    quantization_target=QuantizationTarget.Activation,
    quantization_method=[QuantizationMethod.BFP64_BLOCK],
    identifier="Bfp64BlockInferableQuantizer",
)
class Bfp64BlockInferableQuantizer(BaseInferableQuantizer):
    """
    MCT Inferable Quantizer implementing 8b-is Omni `BFP-64` block floating-point
    quantization for activations or weights (64 lanes per shared exponent).
    """

    def __init__(self, num_bits: int = 8, block_size: int = 64) -> None:
        super().__init__()
        if num_bits not in (8, 16):
            raise ValueError("Bfp64BlockInferableQuantizer supports num_bits in (8, 16)")
        self.num_bits = num_bits
        self.block_size = block_size

    def pack_blocks(self, values: Sequence[float]) -> List[PackedBfp64Block]:
        blocks: List[PackedBfp64Block] = []
        frac_bits = self.num_bits - 2
        max_q = (1 << (self.num_bits - 1)) - 1
        for start in range(0, len(values), self.block_size):
            chunk = [float(v) for v in values[start : start + self.block_size]]
            max_abs = max((abs(v) for v in chunk), default=0.0)
            if max_abs < 1e-30:
                blocks.append(
                    PackedBfp64Block(
                        shared_exp=-40,
                        n_bits=self.num_bits,
                        mantissas=tuple(0 for _ in chunk),
                    )
                )
            else:
                _, exp = math.frexp(max_abs)
                shared_exp = int(exp)
                inv_scale = math.ldexp(1.0, frac_bits - shared_exp)
                mants = tuple(
                    max(-max_q, min(max_q, int(round(v * inv_scale))))
                    for v in chunk
                )
                blocks.append(
                    PackedBfp64Block(
                        shared_exp=shared_exp,
                        n_bits=self.num_bits,
                        mantissas=mants,
                    )
                )
        return blocks

    def __call__(self, values: Sequence[float]) -> List[float]:
        """Emulate inference-time BFP-64 quantization."""
        out: List[float] = []
        for blk in self.pack_blocks(values):
            out.extend(blk.dequantize())
        return out


@mark_quantizer(
    quantization_target=QuantizationTarget.Weights,
    quantization_method=[QuantizationMethod.QUANTUM_TRIT_B158],
    identifier="QuantumTritBitplaneInferableQuantizer",
)
class QuantumTritBitplaneInferableQuantizer(BaseInferableQuantizer):
    """
    MCT Inferable Quantizer implementing 8b-is Quantum 64-Trit Bitplane (`b1.58`)
    weight quantization with `POPCNT` inner-product acceleration.
    """

    def __init__(self, threshold_ratio: float = 0.45) -> None:
        super().__init__()
        self.threshold_ratio = float(threshold_ratio)

    def pack_blocks(self, weights: Sequence[float]) -> List[PackedQuantumTritBlock64]:
        blocks: List[PackedQuantumTritBlock64] = []
        for start in range(0, len(weights), 64):
            chunk = [float(v) for v in weights[start : start + 64]]
            n = len(chunk)
            mean_abs = sum(abs(v) for v in chunk) / max(1, n)
            thresh = self.threshold_ratio * mean_abs
            pos_mask = 0
            neg_mask = 0
            active_sum = 0.0
            active_cnt = 0
            for k, val in enumerate(chunk):
                if val > thresh:
                    pos_mask |= 1 << k
                    active_sum += abs(val)
                    active_cnt += 1
                elif val < -thresh:
                    neg_mask |= 1 << k
                    active_sum += abs(val)
                    active_cnt += 1
            scale = (active_sum / active_cnt) if active_cnt > 0 else mean_abs
            scale_q16 = max(1, int(round(scale * Q16_ONE)))
            blocks.append(
                PackedQuantumTritBlock64(
                    pos_mask=pos_mask,
                    neg_mask=neg_mask,
                    scale_q16=scale_q16,
                    length=n,
                )
            )
        return blocks

    def __call__(self, weights: Sequence[float]) -> List[float]:
        """Emulate inference-time Quantum 64-Trit b1.58 quantization."""
        out: List[float] = []
        for blk in self.pack_blocks(weights):
            out.extend(blk.dequantize())
        return out

    def infer_dot_bfp64(
        self,
        weights: Sequence[float],
        activations: Sequence[float],
        act_quantizer: Bfp64BlockInferableQuantizer,
    ) -> float:
        """
        Execute a complete hardware-emulated `QuantumTrit x BFP-64` dot product
        using bitplane scans and integer accumulation.
        """
        w_blocks = self.pack_blocks(weights)
        a_blocks = act_quantizer.pack_blocks(activations)
        return sum(wb.dot_bfp64(ab) for wb, ab in zip(w_blocks, a_blocks))
