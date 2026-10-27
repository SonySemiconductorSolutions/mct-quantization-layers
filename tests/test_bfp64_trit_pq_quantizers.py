import math
import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "mct_quantizers", "common")
    ),
)

from bfp64_trit_pq_quantizers import (
    Bfp64BlockInferableQuantizer,
    QuantizationMethod,
    QuantizationTarget,
    QuantumTritBitplaneInferableQuantizer,
)


class TestMctBfp64TritPqQuantizers(unittest.TestCase):
    def test_mark_quantizer_metadata(self):
        self.assertEqual(
            Bfp64BlockInferableQuantizer.quantization_target,
            QuantizationTarget.Activation,
        )
        self.assertEqual(
            Bfp64BlockInferableQuantizer.quantization_method,
            [QuantizationMethod.BFP64_BLOCK],
        )
        self.assertEqual(
            QuantumTritBitplaneInferableQuantizer.quantization_target,
            QuantizationTarget.Weights,
        )
        self.assertEqual(
            QuantumTritBitplaneInferableQuantizer.quantization_method,
            [QuantizationMethod.QUANTUM_TRIT_B158],
        )

    def test_bfp64_block_compression_and_fidelity(self):
        values = [math.sin(0.13 * i) * 3.5 for i in range(128)]
        q8 = Bfp64BlockInferableQuantizer(num_bits=8, block_size=64)
        blocks8 = q8.pack_blocks(values)
        self.assertEqual(len(blocks8), 2)
        # 65 bytes per 64-lane block vs 256 bytes FP32 = 3.938x compression
        self.assertEqual(blocks8[0].byte_size, 65)
        rec8 = q8(values)
        for orig, rec in zip(values, rec8):
            self.assertAlmostEqual(orig, rec, delta=0.05)

        q16 = Bfp64BlockInferableQuantizer(num_bits=16, block_size=64)
        rec16 = q16(values)
        for orig, rec in zip(values, rec16):
            self.assertAlmostEqual(orig, rec, delta=2e-4)

    def test_quantum_trit_compression_ratio(self):
        weights = [math.cos(0.19 * i) * 0.8 for i in range(128)]
        blocks = QuantumTritBitplaneInferableQuantizer(threshold_ratio=0.45).pack_blocks(
            weights
        )
        self.assertEqual(len(blocks), 2)
        for block in blocks:
            self.assertEqual(block.length, 64)
            # 20 B per 64 trits (2 x u64 bitmask + Q16 scale) vs 256 B FP32.
            self.assertEqual(block.byte_size, 20)
        packed_bytes = sum(b.byte_size for b in blocks)
        fp32_bytes = len(weights) * 4
        self.assertAlmostEqual(fp32_bytes / packed_bytes, 12.8, places=6)

    def test_quantum_trit_b158_bitplane_and_bfp64_dot(self):
        weights = [math.cos(0.19 * i) * 0.8 for i in range(128)]
        acts = [math.sin(0.11 * i) * 1.2 for i in range(128)]

        wq = QuantumTritBitplaneInferableQuantizer(threshold_ratio=0.45)
        aq = Bfp64BlockInferableQuantizer(num_bits=16, block_size=64)

        w_deq = wq(weights)
        a_deq = aq(acts)
        expected_dot = sum(w * a for w, a in zip(w_deq, a_deq))
        hw_dot = wq.infer_dot_bfp64(weights, acts, aq)
        self.assertAlmostEqual(hw_dot, expected_dot, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
