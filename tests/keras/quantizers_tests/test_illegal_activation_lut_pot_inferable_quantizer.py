# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
import unittest
import warnings

import numpy as np
import tensorflow as tf

from mct_quantizers.keras.quantizers.activation_inferable_quantizers.activation_lut_pot_inferable_quantizer import \
    ActivationLutPOTInferableQuantizer


class TestKerasActivationIllegalLutPotQuantizer(unittest.TestCase):

    def test_illegal_pot_lut_quantizer(self):
        with self.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=8,
                                               cluster_centers=np.asarray([25., 85.]),
                                               threshold=[3.],
                                               signed=True)
        self.assertEqual('Expected threshold to be power of 2 but is [3.0]', str(e.exception))

    def test_illegal_cluster_centers(self):
        with self.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=8,
                                               cluster_centers=np.asarray([25.9, 85.]),
                                               threshold=[4.],
                                               signed=True)
        self.assertEqual('Expected cluster centers to be integers', str(e.exception))

    def test_illegal_num_of_cluster_centers(self):
        cluster_centers = np.asarray([-25, 25, 12, 45, 11])
        with self.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=2,
                                               cluster_centers=cluster_centers,
                                               threshold=[4.],
                                               signed=True)
        self.assertEqual(
            f'Expected num of cluster centers to be less or equal than {2 ** 2} but got '
            f'{len(cluster_centers)}', str(e.exception))

    def test_illegal_cluster_centers_range_(self):
        with self.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=2,
                                               cluster_centers=np.asarray([50]),
                                               threshold=[4.],
                                               signed=True,
                                               multiplier_n_bits=3)
        self.assertEqual('Expected cluster centers in the quantization range', str(e.exception))

    def test_illegal_num_bit_bigger_than_multiplier_n_bits(self):
        with self.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=10,
                                               cluster_centers=np.asarray([25]),
                                               threshold=[4.],
                                               signed=True,
                                               multiplier_n_bits=8)
        self.assertEqual('Look-Up-Table bit configuration has 10 bits. It must be less then 8'
                                             , str(e.exception))

    def test_warning_num_bit_equal_multiplier_n_bits(self):
        with warnings.catch_warnings(record=True) as w:
            ActivationLutPOTInferableQuantizer(num_bits=8,
                                               cluster_centers=np.asarray([25]),
                                               threshold=[4.],
                                               signed=True,
                                               multiplier_n_bits=8)
        self.assertTrue(
            'Num of bits equal to multiplier n bits, Please be aware LUT quantizier may be inefficient '
            'in that case, consider using SymmetricInferableQuantizer instead'
            in [str(warning.message) for warning in w])

    def test_illegal_num_of_thresholds(self):
        with self.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=3,
                                               cluster_centers=np.asarray([25]),
                                               threshold=[4., 2.],
                                               signed=True,
                                               multiplier_n_bits=8)
        self.assertEqual('In per-tensor quantization threshold should be of length 1 but is 2',
                                             str(e.exception))

    def test_illegal_threshold_type(self):
        with self.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=3,
                                               cluster_centers=np.asarray([25]),
                                               threshold=np.asarray([4.]),
                                               signed=True,
                                               multiplier_n_bits=8)
            self.assertEqual(
                'Expected threshold to be of type list but is <class \'numpy.ndarray\'>', str(e.exception))

    def test_illegal_signed_cluster_centers(self):
        with self.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=8,
                                               cluster_centers=np.asarray([-25., 85.]),
                                               threshold=[2.],
                                               signed=False)
        self.assertEqual('Expected unsigned cluster centers in unsigned activation quantization ',
                                             str(e.exception))

    def test_lut_pot_signed_inferable_quantizer(self):
        cluster_centers = np.asarray([-25, 25])
        thresholds = [4.]
        num_bits = 3
        signed = True
        multiplier_n_bits = 8
        eps = 1e-8

        quantizer = ActivationLutPOTInferableQuantizer(num_bits=num_bits,
                                                       cluster_centers=cluster_centers,
                                                       signed=signed,
                                                       threshold=thresholds,
                                                       multiplier_n_bits=
                                                       multiplier_n_bits,
                                                       eps=eps)

        thresholds = np.asarray(thresholds)

        # check config
        quantizer_config = quantizer.get_config()
        self.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.assertTrue(np.all(quantizer_config['cluster_centers'] == cluster_centers))
        self.assertTrue(quantizer_config['threshold'] == thresholds)
        self.assertTrue(quantizer_config['signed'] == signed)
        self.assertTrue(quantizer_config['multiplier_n_bits'] == multiplier_n_bits)
        self.assertTrue(quantizer_config['eps'] == eps)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 50, 3) * 100 - 50, tf.float32)
        quantized_tensor = quantizer(input_tensor)

        # Using a signed quantization, so we expect all values to be between -abs(max(threshold))
        # and abs(max(threshold))

        max_threshold = np.max(np.abs(thresholds))
        delta_threshold = 1 / (2 ** (multiplier_n_bits - 1))

        self.assertTrue(np.max(
            quantized_tensor) <= max_threshold - delta_threshold, f'Quantized values should not contain values greater '
                                                                  f'than maximal threshold ')
        self.assertTrue(np.min(
            quantized_tensor) >= -max_threshold, f'Quantized values should not contain values lower than minimal '
                                                 f'threshold ')

        self.assertTrue(len(np.unique(quantized_tensor)) <= 2 ** num_bits,
                                  f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                                  f'{len(np.unique(quantized_tensor))} unique values')

        # Check quantized tensor assigned correctly
        clip_max = 2 ** (multiplier_n_bits - 1) - 1
        clip_min = -2 ** (multiplier_n_bits - 1)

        quant_tensor_values = (cluster_centers / (2 ** (multiplier_n_bits - int(signed)))) * thresholds

        self.assertTrue(np.all(np.unique(quantized_tensor) == np.sort(quant_tensor_values)))

        # Check quantized tensor assigned correctly
        tensor = tf.clip_by_value((input_tensor / (thresholds + eps)) * (2 ** (num_bits - 1)),
                                  clip_value_max=clip_max, clip_value_min=clip_min)
        tensor = tf.expand_dims(tensor, -1)
        expanded_cluster_centers = cluster_centers.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
        cluster_assignments = tf.argmin(tf.abs(tensor - expanded_cluster_centers), axis=-1)
        centers = tf.gather(cluster_centers.flatten(), cluster_assignments)

        self.assertTrue(np.all(centers / (2 ** (multiplier_n_bits - 1)) * thresholds == quantized_tensor),
                                  "Quantized tensor values weren't assigned correctly")

        # Assert some values are negative (signed quantization)
        self.assertTrue(np.any(quantized_tensor < 0),
                                  f'Expected some values to be negative but quantized tensor is {quantized_tensor}')

    def test_lut_pot_unsigned_inferable_quantizer(self):
        cluster_centers = np.asarray([25, 85])
        thresholds = [2.]
        num_bits = 3
        signed = False
        multiplier_n_bits = 8
        eps = 1e-8

        quantizer = ActivationLutPOTInferableQuantizer(num_bits=num_bits,
                                                       cluster_centers=cluster_centers,
                                                       signed=signed,
                                                       threshold=thresholds,
                                                       multiplier_n_bits=
                                                       multiplier_n_bits,
                                                       eps=eps)
        thresholds = np.asarray(thresholds)

        # check config
        quantizer_config = quantizer.get_config()
        self.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.assertTrue(np.all(quantizer_config['cluster_centers'] == cluster_centers))
        self.assertTrue(quantizer_config['threshold'] == thresholds)
        self.assertTrue(quantizer_config['signed'] == signed)
        self.assertTrue(quantizer_config['multiplier_n_bits'] == multiplier_n_bits)
        self.assertTrue(quantizer_config['eps'] == eps)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 50, 3) * 100 - 50, tf.float32)
        quantized_tensor = quantizer(input_tensor)

        # Using a unsigned quantization, so we expect all values to be between 0
        # and abs(max(threshold))

        max_threshold = np.max(np.abs(thresholds))
        delta_threshold = 1 / (2 ** multiplier_n_bits)

        self.assertTrue(np.max(
            quantized_tensor) <= max_threshold - delta_threshold, f'Quantized values should not contain values greater '
                                                                  f'than maximal threshold ')
        self.assertTrue(np.min(
            quantized_tensor) >= 0, f'Quantized values should not contain values lower than 0')

        self.assertTrue(len(np.unique(quantized_tensor)) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(np.unique(quantized_tensor))} unique values')

        # Check quantized tensor assigned correctly
        clip_max = 2 ** multiplier_n_bits - 1
        clip_min = 0

        quant_tensor_values = (cluster_centers / (2 ** multiplier_n_bits)) * thresholds

        self.assertTrue(np.all(np.unique(quantized_tensor) == np.sort(quant_tensor_values)))

        # Check quantized tensor assigned correctly
        tensor = tf.clip_by_value((input_tensor / (thresholds + eps)) * (2 ** multiplier_n_bits),
                                  clip_value_max=clip_max, clip_value_min=clip_min)
        tensor = tf.expand_dims(tensor, -1)

        expanded_cluster_centers = cluster_centers.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
        cluster_assignments = tf.argmin(tf.abs(tensor - expanded_cluster_centers), axis=-1)
        centers = tf.gather(cluster_centers.flatten(), cluster_assignments)

        self.assertTrue(np.all(centers / (2 ** multiplier_n_bits) * thresholds == quantized_tensor),
                        "Quantized tensor values weren't assigned correctly")

        # Assert all values are non-negative (unsigned quantization)
        self.assertTrue(np.all(quantized_tensor >= 0),
                        f'Expected all values to be non-negative but quantized tensor is {quantized_tensor}')
