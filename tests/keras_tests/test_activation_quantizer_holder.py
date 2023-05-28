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
import os
import tempfile
import unittest

import numpy as np
import tensorflow as tf
from tensorflow import keras

from mct_quantizers.common.constants import ACTIVATION_HOLDER_QUANTIZER
from mct_quantizers.keras.activation_quantization_holder import KerasActivationQuantizationHolder
from mct_quantizers.keras.quantizers.activation_inferable_quantizers.activation_pot_inferable_quantizer import \
    ActivationPOTInferableQuantizer
from mct_quantizers.keras.quantizers.activation_inferable_quantizers.activation_symmetric_inferable_quantizer import \
    ActivationSymmetricInferableQuantizer


class TestKerasActivationQuantizationHolder(unittest.TestCase):

    def test_activation_quantization_holder_inference(self):
        num_bits = 3
        thresholds = [4.]
        signed = True

        quantizer = ActivationSymmetricInferableQuantizer(num_bits=num_bits,
                                                          threshold=thresholds,
                                                          signed=signed)
        model = keras.Sequential([KerasActivationQuantizationHolder(quantizer)])

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 50, 3) * 100 - 50, tf.float32)
        # Quantize tensor
        quantized_tensor = model(input_tensor)

        holder_config = model.layers[0].get_config()
        q_config = holder_config[ACTIVATION_HOLDER_QUANTIZER]['config']
        self.assertTrue(q_config['num_bits'] == num_bits)
        self.assertTrue(q_config['threshold'] == thresholds)
        self.assertTrue(q_config['signed'] == signed)

        # The maximal threshold is 4 using a signed quantization, so we expect all values to be between -4 and 4
        self.assertTrue(np.max( quantized_tensor) < thresholds[0], f'Quantized values should not contain values greater than maximal threshold ')
        self.assertTrue(np.min(quantized_tensor) >= -thresholds[0], f'Quantized values should not contain values lower than minimal threshold ')

        self.assertTrue(len(np.unique(quantized_tensor)) <= 2 ** num_bits, f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has {len(np.unique(quantized_tensor))} unique values')
        # Assert some values are negative (signed quantization)
        self.assertTrue(np.any(quantized_tensor < 0), f'Expected some values to be negative but quantized tensor is {quantized_tensor}')

    def test_activation_quantization_holder_save_and_load(self):
        num_bits = 3
        thresholds = [4.]
        signed = True

        quantizer = ActivationPOTInferableQuantizer(num_bits=num_bits,
                                                    threshold=thresholds,
                                                    signed=signed)
        model = keras.Sequential([KerasActivationQuantizationHolder(quantizer)])
        x = tf.ones((3, 3))
        model(x)

        _, tmp_h5_file = tempfile.mkstemp('.h5')
        keras.models.save_model(model, tmp_h5_file)
        loaded_model = keras.models.load_model(tmp_h5_file, {KerasActivationQuantizationHolder.__name__: KerasActivationQuantizationHolder,
                                                             ActivationPOTInferableQuantizer.__name__: ActivationPOTInferableQuantizer})
        os.remove(tmp_h5_file)
        loaded_model(x)
