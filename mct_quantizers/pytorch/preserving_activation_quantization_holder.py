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

from mct_quantizers.common.base_inferable_quantizer import BaseInferableQuantizer
from mct_quantizers.common.constants import FOUND_TORCH, PRESERVING_ACTIVATION_HOLDER_QUANTIZER
from mct_quantizers.pytorch.activation_quantization_holder import PytorchActivationQuantizationHolder
from mct_quantizers.logger import Logger

if FOUND_TORCH:
    import torch

    class PytorchPreservingActivationQuantizationHolder(PytorchActivationQuantizationHolder):
        """
        Pytorch module to hold an activation quantizer and quantize during inference.
        """
        def __init__(self,
                     activation_holder_quantizer: BaseInferableQuantizer,
                     **kwargs):
            """

            Args:
                activation_holder_quantizer: Quantizer to use during inference.
                **kwargs: Key-word arguments for the base layer
            """

            super(PytorchPreservingActivationQuantizationHolder, self).__init__(activation_holder_quantizer=activation_holder_quantizer,
                                                                                **kwargs)
            self.activation_holder_quantizer.initialize_quantization(None,
                                                                     PRESERVING_ACTIVATION_HOLDER_QUANTIZER + "_out",
                                                                     self)

        def forward(self, inputs):
            """
            Quantizes the input tensor using the activation quantizer of class PytorchPreservingActivationQuantizationHolder.

            Args:
                inputs: Input tensors to quantize with the activation quantizer.

            Returns: Output of the input tensor.

            """
            return inputs

else:
    class PytorchPreservingActivationQuantizationHolder:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            Logger.critical('Installing Pytorch is mandatory '
                            'when using PytorchPreservingActivationQuantizationHolder. '
                            'Could not find the torch package.')  # pragma: no cover