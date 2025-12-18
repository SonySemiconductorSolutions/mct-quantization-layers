# Copyright 2023 Sony Semiconductor Solutions, Inc. All rights reserved.
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

import importlib.util
import sys
import os


def _import_mct_logger():
    """
    Import Logger and set_log_folder from model_compression_toolkit.logger
    without triggering model_compression_toolkit/__init__.py (to avoid circular imports).
    """
    module_name = "model_compression_toolkit.logger"
    
    # If already loaded, return from cache
    if module_name in sys.modules:
        module = sys.modules[module_name]
        return module.Logger, module.set_log_folder
    
    # Find model_compression_toolkit/logger.py in sys.path
    for path in sys.path:
        logger_path = os.path.join(path, "model_compression_toolkit", "logger.py")
        if os.path.exists(logger_path):
            spec = importlib.util.spec_from_file_location(module_name, logger_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module.Logger, module.set_log_folder
    
    raise ImportError("Cannot find model_compression_toolkit.logger")


Logger, set_log_folder = _import_mct_logger()
