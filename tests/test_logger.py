#  Copyright 2024 Sony Semiconductor Solutions, Inc. All rights reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================


import unittest
from unittest.mock import patch, MagicMock

from mct_quantizers.logger import Logger, set_log_folder

class TestLogger(unittest.TestCase):
    def setUp(self):
        self.log_folder = "test_logs"
        self.log_level = Logger.DEBUG
        self.log_message = "Test message"

    @patch('pathlib.Path.mkdir')
    @patch('os.path.exists')
    def test_check_path_create_dir(self, mock_exists, mock_mkdir):
        mock_exists.return_value = False
        Logger._Logger__check_path_create_dir(self.log_folder)  # Using the mangled name
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('mct_quantizers.logger.Logger.get_logger')
    def test_set_logger_level(self, mock_get_logger):
        logger_mock = MagicMock()
        mock_get_logger.return_value = logger_mock
        Logger.set_logger_level(self.log_level)
        logger_mock.setLevel.assert_called_once_with(self.log_level)

    @patch('mct_quantizers.logger.Logger.get_logger')
    def test_set_handler_level(self, mock_get_logger):
        logger_mock = MagicMock()
        handler1_mock = MagicMock()
        handler2_mock = MagicMock()
        logger_mock.handlers = [handler1_mock, handler2_mock]
        mock_get_logger.return_value = logger_mock
        Logger.set_handler_level(self.log_level)
        handler1_mock.setLevel.assert_called_once_with(self.log_level)
        handler2_mock.setLevel.assert_called_once_with(self.log_level)

    @patch('mct_quantizers.logger.logging.getLogger')
    def test_get_logger(self, mock_get_logger):
        Logger.get_logger()
        mock_get_logger.assert_called_once_with('MCT Quantizers')

    @patch('mct_quantizers.logger.logging.StreamHandler')
    @patch('mct_quantizers.logger.Logger.get_logger')
    def test_set_stream_handler(self, mock_get_logger, mock_stream_handler):
        logger_mock = MagicMock()
        stream_handler_mock = MagicMock()
        mock_stream_handler.return_value = stream_handler_mock
        mock_get_logger.return_value = logger_mock
        logger_mock.handlers = []
        
        Logger.set_stream_handler()
        
        mock_stream_handler.assert_called_once()
        logger_mock.addHandler.assert_called_once_with(stream_handler_mock)

    @patch('mct_quantizers.logger.isinstance')
    @patch('mct_quantizers.logger.Logger.get_logger')
    def test_set_stream_handler_already_exists(
            self,
            mock_get_logger,
            mock_isinstance):
        logger_mock = MagicMock()
        existing_handler = MagicMock()
        # Mock isinstance to return True, indicating handler exists
        mock_isinstance.return_value = True
        mock_get_logger.return_value = logger_mock
        logger_mock.handlers = [existing_handler]
        
        Logger.set_stream_handler()
        
        # Should not add a new handler if one already exists
        # addHandler should not be called since handler already exists
        logger_mock.addHandler.assert_not_called()

    @patch('mct_quantizers.logger.Logger.get_logger')
    @patch('mct_quantizers.logger.logging.FileHandler')
    def test_set_log_file(self, mock_file_handler, mock_get_logger):
        logger_mock = MagicMock()
        file_handler_mock = MagicMock()
        mock_file_handler.return_value = file_handler_mock
        mock_get_logger.return_value = logger_mock
        Logger.set_log_file(self.log_folder)
        mock_file_handler.assert_called_once()
        logger_mock.addHandler.assert_called_once_with(file_handler_mock)
        # Verify that setLevel is NOT called on the handler
        file_handler_mock.setLevel.assert_not_called()

    @patch('mct_quantizers.logger.logging.shutdown')
    def test_shutdown(self, mock_shutdown):
        Logger.shutdown()
        mock_shutdown.assert_called_once()
        self.assertIsNone(Logger.LOG_PATH)

    @patch('mct_quantizers.logger.Logger.get_logger')
    def test_critical(self, mock_get_logger):
        logger_mock = MagicMock()
        mock_get_logger.return_value = logger_mock
        with self.assertRaises(Exception) as context:
            Logger.critical(self.log_message)
        self.assertTrue(self.log_message in str(context.exception))
        logger_mock.critical.assert_called_once_with(self.log_message)

    @patch('mct_quantizers.logger.Logger.get_logger')
    def test_debug(self, mock_get_logger):
        logger_mock = MagicMock()
        mock_get_logger.return_value = logger_mock
        Logger.debug(self.log_message)
        logger_mock.debug.assert_called_once_with(self.log_message)

    @patch('mct_quantizers.logger.Logger.get_logger')
    def test_info(self, mock_get_logger):
        logger_mock = MagicMock()
        mock_get_logger.return_value = logger_mock
        Logger.info(self.log_message)
        logger_mock.info.assert_called_once_with(self.log_message)

    @patch('mct_quantizers.logger.Logger.get_logger')
    def test_warning(self, mock_get_logger):
        logger_mock = MagicMock()
        mock_get_logger.return_value = logger_mock
        Logger.warning(self.log_message)
        logger_mock.warning.assert_called_once_with(self.log_message)

    @patch('mct_quantizers.logger.Logger.get_logger')
    def test_error(self, mock_get_logger):
        logger_mock = MagicMock()
        mock_get_logger.return_value = logger_mock
        with self.assertRaises(Exception) as context:
            Logger.error(self.log_message)
        self.assertTrue(self.log_message in str(context.exception))
        logger_mock.error.assert_called_once_with(self.log_message)

    @patch('mct_quantizers.logger.Logger.set_stream_handler')
    @patch('mct_quantizers.logger.Logger.set_log_file')
    @patch('mct_quantizers.logger.Logger.set_logger_level')
    @patch('mct_quantizers.logger.Logger.set_handler_level')
    def test_set_log_folder(self,
                            mock_set_handler_level,
                            mock_set_logger_level,
                            mock_set_log_file,
                            mock_set_stream_handler):
        set_log_folder(self.log_folder, self.log_level)
        mock_set_stream_handler.assert_called_once()
        mock_set_log_file.assert_called_once_with(self.log_folder)
        mock_set_logger_level.assert_called_once_with(self.log_level)
        mock_set_handler_level.assert_called_once_with(self.log_level)

    @patch('mct_quantizers.logger.Logger.get_logger')
    def test_log_level_changes(self, mock_get_logger):
        """Test logging at different levels with level changes"""
        logger_mock = MagicMock()
        mock_get_logger.return_value = logger_mock
        handler_mock = MagicMock()
        logger_mock.handlers = [handler_mock]
        
        # First: Log all levels with default settings
        Logger.debug("DEBUG message 1")
        Logger.info("INFO message 1")
        Logger.warning("WARNING message 1")
        try:
            Logger.error("ERROR message 1")
        except Exception:
            pass  # Expected exception
        
        # Second: Set log level to WARNING
        Logger.set_logger_level(Logger.WARNING)
        Logger.set_handler_level(Logger.WARNING)
        Logger.debug("DEBUG message 2 (should not appear)")
        Logger.info("INFO message 2 (should not appear)")
        Logger.warning("WARNING message 2")
        try:
            Logger.error("ERROR message 2")
        except Exception:
            pass  # Expected exception
        
        # Third: Set log level to INFO
        Logger.set_logger_level(Logger.INFO)
        Logger.set_handler_level(Logger.INFO)
        Logger.debug("DEBUG message 3 (should not appear)")
        Logger.info("INFO message 3")
        Logger.warning("WARNING message 3")
        try:
            Logger.error("ERROR message 3")
        except Exception:
            pass  # Expected exception
        
        # Verify the logger methods were called with correct messages
        # First set of logs
        logger_mock.debug.assert_any_call("DEBUG message 1")
        logger_mock.info.assert_any_call("INFO message 1")
        logger_mock.warning.assert_any_call("WARNING message 1")
        logger_mock.error.assert_any_call("ERROR message 1")
        
        # Second set (all called, but level filtering happens in logger)
        logger_mock.debug.assert_any_call(
            "DEBUG message 2 (should not appear)")
        logger_mock.info.assert_any_call(
            "INFO message 2 (should not appear)")
        logger_mock.warning.assert_any_call("WARNING message 2")
        logger_mock.error.assert_any_call("ERROR message 2")
        
        # Third set
        logger_mock.debug.assert_any_call(
            "DEBUG message 3 (should not appear)")
        logger_mock.info.assert_any_call("INFO message 3")
        logger_mock.warning.assert_any_call("WARNING message 3")
        logger_mock.error.assert_any_call("ERROR message 3")
        
        # Verify setLevel was called for level changes
        logger_mock.setLevel.assert_any_call(Logger.WARNING)
        logger_mock.setLevel.assert_any_call(Logger.INFO)
        handler_mock.setLevel.assert_any_call(Logger.WARNING)
        handler_mock.setLevel.assert_any_call(Logger.INFO)


if __name__ == '__main__':
    unittest.main()

