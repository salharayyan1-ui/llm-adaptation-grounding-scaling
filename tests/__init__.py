"""Tests package initialization.
Ensures mock PyTorch is installed if real PyTorch is not available,
allowing all tests across task1, task2, and task3 to run safely without PyTorch.
"""

from tests.mock_torch import setup_mock_torch
setup_mock_torch()
