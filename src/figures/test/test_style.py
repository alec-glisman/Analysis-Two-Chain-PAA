"""
Author: Alec Glisman (GitHub: @alec-glisman)
Date: 2023-03-28
Description: Test the style module.
"""
# pylint: disable=unused-import

from pathlib import Path
import sys

import pytest

# add local src directory to path
sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

# Local internal dependencies
from figures.style import set_style  # noqa: E402


def test_set_style():
    """
    Verify that the style can be set without error.
    """
    set_style()
