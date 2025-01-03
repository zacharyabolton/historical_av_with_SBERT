import os
import sys
from definitions import ROOT_DIR

# Add src directory to sys.path
# Adapted from Taras Alenin's answer on StackOverflow at:
# https://stackoverflow.com/a/55623567
src_path = os.path.join(ROOT_DIR, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import custom modules
from dataset import CustomDataset  # noqa: E402


class TestPyTorchDataset:
    """
    """
    @classmethod
    def setup_class(cls):
        """
        """
        pass

    @classmethod
    def teardown_class(cls):
        """
        """
        pass