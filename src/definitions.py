"""
This file contains constants intended to be exposed globally.
"""
import os
from pathlib import Path

# Adapted from https://stackoverflow.com/a/25389715
# & https://stackoverflow.com/a/53465812
# Accessed 2024-12-02
ROOT_DIR = os.path.dirname(Path(__file__).parent)
MODEL = 'sentence-transformers/all-MiniLM-L12-v2'