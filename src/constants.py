"""
This file contains constants intended to be exposed globally.
"""
import os
from pathlib import Path

#############################################################################
# CONSTANTS
#############################################################################

# Adapted from https://stackoverflow.com/a/25389715
# & https://stackoverflow.com/a/53465812
# Accessed 2024-12-02
ROOT_DIR = os.path.dirname(Path(__file__).parent)
# https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
MODEL = 'sentence-transformers/all-MiniLM-L12-v2'
# Learning rate of 2e-05 as per Ibrahim et al. (2023) [8:10]
# This is intentionally very small as we are fine-tuning a pre-trained model
INITIAL_LEARNING_RATE = 2e-05
# p1 and p2 decision threshold, between which the model collapses values
# to 0.5, meaning "I don't know". These are set to 0.45 and 0.54
# respectively, matching Ibrahim et al. (2023) [8:9-11].
P1 = 0.45
P2 = 0.54