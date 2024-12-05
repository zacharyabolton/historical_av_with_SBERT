"""
Dataset functions and classes.
"""
import random
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import os
from definitions import ROOT_DIR


def create_training_pairs(a_texts, nota_texts):
    """
    Create training pairs as DataFrame.

    :returns: Training pairs as a Pandas DataFrame.
    :rtype: pandas.core.frame.DataFrame
    """
    # Create positive pairs (same author)
    positive_pairs = []
    for i in range(len(a_texts)):
        for j in range(i + 1, len(a_texts)):  # Note: i + 1 prevents self-pairs
            positive_pairs.append((a_texts[i], a_texts[j], 1))

    # Create negative pairs (different authors)
    negative_pairs = []
    for a_text in a_texts:
        for imposter_text in nota_texts:
            negative_pairs.append((a_text, imposter_text, 0))

    # Balance positive and negative samples
    min_samples = min(len(positive_pairs), len(negative_pairs))
    positive_pairs = random.sample(positive_pairs, min_samples)
    negative_pairs = random.sample(negative_pairs, min_samples)

    # Combine and shuffle
    all_pairs = positive_pairs + negative_pairs

    # Create a dataframe of our labeled pairs
    data = pd.DataFrame(
        all_pairs,
        columns=['Sentence A', 'Sentence B', 'Label']
    )
    return data


class CustomDataset(Dataset):
    """
    Custom torch dataset.

    :param data_dir: The directory of the data files.
    :type data_dir: string
    """

    def __init__(self, data_dir):
        project_root = ROOT_DIR
        self._data_dir = os.path.join(project_root, data_dir)

        self._A_dir = os.path.join(self._data_dir, "A")
        self._notA_dir = os.path.join(self._data_dir, "notA")
        self._U_dir = os.path.join(self._data_dir, "U")

        self._A_text = self._get_text(self._A_dir)
        self._notA_text = self._get_text(self._notA_dir)
        self._U_text = self._get_text(self._U_dir)

        self._A_tokens = self._tokenize(self._A_text)
        self._notA_tokens = self._tokenize(self._notA_text)
        self._U_tokens = self._tokenize(self._U_text)

    def __len__(self):
        return np.nan

    def __getitem__(self):
        return np.nan

    def _get_text(self, dir):
        """
        Concatenates the contents from all `.txt` files in a given directory.

        :param dir: The directory to get text from.
        :type dir: string
        :rtype: string
        """
        text = ""
        dir_contents = os.listdir(dir)
        dir_contents_sorted = sorted(dir_contents)
        for file in dir_contents_sorted:
            if file.endswith(".txt"):
                with open(os.path.join(dir, file), 'r') as file:
                    text += file.read() + " "
        return text[0:-1]

    def _tokenize(self, text):
        """
        Tokenize input text.

        :param text:
        :type text: string
        :rtype: transformers.tokenization_utils_base.BatchEncoding
        """
        model = 'sentence-transformers/all-MiniLM-L12-v2'
        tokenizer = AutoTokenizer.from_pretrained(model)

        # Tokenize sentences
        encoded_input = tokenizer(
            text,
            padding=False,
            truncation=False
        )

        return encoded_input

    def _chunk_tokens(self, tokens, chunk_size):
        """
        Chunk input tokens.

        :param tokens: Tokenized text
        """
        


    #     """
    #     Convert a string into an array of contiguous substrings of length `chunk_size`.
    #     """
    #     # Adapted from:
    #     # Bolton, Z. 2024. True Love or Lost Cause.
    #     # Gist 34bd09f76f94111ac0113fb5da1ea14e.
    #     # Retrieved November 8, 2024 from
    #     # https://gist.github.com/zacharyabolton/34bd09f76f94111ac0113fb5da1ea14e

    #     # Generate chunks
    #     chunks = [
    #         input_string[x:x+chunk_size]
    #         for x
    #         in range(0, len(input_string), chunk_size)
    #     ]
    #     # Pad the final chunk if shorter than 2500
    #     chunks[-1] += " " * (chunk_size - len(chunks[-1]))
    #     return chunks