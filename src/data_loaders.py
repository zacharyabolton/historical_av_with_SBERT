"""
Adapted from Jacob Tyo (2022) [32]:
[32] Jacob Tyo. (JacobTyo). 2022. Valla: https://github.com/JacobTyo/Valla. Retrieved December 13, 2024 from https://github.com/JacobTyo/Valla/blob/f5022b8f90d909d530ddb205f6e35228e6f35cda/valla/dsets/loaders.py  # noqa: E501
"""
# A generic loader for the av data formats - read them in as a list of
# samples
import csv
import pandas as pd
from typing import List, Union
import sys
import os

csv.field_size_limit(sys.maxsize)


def get_txt_filenames(input_folder):
    """Get all .txt file names in a directory."""
    return [file
            for file in
            os.listdir(input_folder) if file.endswith(".txt")]


def combine_text_files(input_folder, files):
    """Combine all .txt files in an input directory into a single string."""
    text = ""
    for filename in files:
        if filename.endswith(".txt"):
            with open(os.path.join(input_folder, filename),
                      "r",
                      encoding="utf-8") as file:
                text = text + " ".join(file.readlines()) + " "
                return text


def av_as_pandas(data: List[List[Union[int, str]]]) -> pd.DataFrame:
    return pd.DataFrame(data, columns=['same/diff', 'text0', 'text1'])


def get_av_dataset(dataset_path: str) -> List[List[Union[int, str, str]]]:
    data = []
    with open(dataset_path, 'r', errors='ignore') as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i > 0:  # skip header
                data.append([int(line[0]), str(line[1]), str(line[2])])
    return data


def get_av_as_pandas(dataset_path: str) -> pd.DataFrame:
    return pd.read_csv(dataset_path,
                       header=0,
                       names=['same/diff', 'text0', 'text1'])