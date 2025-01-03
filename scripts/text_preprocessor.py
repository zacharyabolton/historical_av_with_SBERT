"""
Loop through each document in the corpus and preprocess it in a similar
fashion to Bolton (2024):

Bolton, Z. 2024. True Love or Lost Cause. Gist
34bd09f76f94111ac0113fb5da1ea14e. Retrieved November 8, 2024 from
https://gist.github.com/zacharyabolton/34bd09f76f94111ac0113fb5da1ea14e

This preserves comparibility to this precursor study, as well as reduces
noise introduced by OCR in some of the collected works.

This routine converts all text to lowercase and removes all non-alphanumeric
characters.
"""
from nltk.probability import FreqDist
import os
import re
import argparse


def remove_non_alpha_symbols(input_string):
    """
    Remove all non-alphanumeric (a-z, and 0-9) and non-whitespace symbols.
    """

    # Based on limasxgoesto0's answer on StackOverflow at
    # https://stackoverflow.com/a/22521156 [14]
    regex = re.compile(r'[^a-zA-Z0-9\s]')
    alpha_string = regex.sub('', input_string)
    # End of adapted code

    # Reduce multiple spaces to one
    return ' '.join(alpha_string.split())


def preprocess(data_dir, canonical_class_labels):
    """
    Lowercase and remove all non-alphanumeric characters from an input text.

    :param data_dir: The relative path to the directory containing class
    based sub-directories for the input data corpus.
    :type data_dir: string
    :param canonical_class_labels: A list of strings representing the
    subdiretories named after thier respective canonical class names.
    :type canonical_class_labels: list
    """

    # Create new directory for preprocessed text
    preprocessed_dir = os.path.join(data_dir, 'preprocessed')
    os.makedirs(preprocessed_dir, exist_ok=True)

    # Set the path to the original undistorted data
    data_dir = os.path.join(data_dir, 'cleaned')

    # Loop through the sub-directories to collect samples in each input
    # class.
    for label in canonical_class_labels:
        # Create the path to the class samples.
        class_dir = os.path.join(data_dir, label)
        # Create an output path to the preprocessed class samples
        preprocessed_class_dir = os.path.join(preprocessed_dir, label)
        os.makedirs(preprocessed_class_dir, exist_ok=True)
        # Loop through each file in the class directory.
        for file in os.listdir(class_dir):
            # Only process `.txt` files.
            if file.endswith('.txt'):
                # Open the file.
                with open(os.path.join(class_dir, file)) as f:
                    # Read in the file's contents.
                    text = f.read()
                    # Lowercase the text
                    text = text.lower()
                    # Remove all non-alphanumeric characters
                    text = remove_non_alpha_symbols(text)
                    # Create path to new preprocessed file
                    preprocessed_text_file = os.path.join(preprocessed_class_dir, file)
                    # Save new preprocessed files
                    with open(preprocessed_text_file, 'w') as f:
                        f.write(text)


def run_preprocessor(args):
    data_dir = args.data_dir
    canonical_class_names = args.canonical_class_names

    assert type(data_dir), 'str'
    assert type(canonical_class_names), 'list'

    preprocess(data_dir, canonical_class_names)


if __name__ == '__main__':
    # Adapted from:
    # https://github.com/JacobTyo/Valla/blob/main/valla/methods/AA_MHC.py

    # get command line args
    parser = argparse.ArgumentParser(
        description='Distort text from an input directory.')

    # Set the data directory path
    parser.add_argument('-d',
                        '--data_dir',
                        type=str,
                        help='<Required> The relative path to the data root',
                        required=True)

    # Set the class conical names
    # Adapted from: https://stackoverflow.com/a/15753721
    parser.add_argument('-c',
                        '--canonical_class_names',
                        type=str,
                        nargs='+',
                        help='<Required> Names of class based subdirectories',
                        required=True)

    args = parser.parse_args()

    run_preprocessor(args)