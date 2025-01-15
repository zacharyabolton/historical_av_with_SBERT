"""
Loop through each document in the corpus and distort it with both DV-MA,
and DV-SA using two values of K resulting in 4 new versions of each
document.
The results should be a directory structure similar to the input
`data_directory` stored in subdirectores with the following naming
pattern:
"/DV-<MA | SA>-k-<{k}>/"
E.g. If `k` values of 300, and 3000, the directory for the output
relative to the input path, should be:
./normalized
  ├── DV-MA-k-300/
  │   ├── A
  │   │   ├── text1.txt
  │   │   ├── text2.txt
  │   │   ├── text3.txt
  │   │   └── ...
  │   ├── notA
  │   │   └── ...
  │   └── U
  │       └── ...
  ├── DV-MA-k-3000/
  │   └── ...
  ├── DV-SA-k-300/
  │   └── ...
  └── DV-SA-k-3000/
      └── ...
Note the `./` at the head, indicating the following structure should be
rooted at the same level as the input path. E.g. if the input data path is
`../data/test` the structure should be rooted at `../data/test`.
"""
from nltk.probability import FreqDist
import os
import re
import argparse


def get_W_k(data_dir, canonical_class_labels, k):
    """
    Get the top `k` most frequent words from a given corpus as a list of
    strings.

    :param data_dir: The relative path to the directory containing class
    based sub-directories for the input data corpus.
    :type data_dir: string
    :param canonical_class_labels: A list of strings representing the
    subdiretories named after thier respective canonical class names.
    :type canonical_class_labels: list
    :param k: The number of most frequent words to retrieve from the
    corpus.
    :type k: int
    :rtype list:
    """

    # Set the path to the original undistorted data
    undistorted_dir = os.path.join(data_dir, 'normalized', 'undistorted')

    # Setup the NLTK FeqDist object to calculate the occurances of all
    # words in the corpus.
    fdist = FreqDist()

    # Loop through the sub-directories to collect samples in each input
    # class.
    for label in canonical_class_labels:
        # Create the path to the class samples.
        class_dir = os.path.join(undistorted_dir, label)
        # Loop through each file in the class directory.
        for file in os.listdir(class_dir):
            # Only process `.txt` files.
            if file.endswith('.txt'):
                # Open the file.
                with open(os.path.join(class_dir, file)) as f:
                    # Read in the file's contents.
                    text = f.read()
                    # 'Tokenize' the text into words by splitting on
                    # whitespace.
                    words = text.split()
                    # Loop through all the words in the text.
                    for word in words:
                        # Add or increment each word occurence to the
                        # frequency distribution object.
                        fdist[word.lower()] += 1

    # Retrieve the top `k` most frequent words from the frequency
    # distribution object as a list of strings.
    # adapted from:
    # https://www.geeksforgeeks.org/python-convert-a-list-of-tuples-into-dictionary/  # noqa: E501
    W_k = dict(fdist.most_common(n=k)).keys()

    return W_k


def dv_ma(text, W_k):
    """
    Replace words and numbers in an input text that are not in `W_k`
    (the `k` most frequent words in a chosen corpus) with strings of
    asterisks or hash symbols (respectively) of equivalent length.

    :param text: Input text to distort.
    :type text: string
    :param W_k: Top `k` most frequent words in a given corpus.
    :type W_k: list
    :rtype string:
    """

    # Tokenize `text` into words by splitting on whitespace.
    words = text.split()
    # Loop through words
    for i, word in enumerate(words):
        if (word.lower() in W_k):
            # Leave words not in `W_k` untouched
            continue
        else:
            # Distort words not in `W_k`
            words[i] = re.sub(r'[^\d]', '*', word)
            # Distort digits not in `W_k`
            words[i] = re.sub(r'[\d]', '#', words[i])
    return ' '.join(words)


def dv_sa(text, W_k):
    """
    Replace words and numbers in an input text that are not in `W_k` (the
    `k` most frequent words in a chosen corpus) with a single asterisk or
    hash symbol (respectively).

    :param text: Input text to distort.
    :type text: string
    :param W_k: Top `k` most frequent words in a given corpus.
    :type W_k: list
    :rtype string:
    """

    # Tokenize `text` into words by splitting on whitespace.
    words = text.split()
    # Loop through words
    for i, word in enumerate(words):
        if (word.lower() in W_k):
            # Leave words not in `W_k` untouched
            continue
        else:
            # Distort words not in `W_k`
            words[i] = re.sub(r'[^\d]+', '*', word)
            # Distort digits not in `W_k`
            words[i] = re.sub(r'[\d]+', '#', words[i])
    return ' '.join(words)


def distort_text(data_dir, canonical_class_names, ks):
    """
    Main worker routine of text_distorter.py script.
    Creates directories for new 'views' of distorted corpus, reads in
    corpus to distort, and saves to appropriate sub-directories.

    :param data_dir: Path to the parent dir of the source dataset.
    :type data_dir: str
    :param canonical_class_names: Names of subdirectories containing data
    by class.
    :type canonical_class_names: list
    :param ks: Distortion values `k`.
    :type ks: list
    """
    # Main loop
    for k in ks:
        # Get the top `k` most frequent words from the entire corpus.
        W_k = get_W_k(data_dir, canonical_class_names, k)
        # Create two target directories for this value of `k`
        dv_ma_k_path = os.path.join(data_dir, 'normalized',
                                    f'DV-MA-k-{k}')
        os.makedirs(dv_ma_k_path, exist_ok=True)
        dv_sa_k_path = os.path.join(data_dir, 'normalized',
                                    f'DV-SA-k-{k}')
        os.makedirs(dv_sa_k_path, exist_ok=True)
        # Loop through the canonical class names
        for canonical_class in canonical_class_names:
            # Get the path to the directory for that class
            class_directory = os.path.join(data_dir, 'normalized',
                                           'undistorted', canonical_class)
            # Create new class based subdirectories for output
            dv_ma_k_class_path = os.path.join(dv_ma_k_path,
                                              canonical_class)
            os.makedirs(dv_ma_k_class_path, exist_ok=True)
            dv_sa_k_class_path = os.path.join(dv_sa_k_path,
                                              canonical_class)
            os.makedirs(dv_sa_k_class_path, exist_ok=True)
            # Loop through each file in the sub class directory
            for file in os.listdir(class_directory):
                # Ignore non `.txt` files
                if file.endswith('.txt'):
                    # Get full path to file
                    file_path = os.path.join(class_directory, file)
                    # Read in the file
                    with open(file_path, 'r') as f:
                        # Save contents to variable
                        text = f.read()
                        # Distort the text with the Multiple Askterisks
                        # algo
                        dv_ma_text = dv_ma(text, W_k)
                        # Distort the text with the Single Askterisk algo
                        dv_sa_text = dv_sa(text, W_k)
                    # Get path to new distorted files
                    ma_file_path = os.path.join(dv_ma_k_class_path, file)
                    sa_file_path = os.path.join(dv_sa_k_class_path, file)
                    # Save new distorted files
                    with open(ma_file_path, 'w') as f:
                        f.write(dv_ma_text)
                    with open(sa_file_path, 'w') as f:
                        f.write(dv_sa_text)


def run_text_distorter(args):
    """
    Wrapper for main worker routine of text_distorter.py script.

    :param args: argparse object coming from the CLI which bundles the
    `data_dir`, `canonical_class_names`, and `k_values` parameters for use
    in the primary worker loop for the text_distorter script.
    :type args: argparse.Namespace
    """
    data_dir = args.data_dir
    data_dir = data_dir
    canonical_class_names = args.canonical_class_names
    ks = args.k_values

    assert type(data_dir) is str
    assert type(canonical_class_names) is list
    assert type(ks) is list

    distort_text(data_dir, canonical_class_names, ks)


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
                        help=('<Required> The relative path to the data'
                              ' root'),
                        required=True)

    # Set the class conical names
    # Adapted from: https://stackoverflow.com/a/15753721
    parser.add_argument('-c',
                        '--canonical_class_names',
                        type=str,
                        nargs='+',
                        help=('<Required> Names of class based'
                              ' subdirectories'),
                        required=True)

    # # Set the different values of `k`
    # ks = [300, 3000]
    parser.add_argument('-k',
                        '--k_values',
                        type=int,
                        nargs='+',
                        help='<Required> `k` values to distort with',
                        required=True)

    args = parser.parse_args()

    run_text_distorter(args)