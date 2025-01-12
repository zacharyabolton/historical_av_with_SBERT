"""
This module creates test data for the pytest test suite in `/tests/`. It
writes files to disk in the format expected in order to run tests on toy
data.
"""
import pandas as pd
import os
import sys
import shutil
from constants import ROOT_DIR

# Add src directory to sys.path
# Adapted from Taras Alenin's answer on StackOverflow at:
# https://stackoverflow.com/a/55623567
scripts_path = os.path.join(ROOT_DIR, 'scripts')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

# Import custom modules
from text_distorter import distort_text  # noqa: E402
from text_normalizer import normalize_text  # noqa: E402


def generate_test_data(test_suite_dir, dataset, metadata_rows):
    """
    :param test_suite_dir: <Required> Path to place the parent directory
    of all the test data.
    :type test_suite_dir: str

    :param dataset: <Required> A list of strings that will be written to
    files for testing.
    :type test_suite_dir: list

    :param metadata_rows: <Required> A list of lists representing metadata
    for each file that will be written to disk.
    :type metadata_rows: list

    :returns: A tuple of (`paths`, canonical_class_labels`). `paths` - A
    dict containing paths to the different subdirectories in the test
    directory and the metadata `.csv` file. `canonical_class_labels` - A
    list of canonical class labels (A, notA, U).
    :rtype: tuple
    """
    paths = {}
    # Create a mock metadata table
    metadata_columns = ['file',
                        'author_short',
                        'author',
                        'genre',
                        'imposter_for',
                        'canonical_class_label',
                        'class',
                        'omit',
                        'num_words']
    df_TEST_metadata = pd.DataFrame(columns=metadata_columns)
    for row in metadata_rows:
        df_TEST_metadata.loc[len(df_TEST_metadata)] = row

    # Create test data tree and save dataset and metadata table within
    data_dir = '../data'
    test_dir = os.path.join(data_dir, 'test')
    paths['test_suite_dir'] = os.path.join(test_dir, test_suite_dir)

    # Delete the whole tree just in case the `teardown_class` was
    # toggled off during testing
    # Adapted from: https://stackoverflow.com/a/13118112
    shutil.rmtree(paths['test_suite_dir'], ignore_errors=True)

    # Create main mock directory
    os.mkdir(paths['test_suite_dir'])
    # Create sub directories
    paths['cleaned_dir'] = os.path.join(paths['test_suite_dir'],
                                        'cleaned')
    os.mkdir(paths['cleaned_dir'])
    paths['normalized_dir'] = os.path.join(paths['test_suite_dir'],
                                           'normalized')
    os.mkdir(paths['normalized_dir'])
    paths['undistorted_dir'] = os.path.join(paths['normalized_dir'],
                                            'undistorted')
    os.mkdir(paths['undistorted_dir'])

    # Create mock canonical_class_label based sub-directories
    A_dir = os.path.join(paths['cleaned_dir'], 'A')
    notA_dir = os.path.join(paths['cleaned_dir'], 'notA')
    U_dir = os.path.join(paths['cleaned_dir'], 'U')
    os.mkdir(A_dir)
    os.mkdir(notA_dir)
    os.mkdir(U_dir)
    A_dir = os.path.join(paths['undistorted_dir'], 'A')
    notA_dir = os.path.join(paths['undistorted_dir'], 'notA')
    U_dir = os.path.join(paths['undistorted_dir'], 'U')
    os.mkdir(A_dir)
    os.mkdir(notA_dir)
    os.mkdir(U_dir)

    # Create the file path to the metadata
    paths['test_metadata_path'] = os.path.\
        join(paths['normalized_dir'], 'metadata.csv')
    # Save the metadata
    df_TEST_metadata.to_csv(paths['test_metadata_path'], index=False)

    # Write mock data to appropriate files
    canonical_class_labels = sorted(os.listdir(paths['cleaned_dir']))
    rows_idx = 0
    for i, canonical_class in enumerate(canonical_class_labels):
        num_files = sum([1 for i
                         in metadata_rows
                         if i[5] == canonical_class])
        for j, test_file in enumerate(range(num_files)):
            if metadata_rows[rows_idx + j][7] is not True:
                file_name = f"{canonical_class}-{test_file}.txt"
                file_path = os.path.join(paths['cleaned_dir'],
                                         canonical_class,
                                         file_name)
                with open(file_path, 'w') as f:
                    f.write(dataset[rows_idx + j])
        rows_idx += num_files

    # run the normalizing routine
    normalize_text(paths['test_suite_dir'], canonical_class_labels)

    # Run the text_distorter script with `k` values of 0, 2, and 8 (the
    # length of the vocabulary)
    ks = [0, 2, 8]
    distort_text(paths['test_suite_dir'], canonical_class_labels, ks)

    return paths, canonical_class_labels


def destroy_test_data(test_suite_dir):
    """
    Remove the test dataset directory, and all it's sub directories and
    files.

    :param test_suite_dir: <Required> Path to the parent directory of all
    the test data.
    :type test_suite_dir: str
    """
    # Get full path to test data tree
    data_dir = '../data'
    test_dir = os.path.join(data_dir, 'test')
    path = os.path.join(test_dir, test_suite_dir)
    shutil.rmtree(path, ignore_errors=True)