"""
This module creates test data for the `LILADataset` test suite and the
`SiameseSBERT` test suite. It writes files to disk in the format expected
by both in order to run tests on toy data.
"""
import pandas as pd
import os
import shutil


def generate_test_data(test_suite_dir):
    """
    Create a mock dataset that can test some edge cases
    Dataset characteristics (minus omitted work `A-4.txt`):
        Word Frequencies:
        - elit:        1
        - adipiscing:  2
        - consectetur: 3
        - amet:        4
        - s9t:         5
        - 42:          6
        - ipsum:       7
        - lorem:       8
        - foo:         24
        Vocab size: 9
        Num words: 60
        Average doc length: 5
        Shortest doc: 1
        Longest doc: 15

    :param test_suite_dir: <Required> Path to place the parent directory of
    all the test data.
    :type test_suite_dir: str

    :returns: Tuple of `(dataset, paths)` which is an array containing the
    raw text of the `.txt` files written to disk, and a dict containing
    paths to the different subdirectories in the test directory and the
    metadata `.csv` file.
    :rtype: dict
    """
    paths = {}
    dataset = [
        'lorem ipsum',                                      # A-0
        'consectetur lorem',                                # A-1
        '42 adipiscing ipsum 42 s9t amet',                  # A-2
        'adipiscing elit amet',                             # A-3
        'NONE OF THIS SHOULD SHOW UP',                      # A-4
        'lorem 42 consectetur amet consectetur amet s9t',   # U-0
        'lorem',                                            # U-1
        '42 ipsum s9t',                                     # U-2
        'ipsum lorem',                                      # U-3
        '42 42 lorem s9t foo foo foo foo foo foo foo foo',  # notA-0
        'ipsum lorem s9t foo',                              # notA-1
        'ipsum ipsum foo foo foo foo foo foo foo foo foo'
        ' foo foo foo foo',                                 # notA-2
        'lorem foo foo'                                     # notA-3
    ]

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
    rows = [['A-0.txt', 'aauth', 'A author', 'mock genre 1',
             None, 'A', 1, False, len(dataset[0].split())],
            ['A-1.txt', 'aauth', 'A author', 'mock genre 1',
             None, 'A', 1, False, len(dataset[1].split())],
            ['A-2.txt', 'aauth', 'A author', 'mock genre 1',
             None, 'A', 1, False, len(dataset[2].split())],
            ['A-3.txt', 'aauth', 'A author', 'mock genre 2',
             None, 'A', 1, False, len(dataset[3].split())],
            ['A-4.txt', 'aauth', 'A author', 'mock genre 2',
             None, 'A', 1, True, len(dataset[4].split())],

            ['U-0.txt', None, None, 'mock genre 3',
             None, 'U', None, False, len(dataset[5].split())],
            ['U-1.txt', None, None, 'mock genre 3',
             None, 'U', None, False, len(dataset[6].split())],
            ['U-2.txt', None, None, 'mock genre 3',
             None, 'U', None, False, len(dataset[7].split())],
            ['U-3.txt', None, None, 'mock genre 3',
             None, 'U', None, False, len(dataset[8].split())],

            ['notA-0.txt', 'naauth', 'Imposter author',
             'mock genre 1', 'John', 'notA', 0, False,
             len(dataset[9].split())],
            ['notA-1.txt', 'naauth', 'Imposter author',
             'mock genre 2', 'John', 'notA', 0, False,
             len(dataset[10].split())],
            ['notA-2.txt', 'naauth', 'Imposter author',
             'mock genre 1', 'Jane', 'notA', 0, False,
             len(dataset[11].split())],
            ['notA-3.txt', 'naauth', 'Imposter author',
             'mock genre 2', 'Jane', 'notA', 0, False,
             len(dataset[12].split())]]
    for row in rows:
        df_TEST_metadata.loc[len(df_TEST_metadata)] = row

    # Create test data tree and save dataset and metadata table within
    data_dir = '../data'
    test_dir = os.path.join(data_dir, 'test')
    paths['test_suite_dir'] = os.path.join(test_dir, test_suite_dir)

    # Delete the whole tree just in case the `teardown_class` was
    # toggled off during testing
    # Adapted from: https://stackoverflow.com/a/13118112
    shutil.rmtree(paths['test_suite_dir'], ignore_errors=True)
    os.mkdir(paths['test_suite_dir'])
    # Create a mock source directory
    paths['undistorted_dir'] = os.path.join(paths['test_suite_dir'],
                                            'undistorted')
    os.mkdir(paths['undistorted_dir'])
    # Create the file path to the metadata
    paths['test_metadata_path'] = os.path.\
        join(paths['test_suite_dir'], 'metadata.csv')
    # Save the metadata
    df_TEST_metadata.to_csv(paths['test_metadata_path'], index=False)
    # Create mock sub-directories
    A_dir = os.path.join(paths['undistorted_dir'], 'A')
    notA_dir = os.path.join(paths['undistorted_dir'], 'notA')
    U_dir = os.path.join(paths['undistorted_dir'], 'U')
    os.mkdir(A_dir)
    os.mkdir(notA_dir)
    os.mkdir(U_dir)

    # Write mock data to appropriate files
    canonical_class_labels = sorted(os.listdir(paths['undistorted_dir']))
    rows_idx = 0
    for i, canonical_class in enumerate(canonical_class_labels):
        num_files = sum([1 for i
                         in rows
                         if i[5] == canonical_class])
        for j, test_file in enumerate(range(num_files)):
            if rows[rows_idx + j][7] is not True:
                file_name = f"{canonical_class}-{test_file}.txt"
                file_path = os.path.join(paths['undistorted_dir'],
                                         canonical_class,
                                         file_name)
                with open(file_path, 'w') as f:
                    f.write(dataset[rows_idx + j])
        rows_idx += num_files

    return dataset, paths


def destroy_test_data(test_suite_dir):
    """
    Remove the test dataset directory, and all it's sub directories and
    files.

    :param test_suite_dir: <Required> Path to the parent directory of all the
    test data.
    :type test_suite_dir: str
    """
    # Get full path to test data tree
    data_dir = '../data'
    test_dir = os.path.join(data_dir, 'test')
    path = os.path.join(test_dir, test_suite_dir)
    shutil.rmtree(path, ignore_errors=True)