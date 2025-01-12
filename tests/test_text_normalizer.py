"""
This test suite tests this project's implementation of text normalization
procedure from Bolton (2024) [5] (minus numeral removal):

[5] Bolton, Z. 2024. True Love or Lost Cause. Gist
34bd09f76f94111ac0113fb5da1ea14e. Retrieved November 8, 2024 from
https://gist.github.com/zacharyabolton/34bd09f76f94111ac0113fb5da1ea14e
"""
import os
# Import custom modules
from generate_test_data import generate_test_data, destroy_test_data  # noqa: E402


class TestPreprocessing:
    """
    A unified class to allow for easy setup and teardown of global and
    reused data and objects, and sharing of common methods. See
    `setup_class`, `teardown_class`.
    """
    @classmethod
    def setup_class(cls):
        """
        Setup global and reused test data and objects.
        """
        # Create a mock dataset to test normalization edge cases
        cls.dataset = [
            'here is some text that should not be touched',      # a-0
            'here is some text that should be touched',          # a-1
            'here are s0me 1ntegers',                            # a-2
            'and sp*c|al cháracter$',                            # a-3
            """here
            is  some     white space""",                         # u-0
            'and back to untouched text',                        # u-1
            """
            white space
            capitalizations
            **and special chars**!@#$%^&*()_-+=\\|~`
            """,                                                 # u-2
            'white space      and capitalizations',              # u-3
            'white space and special chars %#^&!@*!',         # nota-0
            'caps and spe^#@&@*!*(!cial ch372ars ',           # nota-1
            '',                                               # nota-2
            ' - -------'                                      # nota-3
        ]
        cls.metadata_rows = [['A-0.txt', 'aauth', 'A author',
                              'mock genre 1', None, 'A', 1, False,
                              len(cls.dataset[0].split())],
                             ['A-1.txt', 'aauth', 'A author',
                              'mock genre 1', None, 'A', 1, False,
                              len(cls.dataset[1].split())],
                             ['A-2.txt', 'aauth', 'A author',
                              'mock genre 1', None, 'A', 1, False,
                              len(cls.dataset[2].split())],
                             ['A-3.txt', 'aauth', 'A author',
                              'mock genre 2', None, 'A', 1, False,
                              len(cls.dataset[3].split())],

                             ['U-0.txt', None, None, 'mock genre 3',
                              None, 'U', None, False,
                              len(cls.dataset[4].split())],
                             ['U-1.txt', None, None, 'mock genre 3',
                              None, 'U', None, False,
                              len(cls.dataset[5].split())],
                             ['U-2.txt', None, None, 'mock genre 3',
                              None, 'U', None, False,
                              len(cls.dataset[6].split())],
                             ['U-3.txt', None, None, 'mock genre 3',
                              None, 'U', None, False,
                              len(cls.dataset[7].split())],

                             ['notA-0.txt', 'naauth', 'Imposter author',
                              'mock genre 1', 'John', 'notA', 0, False,
                              len(cls.dataset[8].split())],
                             ['notA-1.txt', 'naauth', 'Imposter author',
                              'mock genre 2', 'John', 'notA', 0, False,
                              len(cls.dataset[9].split())],
                             ['notA-2.txt', 'naauth', 'Imposter author',
                              'mock genre 1', 'Jane', 'notA', 0, False,
                              len(cls.dataset[10].split())],
                             ['notA-3.txt', 'naauth', 'Imposter author',
                              'mock genre 2', 'Jane', 'notA', 0, False,
                              len(cls.dataset[11].split())]]

        # Set name of directory where all test data for this test run will
        # be placed.
        cls.test_data_directory = 'normalization_test_dir'

        # Generate the test data and relevent paths
        cls.paths, cls.canonical_class_labels = generate_test_data(
            cls.test_data_directory, cls.dataset, cls.metadata_rows)

    @classmethod
    def teardown_class(cls):
        """
        Clean up test data.
        """
        # Remove everything after tests have run
        destroy_test_data(cls.test_data_directory)

    def test_dataset_normalization(cls):
        """
        test the normalization pipeline, which should lowercase all text,
        and remove all non-alphanumeric characters. it should write the
        normalized results to files in a directory tree which mirrors its
        source tree.

        this test tests both the output text as well as the resulting
        directory structure.
        """

        # get path to expected output directory which should have been
        # created by the `normalize()` function.
        expected_output_dir = os.path.join(
            cls.paths['test_suite_dir'], 'normalized', 'undistorted')

        # hardcode the expected file contents output from the
        # `normalize()` function.
        expected_output = [
            'here is some text that should not be touched',      # a-0
            'here is some text that should be touched',          # a-1
            'here are s0me 1ntegers',                            # a-2
            'and spcal chracter',                                # a-3
            'here is some white space',                          # u-0
            'and back to untouched text',                        # u-1
            'white space capitalizations and special chars',     # u-2
            'white space and capitalizations',                   # u-3
            'white space and special chars',                  # nota-0
            'caps and special ch372ars',                      # nota-1
            '',                                               # nota-2
            ''                                                # nota-3
        ]

        # check that the `normalize()` routine created the expected root
        # directory.
        assert os.path.isdir(expected_output_dir)

        rows_idx = 0
        # loop through the sub directories in the expected root dir if it
        # exists.
        for i, canonical_class in enumerate(cls.canonical_class_labels):
            # check that the expected sub-dirs exist.
            assert os.path.isdir(
                os.path.join(expected_output_dir, canonical_class))
            num_files = sum([1 for i
                             in cls.metadata_rows
                             if i[5] == canonical_class])
            # loop through each file in the sub-dir
            for j, test_file in enumerate(range(num_files)):
                # hardcode the expected file name and path.
                file_name = f"{canonical_class}-{test_file}.txt"
                file_path = os.path.join(expected_output_dir,
                                         canonical_class,
                                         file_name)

                # check that the expected file exists with the proper name
                assert os.path.exists(file_path)

                # read in the file if it exists.
                with open(file_path, 'r') as f:
                    contents = f.read()

                    # check that the contents of the file output by the
                    # `normalize()` function match the expected contents.
                    assert (expected_output[rows_idx + j] ==
                            contents)
            rows_idx += num_files