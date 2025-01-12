"""
This test suite tests this project's implementation of text distortion
using Distorted View with Multiple Asterisks (DV-MA) and Distorted View
with Single Asterisk algorithms proposed in Stamatatos et al. (2017) [12]:

[12] Efstathios Stamatatos. 2017. Authorship Attribution Using Text
Distortion. In Proceedings of the 15th Conference of the European Chapter
of the Association for Computational Linguistics: Volume 1, Long Papers,
Association for Computational Linguistics, Valencia, Spain, 1138–1149.
Retrieved from https://aclanthology.org/E17-1107

It tests varying values of `k` (the number of most occuring vocabulary of
words to preserve and not distort) including edge cases.
"""
import os
# Import custom modules
from generate_test_data import generate_test_data, destroy_test_data  # noqa: E402


class TestPreprocessing:
    """
    A unified class to allow for easy setup and teardown of global and
    reused data and objects, and sharing of common methods. See
    `setup_class`, `teardown_class`, `compare_contents`.
    """
    @classmethod
    def setup_class(cls):
        """
        Setup global and reused test data and objects.

        The mock dataset should have the following word frequencies:
        - elit:        1
        - adipiscing:  2
        - consectetur: 3
        - amet:        4
        - s9t:         5
        - 42:          6
        - ipsum:       7
        - lorem:       8
        """
        cls.dataset = [
            'lorem ipsum',                                        # A-0
            'consectetur lorem',                                  # A-1
            '42 adipiscing ipsum 42 s9t amet',                    # A-2
            'adipiscing elit amet',                               # A-3
            'lorem 42 consectetur amet consectetur amet s9t',     # U-0
            'lorem',                                              # U-1
            '42 ipsum s9t',                                       # U-2
            'ipsum lorem',                                        # U-3
            '42 42 lorem s9t',                                 # notA-0
            'ipsum lorem s9t',                                 # notA-1
            'ipsum ipsum',                                     # notA-2
            'lorem'                                            # notA-3
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
        cls.test_data_directory = 'distortion_test_dir'

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

    def compare_contents(cls, expected_output, output_dir):
        """
        This is a generalized function for comparing the contents on a
        distorted view directory and the expected output to assert
        equality and existence of tree structure.

        :param expected_output: <Required> The expected text contents of
        all the files in a given view directory. See the docstring for
        `setup_class()` for the required structure.
        :type expected_output: list
        :param output_dir: <Required> The name of the directory where the
        view's subdirectories reside.
        :type output_dir: str
        """
        # Get the DV-MA-k-0 dir
        source_dir = os.path.join(cls.paths['test_suite_dir'],
                                  'normalized', output_dir)

        # Check that the `distort_text()` function created the expected
        # view directory.
        assert os.path.isdir(source_dir) is True

        canonical_class_labels = sorted(os.listdir(source_dir))
        rows_idx = 0

        # Loop through the sub directories in the expected root dir if it
        # exists.
        for i, canonical_class in enumerate(canonical_class_labels):
            # Check that the expected sub-dirs exist.
            assert os.path.isdir(os.path.join(source_dir,
                                              canonical_class)) is True
            num_files = sum([1 for i
                             in cls.metadata_rows
                             if i[5] == canonical_class])
            # Loop through each file in the sub-dir
            for j, test_file in enumerate(range(num_files)):
                # Hardcode the expected file name and path.
                file_name = f"{canonical_class}-{test_file}.txt"
                file_path = os.path.join(source_dir,
                                         canonical_class,
                                         file_name)

                # Check that the expected file exists with the proper
                # name.
                assert os.path.exists(file_path) is True

                # Read in the file if it exists.
                with open(file_path, 'r') as f:
                    contents = f.read()

                    # Check that the contents of the file output by the
                    # `distort_text()` function match the expected
                    # contents.
                    assert (
                        expected_output[rows_idx + j] ==
                        contents) is True
            rows_idx += num_files

    def test_dv_ma_k_0(cls):
        """
        This test tests this project's implementation of the Distorted
        View with Multiple Asterisks (DV-MA) algorithm.

        It tests the edge case of a `k` value of zero, which should result
        in all words being distorted to strings of asterisks or hash
        symbols based on the words' length and alphanumeric type.
        """

        # Hardcode the expected file contents output from the
        # `distort_text()` function.
        expected_output = [
            '***** *****',                                        # A-0
            '*********** *****',                                  # A-1
            '## ********** ***** ## *#* ****',                    # A-2
            '********** **** ****',                               # A-3
            '***** ## *********** **** *********** **** *#*',     # U-0
            '*****',                                              # U-1
            '## ***** *#*',                                       # U-2
            '***** *****',                                        # U-3
            '## ## ***** *#*',                                 # notA-0
            '***** ***** *#*',                                 # notA-1
            '***** *****',                                     # notA-2
            '*****'                                            # notA-3
        ]

        cls.compare_contents(expected_output, 'DV-MA-k-0')

    def test_dv_ma_k_2(cls):
        """
        This test tests this project's implementation of the Distorted
        View with Multiple Asterisks (DV-MA) algorithm.

        It tests the case of a `k` value of two, which should result in
        all but two vocabulary words being distorted to strings of
        asterisks or hash symbols based on the words' length and
        alphanumeric type.
        """

        # Hardcode the expected file contents output from the
        # `distort_text()` function.
        expected_output = [
            'lorem ipsum',                                        # A-0
            '*********** lorem',                                  # A-1
            '## ********** ipsum ## *#* ****',                    # A-2
            '********** **** ****',                               # A-3
            'lorem ## *********** **** *********** **** *#*',     # U-0
            'lorem',                                              # U-1
            '## ipsum *#*',                                       # U-2
            'ipsum lorem',                                        # U-3
            '## ## lorem *#*',                                 # notA-0
            'ipsum lorem *#*',                                 # notA-1
            'ipsum ipsum',                                     # notA-2
            'lorem'                                            # notA-3
        ]

        cls.compare_contents(expected_output, 'DV-MA-k-2')

    def test_dv_ma_k_all(cls):
        """
        This test tests this project's implementation of the Distorted
        View with Multiple Asterisks (DV-MA) algorithm.

        It tests the edge case of a `k` value equal to the corpus'
        vocabulary size, which should result in no words being distorted.
        """

        # Hardcode the expected file contents output from the
        # `distort_text()` function.
        expected_output = [
            'lorem ipsum',                                        # A-0
            'consectetur lorem',                                  # A-1
            '42 adipiscing ipsum 42 s9t amet',                    # A-2
            'adipiscing elit amet',                               # A-3
            'lorem 42 consectetur amet consectetur amet s9t',     # U-0
            'lorem',                                              # U-1
            '42 ipsum s9t',                                       # U-2
            'ipsum lorem',                                        # U-3
            '42 42 lorem s9t',                                 # notA-0
            'ipsum lorem s9t',                                 # notA-1
            'ipsum ipsum',                                     # notA-2
            'lorem'                                            # notA-3
        ]

        cls.compare_contents(expected_output, 'DV-MA-k-8')

    def test_dv_sa_k_0(cls):
        """
        This test tests this project's implementation of the Distorted
        View with Single Asterisk (DV-SA) algorithm.

        It tests the edge case of a `k` value of zero, which should result
        in all words being distorted to single asterisks or hash symbols
        based on the words' alphanumeric type.
        """

        # Hardcode the expected file contents output from the
        # `distort_text()` function.
        expected_output = [
            '* *',                                                # A-0
            '* *',                                                # A-1
            '# * * # *#* *',                                      # A-2
            '* * *',                                              # A-3
            '* # * * * * *#*',                                    # U-0
            '*',                                                  # U-1
            '# * *#*',                                            # U-2
            '* *',                                                # U-3
            '# # * *#*',                                       # notA-0
            '* * *#*',                                         # notA-1
            '* *',                                             # notA-2
            '*'                                                # notA-3
        ]

        cls.compare_contents(expected_output, 'DV-SA-k-0')

    def test_dv_sa_k_2(cls):
        """
        This test tests this project's implementation of the Distorted
        View with Single Asterisk (DV-SA) algorithm.

        It tests the case of a `k` value of two, which should result in
        all but two vocabulary words being distorted to single asterisks
        or hash symbols based on the words' alphanumeric type.
        """

        # Hardcode the expected file contents output from the
        # `distort_text()` function.
        expected_output = [
            'lorem ipsum',                                        # A-0
            '* lorem',                                            # A-1
            '# * ipsum # *#* *',                                  # A-2
            '* * *',                                              # A-3
            'lorem # * * * * *#*',                                # U-0
            'lorem',                                              # U-1
            '# ipsum *#*',                                        # U-2
            'ipsum lorem',                                        # U-3
            '# # lorem *#*',                                   # notA-0
            'ipsum lorem *#*',                                 # notA-1
            'ipsum ipsum',                                     # notA-2
            'lorem'                                            # notA-3
        ]

        cls.compare_contents(expected_output, 'DV-SA-k-2')

    def test_dv_sa_k_all(cls):
        """
        This test tests this project's implementation of the Distorted
        View with Single Asterisk (DV-SA) algorithm.

        It tests the edge case of a `k` value equal to the corpus'
        vocabulary size, which should result in no words being distorted.
        """

        # Hardcode the expected file contents output from the
        # `distort_text()` function.
        expected_output = [
            'lorem ipsum',                                        # A-0
            'consectetur lorem',                                  # A-1
            '42 adipiscing ipsum 42 s9t amet',                    # A-2
            'adipiscing elit amet',                               # A-3
            'lorem 42 consectetur amet consectetur amet s9t',     # U-0
            'lorem',                                              # U-1
            '42 ipsum s9t',                                       # U-2
            'ipsum lorem',                                        # U-3
            '42 42 lorem s9t',                                 # notA-0
            'ipsum lorem s9t',                                 # notA-1
            'ipsum ipsum',                                     # notA-2
            'lorem'                                            # notA-3
        ]

        cls.compare_contents(expected_output, 'DV-SA-k-8')