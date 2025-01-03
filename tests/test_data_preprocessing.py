import os
import sys
import shutil
from definitions import ROOT_DIR

# Add scripts directory to sys.path
# Adapted from Taras Alenin's answer on StackOverflow at:
# https://stackoverflow.com/a/55623567
scripts_path = os.path.join(ROOT_DIR, 'scripts')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

# Import custom modules
from text_normalizer import normalize  # noqa: E402
from text_distorter import distort_text  # noqa: E402



class TestDVAlgos:
    """
    This test tests this project's implementation of the Distorted View with
    Multiple Asterisks (DV-MA) and Distorted View with Single Asterisk
    algorithms proposed in Stamatatos et al. (2017):

    Efstathios Stamatatos. 2017. Authorship Attribution Using Text
    Distortion. In Proceedings of the 15th Conference of the European Chapter
    of the Association for Computational Linguistics: Volume 1, Long Papers,
    Association for Computational Linguistics, Valencia, Spain, 1138–1149.
    Retrieved from https://aclanthology.org/E17-1107

    It tests varying values of `k` (the number of most occuring vocabulary of
    words to preserve and not distort) including edge cases.
    """
    @classmethod
    def setup_class(cls):
        """
        The follwing dataset should have the following word frequencies:
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
            'lorem ipsum',                                        # A-0.txt
            'consectetur lorem',                                  # A-1.txt
            '42 adipiscing ipsum 42 s9t amet',                    # A-2.txt
            'adipiscing elit amet',                               # A-3.txt
            'lorem 42 consectetur amet consectetur amet s9t',     # U-0.txt
            'lorem',                                              # U-1.txt
            '42 ipsum s9t',                                       # U-2.txt
            'ipsum lorem',                                        # U-3.txt
            '42 42 lorem s9t',                                 # notA-0.txt
            'ipsum lorem s9t',                                 # notA-1.txt
            'ipsum ipsum',                                     # notA-2.txt
            'lorem'                                            # notA-3.txt
        ]

        test_dir = '../data/test'
        cls.distortion_test_dir = os.path.join(test_dir,
                                               'distortion_test_data')
        cls.normalizing_test_dir = os.path.join(test_dir,
                                                'normalizing_test_data')

        # Adapted from: https://stackoverflow.com/a/13118112
        shutil.rmtree(cls.distortion_test_dir, ignore_errors=True)
        os.mkdir(cls.distortion_test_dir)
        shutil.rmtree(cls.normalizing_test_dir, ignore_errors=True)
        os.mkdir(cls.normalizing_test_dir)


        # Create a mock source directory
        normalized_dir = os.path.join(cls.distortion_test_dir,
                                      'undistorted')
        os.mkdir(normalized_dir)

        # Create mock sub-directories
        A_dir = os.path.join(normalized_dir, 'A')
        notA_dir = os.path.join(normalized_dir, 'notA')
        U_dir = os.path.join(normalized_dir, 'U')

        os.mkdir(A_dir)
        os.mkdir(notA_dir)
        os.mkdir(U_dir)

        # Write mock data to appropriate files
        cls.num_files = 4

        cls.canonical_class_labels = sorted(os.listdir(normalized_dir))

        for i, canonical_class in enumerate(cls.canonical_class_labels):
            for j, test_file in enumerate(range(cls.num_files)):
                file_name = f"{canonical_class}-{test_file}.txt"
                file_path = os.path.join(normalized_dir,
                                         canonical_class,
                                         file_name)
                with open(file_path, 'w') as f:
                    f.write(cls.dataset[i * cls.num_files + j])

        # Run the text_distorter script with `k` values of 0, 2, and 8 (the
        # length of the vocabulary)
        ks = [0, 2, 8]
        distort_text(cls.distortion_test_dir, cls.canonical_class_labels, ks)

    @classmethod
    def teardown_class(cls):
        """
        Runs once after all tests in this class have completed.
        Optional, if you want to remove the test data at the end or clean up.
        """
        # Remove everything after tests have run
        shutil.rmtree(cls.distortion_test_dir, ignore_errors=True)
        shutil.rmtree(cls.normalizing_test_dir, ignore_errors=True)
        pass

    def compare_contents(self, expected_output, output_dir):
        """
        This is a generalized function for comparing the contents on a
        distorted view directory and the expected output to assert equality
        and existence of tree structure.

        :param expected_output: The expected text contents of all the files
        in a given view directory. See the docstring for `setup_class()` for
        the required structure.
        :type expected_output: list
        :param output_dir: The name of the directory where the view's
        subdirectories reside.
        :type output_dir: str
        """
        # Get the DV-MA-k-0 dir
        source_dir = os.path.join(self.__class__.distortion_test_dir,
                                  output_dir)

        # Check that the `distort_text()` function created the expected view
        # directory.
        assert os.path.isdir(source_dir)

        canonical_class_labels = sorted(os.listdir(source_dir))

        # Loop through the sub directories in the expected root dir if it
        # exists.
        for i, canonical_class in enumerate(canonical_class_labels):
            # Check that the expected sub-dirs exist.
            assert os.path.isdir(os.path.join(source_dir,
                                              canonical_class)), True

            # Loop through each file in the sub-dir
            for j, test_file in enumerate(range(self.__class__.num_files)):
                # Hardcode the expected file name and path.
                file_name = f"{canonical_class}-{test_file}.txt"
                file_path = os.path.join(source_dir,
                                         canonical_class,
                                         file_name)

                # Check that the expected file exists with the proper name.
                print(file_path)
                print(os.path.exists(file_path))
                assert os.path.exists(file_path)

                # Read in the file if it exists.
                with open(file_path, 'r') as f:
                    contents = f.read()

                    # Check that the contents of the file output by the
                    # `distort_text()` function match the expected contents.
                    assert (
                        expected_output[i * self.__class__.num_files + j] ==
                        contents), True

    def test_dv_ma_k_0(self):
        """
        This test tests this project's implementation of the Distorted View
        with Multiple Asterisks (DV-MA) algorithm.

        It tests the edge case of a `k` value of zero, which should result in
        all words being distorted to strings of asterisks or hash symbols
        based on the words' length and alphanumeric type.
        """

        # Hardcode the expected file contents output from the
        # `distort_text()` function.
        expected_output = [
            '***** *****',                                        # A-0.txt
            '*********** *****',                                  # A-1.txt
            '## ********** ***** ## *#* ****',                    # A-2.txt
            '********** **** ****',                               # A-3.txt
            '***** ## *********** **** *********** **** *#*',     # U-0.txt
            '*****',                                              # U-1.txt
            '## ***** *#*',                                       # U-2.txt
            '***** *****',                                        # U-3.txt
            '## ## ***** *#*',                                 # notA-0.txt
            '***** ***** *#*',                                 # notA-1.txt
            '***** *****',                                     # notA-2.txt
            '*****'                                            # notA-3.txt
        ]

        self.compare_contents(expected_output, 'DV-MA-k-0')


    def test_dv_ma_k_2(self):
        """
        This test tests this project's implementation of the Distorted View
        with Multiple Asterisks (DV-MA) algorithm.

        It tests the case of a `k` value of two, which should result in all
        but two vocabulary words being distorted to strings of asterisks or
        hash symbols based on the words' length and alphanumeric type.
        """

        # Hardcode the expected file contents output from the
        # `distort_text()` function.
        expected_output = [
            'lorem ipsum',                                        # A-0.txt
            '*********** lorem',                                  # A-1.txt
            '## ********** ipsum ## *#* ****',                    # A-2.txt
            '********** **** ****',                               # A-3.txt
            'lorem ## *********** **** *********** **** *#*',     # U-0.txt
            'lorem',                                              # U-1.txt
            '## ipsum *#*',                                       # U-2.txt
            'ipsum lorem',                                        # U-3.txt
            '## ## lorem *#*',                                 # notA-0.txt
            'ipsum lorem *#*',                                 # notA-1.txt
            'ipsum ipsum',                                     # notA-2.txt
            'lorem'                                            # notA-3.txt
        ]

        self.compare_contents(expected_output, 'DV-MA-k-2')

    def test_dv_ma_k_all(self):
        """
        This test tests this project's implementation of the Distorted View
        with Multiple Asterisks (DV-MA) algorithm.

        It tests the edge case of a `k` value equal to the corpus' vocabulary
        size, which should result in no words being distorted.
        """

        # Hardcode the expected file contents output from the
        # `distort_text()` function.
        expected_output = [
            'lorem ipsum',                                        # A-0.txt
            'consectetur lorem',                                  # A-1.txt
            '42 adipiscing ipsum 42 s9t amet',                    # A-2.txt
            'adipiscing elit amet',                               # A-3.txt
            'lorem 42 consectetur amet consectetur amet s9t',     # U-0.txt
            'lorem',                                              # U-1.txt
            '42 ipsum s9t',                                       # U-2.txt
            'ipsum lorem',                                        # U-3.txt
            '42 42 lorem s9t',                                 # notA-0.txt
            'ipsum lorem s9t',                                 # notA-1.txt
            'ipsum ipsum',                                     # notA-2.txt
            'lorem'                                            # notA-3.txt
        ]

        self.compare_contents(expected_output, 'DV-MA-k-8')

    def test_dv_sa_k_0(self):
        """
        This test tests this project's implementation of the Distorted View
        with Single Asterisk (DV-SA) algorithm.

        It tests the edge case of a `k` value of zero, which should result in
        all words being distorted to single asterisks or hash symbols based
        on the words' alphanumeric type.
        """

        # Hardcode the expected file contents output from the
        # `distort_text()` function.
        expected_output = [
            '* *',                                                # A-0.txt
            '* *',                                                # A-1.txt
            '# * * # *#* *',                                      # A-2.txt
            '* * *',                                              # A-3.txt
            '* # * * * * *#*',                                    # U-0.txt
            '*',                                                  # U-1.txt
            '# * *#*',                                            # U-2.txt
            '* *',                                                # U-3.txt
            '# # * *#*',                                       # notA-0.txt
            '* * *#*',                                         # notA-1.txt
            '* *',                                             # notA-2.txt
            '*'                                                # notA-3.txt
        ]

        self.compare_contents(expected_output, 'DV-SA-k-0')

    def test_dv_sa_k_2(self):
        """
        This test tests this project's implementation of the Distorted View
        with Single Asterisk (DV-SA) algorithm.

        It tests the case of a `k` value of two, which should result in all
        but two vocabulary words being distorted to single asterisks or hash
        symbols based on the words' alphanumeric type.
        """

        # Hardcode the expected file contents output from the
        # `distort_text()` function.
        expected_output = [
            'lorem ipsum',                                        # A-0.txt
            '* lorem',                                            # A-1.txt
            '# * ipsum # *#* *',                                # A-2.txt
            '* * *',                                              # A-3.txt
            'lorem # * * * * *#*',                               # U-0.txt
            'lorem',                                              # U-1.txt
            '# ipsum *#*',                                       # U-2.txt
            'ipsum lorem',                                        # U-3.txt
            '# # lorem *#*',                                 # notA-0.txt
            'ipsum lorem *#*',                                 # notA-1.txt
            'ipsum ipsum',                                     # notA-2.txt
            'lorem'                                            # notA-3.txt
        ]

        self.compare_contents(expected_output, 'DV-SA-k-2')


    def test_dv_sa_k_all(self):
        """
        This test tests this project's implementation of the Distorted View
        with Single Asterisk (DV-SA) algorithm.

        It tests the edge case of a `k` value equal to the corpus' vocabulary
        size, which should result in no words being distorted.
        """

        # Hardcode the expected file contents output from the
        # `distort_text()` function.
        expected_output = [
            'lorem ipsum',                                        # A-0.txt
            'consectetur lorem',                                  # A-1.txt
            '42 adipiscing ipsum 42 s9t amet',                    # A-2.txt
            'adipiscing elit amet',                               # A-3.txt
            'lorem 42 consectetur amet consectetur amet s9t',     # U-0.txt
            'lorem',                                              # U-1.txt
            '42 ipsum s9t',                                       # U-2.txt
            'ipsum lorem',                                        # U-3.txt
            '42 42 lorem s9t',                                 # notA-0.txt
            'ipsum lorem s9t',                                 # notA-1.txt
            'ipsum ipsum',                                     # notA-2.txt
            'lorem'                                            # notA-3.txt
        ]

        self.compare_contents(expected_output, 'DV-SA-k-8')

    def test_dataset_normalization(self):
        """
        Test the normalization pipeline, which should lowercase all text, and
        remove all non-alphanumeric characters. It should write the
        normalized results to files in a directory tree which mirrors its
        source tree.

        This test tests both the output text as well as the resulting
        directory structure.
        """

        # Create a mock source directory
        cleaned_dir = os.path.join(self.__class__.normalizing_test_dir,
                                   'cleaned')
        os.mkdir(cleaned_dir)

        # Create mock sub-directories
        A_dir = os.path.join(cleaned_dir, 'A')
        notA_dir = os.path.join(cleaned_dir, 'notA')
        U_dir = os.path.join(cleaned_dir, 'U')

        os.mkdir(A_dir)
        os.mkdir(notA_dir)
        os.mkdir(U_dir)

        # Create mock data
        test_contents = [
            'here is some text that should not be touched',      # A-0.txt
            'Here is some text that should be touched',          # A-1.txt
            'here are s0me 1ntegers',                            # A-2.txt
            'and sp*c|al cháracter$',                            # A-3.txt
            """here
            is  some     white space""",                         # U-0.txt
            'and back to untouched text',                        # U-1.txt
            """
            White space
            CAPITALIZATIONS
            **and special chars**!@#$%^&*()_-+=\\|~`
            """,                                                 # U-2.txt
            'white space      and CAPITALIZATIONS',              # U-3.txt
            'white space and special chars %#^&!@*!',         # notA-0.txt
            'CAPS and spe^#@&@*!*(!cial ch372ars ',           # notA-1.txt
            '',                                               # notA-2.txt
            ' - -------'                                      # notA-3.txt
        ]

        # Write mock data to appropriate files
        num_files = 4
        canonical_class_labels = sorted(os.listdir(cleaned_dir))
        for i, canonical_class in enumerate(canonical_class_labels):
            for j, test_file in enumerate(range(num_files)):
                file_name = f"{canonical_class}-{test_file}.txt"
                file_path = os.path.join(cleaned_dir,
                                         canonical_class,
                                         file_name)
                with open(file_path, 'w') as f:
                    f.write(test_contents[i * num_files + j])

        # Run the normalizing routine
        normalize(self.__class__.normalizing_test_dir,
                   canonical_class_labels)

        # Get path to expected output directory which should have been
        # created by the `normalize()` function.
        expected_output_dir = os.path.join(
            self.__class__.normalizing_test_dir, 'normalized')

        # Hardcode the expected file contents output from the `normalize()`
        # function.
        expected_output = [
            'here is some text that should not be touched',      # A-0.txt
            'here is some text that should be touched',          # A-1.txt
            'here are s0me 1ntegers',                            # A-2.txt
            'and spcal chracter',                                # A-3.txt
            'here is some white space',                          # U-0.txt
            'and back to untouched text',                        # U-1.txt
            'white space capitalizations and special chars',     # U-2.txt
            'white space and capitalizations',                   # U-3.txt
            'white space and special chars',                  # notA-0.txt
            'caps and special ch372ars',                      # notA-1.txt
            '',                                               # notA-2.txt
            ''                                                # notA-3.txt
        ]

        # Check that the `normalize()` routine created the expected root
        # directory.
        assert os.path.isdir(expected_output_dir), True

        # Loop through the sub directories in the expected root dir if it
        # exists.
        for i, canonical_class in enumerate(canonical_class_labels):
            # Check that the expected sub-dirs exist.
            assert os.path.isdir(
                os.path.join(expected_output_dir, canonical_class)), True

            # Loop through each file in the sub-dir
            for j, test_file in enumerate(range(num_files)):
                # Hardcode the expected file name and path.
                file_name = f"{canonical_class}-{test_file}.txt"
                file_path = os.path.join(expected_output_dir,
                                         canonical_class,
                                         file_name)

                # Check that the expected file exists with the proper name.
                assert os.path.exists(file_path), True

                # Read in the file if it exists.
                with open(file_path, 'r') as f:
                    contents = f.read()

                    # Check that the contents of the file output by the
                    # `normalize()` function match the expected contents.
                    assert (expected_output[i * num_files + j] ==
                            contents), True