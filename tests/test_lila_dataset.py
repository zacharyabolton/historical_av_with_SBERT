"""
This test suite tests this project's implementaion of the abstract PyTorch
`Dataset` class.

https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
"""
import os
import sys
import shutil
import torch
import torch.nn.functional as F
from constants import ROOT_DIR, LEARNING_RATE

# Add src directory to sys.path
# Adapted from Taras Alenin's answer on StackOverflow at:
# https://stackoverflow.com/a/55623567
src_path = os.path.join(ROOT_DIR, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import custom modules
from lila_dataset import LILADataset  # noqa: E402


class TestLILADataset:
    """
    A unified class to allow for easy setup and teardown of global and reused
    data and objects, and sharing of common methods. See `setup_class`,
    `teardown_class`, `helper_test_chunking`.
    """
    @classmethod
    def setup_class(cls):
        """
        Setup global and reused test data and objects.

        Dataset characteristics:
            Word Frequencies:
            - elit:        1
            - adipiscing:  2
            - consectetur: 3
            - amet:        4
            - s9t:         5
            - 42:          6
            - ipsum:       7
            - lorem:       8
            Vocab size: 8
            Num words: 36
            Average doc length: 3
            Shortest doc: 1
            Longest doc: 7
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
        cls.pytorch_ds_test_dir = os.path.join(test_dir,
                                           'distortion_test_data')

        # Adapted from: https://stackoverflow.com/a/13118112
        shutil.rmtree(cls.pytorch_ds_test_dir, ignore_errors=True)
        os.mkdir(cls.pytorch_ds_test_dir)

        # Create a mock source directory
        cls.undistorted_dir = os.path.join(cls.pytorch_ds_test_dir,
                                           'undistorted')
        os.mkdir(cls.undistorted_dir)

        # Create mock sub-directories
        A_dir = os.path.join(cls.undistorted_dir, 'A')
        notA_dir = os.path.join(cls.undistorted_dir, 'notA')
        U_dir = os.path.join(cls.undistorted_dir, 'U')

        os.mkdir(A_dir)
        os.mkdir(notA_dir)
        os.mkdir(U_dir)

        # Write mock data to appropriate files
        num_files = 4

        canonical_class_labels = sorted(os.listdir(cls.undistorted_dir))

        for i, canonical_class in enumerate(canonical_class_labels):
            for j, test_file in enumerate(range(num_files)):
                file_name = f"{canonical_class}-{test_file}.txt"
                file_path = os.path.join(cls.undistorted_dir,
                                         canonical_class,
                                         file_name)
                with open(file_path, 'w') as f:
                    f.write(cls.dataset[i * num_files + j])

        # Insantiate PyTorch dataset object
        cls.chunk_size = 5
        cls.ds = LILADataset(cls.undistorted_dir, chunk_size=cls.chunk_size)

    @classmethod
    def teardown_class(cls):
        """
        Clean up test data.
        """
        # Remove everything after tests have run
        # shutil.rmtree(cls.pytorch_ds_test_dir, ignore_errors=True)

    def test_dataset_exists(cls):
        """
        Trivial test to ensure LILADataset can instantiate
        """
        assert isinstance(cls.ds, LILADataset)

    def test_dataset_subdirs_exist(cls):
        """
        Ensure the LILADataset points to the currect directories and that
        they exist.
        """
        assert os.path.exists(cls.ds._data_dir)
        assert os.path.exists(cls.ds._A_dir)
        assert os.path.exists(cls.ds._notA_dir)
        assert os.path.exists(cls.ds._U_dir)

    def test_docs_contents(cls):
        """
        Test that the contents of the LILADataset docs are the same as the
        mock data passed in.
        """
        assert cls.ds._A_docs == ['lorem ipsum',
                                  'consectetur lorem',
                                  '42 adipiscing ipsum 42 s9t amet',
                                  'adipiscing elit amet']
        assert cls.ds._U_docs == ['lorem 42 consectetur amet consectetur amet s9t',
                                  'lorem',
                                  '42 ipsum s9t',
                                  'ipsum lorem']
        assert cls.ds._notA_docs == ['42 42 lorem s9t',
                                     'ipsum lorem s9t',
                                     'ipsum ipsum',
                                     'lorem']

    def test_tokenization(cls):
        """
        Test that the LILADataset tokenizes the mock data correctly. While
        this is not an exhaustive test, it tests boundary conditions and for
        'sane' tokenized data characteristics.
        """
        assert (len(cls.ds._A_docs_tokenized) == 4) is True
        assert (len(cls.ds._U_docs_tokenized) == 4) is True
        assert (len(cls.ds._notA_docs_tokenized) == 4) is True

        At_ids = cls.ds._A_docs_tokenized[0].input_ids.tolist()[0]
        nAt_ids = cls.ds._notA_docs_tokenized[0].input_ids.tolist()[0]
        Ut_ids = cls.ds._U_docs_tokenized[0].input_ids.tolist()[0]
        # test for special BERT [CLS] and [SEP] tokens
        assert (At_ids[0] == 101) is True
        assert (Ut_ids[0] == 101) is True
        assert (nAt_ids[0] == 101) is True
        assert (At_ids[-1] == 102) is True
        assert (Ut_ids[-1] == 102) is True
        assert (nAt_ids[-1] == 102) is True
        # test for 'sane' middle values
        assert (At_ids[1] > 102) is True
        assert (Ut_ids[1] > 102) is True
        assert (nAt_ids[1] > 102) is True
        # ensure nothing else screwy happened
        assert (len(At_ids) >= 4) is True  # Num words in A_docs[0] + 2
        assert (len(Ut_ids) >= 9) is True  # Num words in U_docs[0] + 2
        assert (len(nAt_ids) >= 6) is True  # Num words in notA_docs[0] + 2
        assert (At_ids != nAt_ids) is True
        assert (At_ids != Ut_ids) is True
        assert (nAt_ids != Ut_ids) is True

    def helper_test_chunking(cls, docs_tokenized, docs_chunked):
        """
        A generalized routine for the `test_chunking` test.
        This routine does that actual work of testing the chunking mechanism
        in LILADataset. It is parameterized to allow testing for different
        class based collections of tokenized docs (A, U, notA).

        :param docs_tokenized: The collection of tokenized docs belonging to
        the LILADataset. (A, U, or notA)
        :type docs_tokenized: list
        :param docs_chunked: The actual collection of chunked and tokenized
        documents belonging to the LILADataset (A, U, notA)
        :type docs_chunked: list
        """
        for i, doc in enumerate(docs_tokenized):
            ts = doc.input_ids[0, 1:-1]
            length = ts.size()[0]
            chunk_len = cls.chunk_size - 2
            real_chunks = docs_chunked[i]
            # Create tensors for special tokens
            cls_token = torch.tensor([cls.ds.tokenizer.cls_token_id])
            sep_token = torch.tensor([cls.ds.tokenizer.sep_token_id])
            for chunk_start in range(0, length, chunk_len):
                expected_chunk_ids = ts[chunk_start:chunk_start + chunk_len]
                # Add CLS token at start
                expected_chunk_ids = torch.cat([cls_token,
                                                expected_chunk_ids,
                                                sep_token], dim=0)
                short = (chunk_len + 2) - expected_chunk_ids.size()[0]
                if short > 0:
                    # Adapted from:
                    # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
                    expected_chunk_ids = F.pad(expected_chunk_ids,
                                   (0, short),
                                   "constant", 0)  # effectively zero padding
                expected_chunk_ids = expected_chunk_ids.unsqueeze(dim=0)
                real_chunk_index = int(chunk_start/chunk_len)
                real_chunk_ids = real_chunks[real_chunk_index]['input_ids']
                # Adapted from:
                # https://stackoverflow.com/a/54187453
                assert torch.equal(real_chunk_ids, expected_chunk_ids), True

    def test_chunking(cls):
        """
        This is a test of the chunking mechanism in the LILADataset. It calls
        `helper_test_chunking()` method which does that actual end-testing,
        as well as trivially checks for correct lengths of the collections.
        """
        cls.helper_test_chunking(cls.ds._A_docs_tokenized,
                                 cls.ds._A_docs_chunked)
        cls.helper_test_chunking(cls.ds._U_docs_tokenized,
                                 cls.ds._U_docs_chunked)
        cls.helper_test_chunking(cls.ds._notA_docs_tokenized,
                                 cls.ds._notA_docs_chunked)

        assert (len(cls.ds._A_docs_chunked) == 4) is True
        assert (len(cls.ds._U_docs_chunked) == 4) is True
        assert (len(cls.ds._notA_docs_chunked) == 4) is True

    def test_pair_creation(cls):
        """
        This is a test of the pair creation method in LILADataset.

        This method should create all possible pairs for `A` chunks, minus
        self pairs and pairs that only differ in order. The calculation for
        how pairs same-author pairs should result from this is
        `N_A choose 2`, where `N_A` is the number of A chunks:

        N_A(N_A - 1)/2

        It should also create all possible pairs of `A` and `notA`, ignoring
        differently ordered pairs:

        N_A * N_notA

        Therefore the total number of pairs should be:

        N_A(N_A - 1)/2 + N_A * N_notA

        For our test datast this should be:

        12(12 - 1)/2 = 66
        12 * 9 = 108
        66 + 108 = 174

        Pairs should be tuples, where the first and second element are chunks
        in the form of PyTorch embeddings, and the third element is a class
        lable: 1 for same-athor, 0 for different-author.
        """

        # Check that the correct number of pairs were generated
        assert (len(cls.ds._pairs) == 174), True

        # Save number of same author pairs for splitting on class type
        num_sa_pairs = 66

        # Check that the first block of pairs are all same-author
        for pair in cls.ds._pairs[:num_sa_pairs]:
            assert pair[2] == 1
        # Check that the last block of pairs are all different-author
        for pair in cls.ds._pairs[num_sa_pairs:]:
            assert pair[2] == 0

        # Check that the first and second tuple elements are of the right
        # shape for PyTorch model ingestion
        assert (type(cls.ds._pairs[0][0]) == dict), True
        assert 'input_ids' in cls.ds._pairs[0][0]
        assert 'attention_mask' in cls.ds._pairs[0][0]