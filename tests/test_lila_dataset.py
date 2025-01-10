"""
This test suite tests this project's implementaion of the abstract PyTorch
`Dataset` class.

https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
"""
import math
import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
from constants import ROOT_DIR

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
    A unified class to allow for easy setup and teardown of global and
    reused data and objects, and sharing of common methods. See
    `setup_class`, `teardown_class`, `helper_test_cnking`.
    """
    @classmethod
    def setup_class(cls):
        """
        Setup global and reused test data and objects.

        """
        # Create a mock dataset that can test some edge cases
        # Dataset characteristics (minus omitted work `A-4.txt`):
        #     Word Frequencies:
        #     - elit:        1
        #     - adipiscing:  2
        #     - consectetur: 3
        #     - amet:        4
        #     - s9t:         5
        #     - 42:          6
        #     - ipsum:       7
        #     - lorem:       8
        #     - foo:         24
        #     Vocab size: 9
        #     Num words: 60
        #     Average doc length: 5
        #     Shortest doc: 1
        #     Longest doc: 15
        cls.dataset = [
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
                 None, 'A', 1, False, len(cls.dataset[0].split())],
                ['A-1.txt', 'aauth', 'A author', 'mock genre 1',
                 None, 'A', 1, False, len(cls.dataset[1].split())],
                ['A-2.txt', 'aauth', 'A author', 'mock genre 1',
                 None, 'A', 1, False, len(cls.dataset[2].split())],
                ['A-3.txt', 'aauth', 'A author', 'mock genre 2',
                 None, 'A', 1, False, len(cls.dataset[3].split())],
                ['A-4.txt', 'aauth', 'A author', 'mock genre 2',
                 None, 'A', 1, True, len(cls.dataset[4].split())],

                ['U-0.txt', None, None, 'mock genre 3',
                 None, 'U', None, False, len(cls.dataset[5].split())],
                ['U-1.txt', None, None, 'mock genre 3',
                 None, 'U', None, False, len(cls.dataset[6].split())],
                ['U-2.txt', None, None, 'mock genre 3',
                 None, 'U', None, False, len(cls.dataset[7].split())],
                ['U-3.txt', None, None, 'mock genre 3',
                 None, 'U', None, False, len(cls.dataset[8].split())],

                ['notA-0.txt', 'naauth', 'Imposter author',
                 'mock genre 1', 'John', 'notA', 0, False,
                 len(cls.dataset[9].split())],
                ['notA-1.txt', 'naauth', 'Imposter author',
                 'mock genre 2', 'John', 'notA', 0, False,
                 len(cls.dataset[10].split())],
                ['notA-2.txt', 'naauth', 'Imposter author',
                 'mock genre 1', 'Jane', 'notA', 0, False,
                 len(cls.dataset[11].split())],
                ['notA-3.txt', 'naauth', 'Imposter author',
                 'mock genre 2', 'Jane', 'notA', 0, False,
                 len(cls.dataset[12].split())]]
        for row in rows:
            df_TEST_metadata.loc[len(df_TEST_metadata)] = row

        # Create test data tree and save dataset and metadata table within
        data_dir = '../data'
        test_dir = os.path.join(data_dir, 'test')
        cls.pytorch_ds_test_dir = os.path.join(test_dir,
                                               'distortion_test_data')
        # Delete the whole tree just in case the `teardown_class` was
        # toggled off during testing
        # Adapted from: https://stackoverflow.com/a/13118112
        shutil.rmtree(cls.pytorch_ds_test_dir, ignore_errors=True)
        os.mkdir(cls.pytorch_ds_test_dir)
        # Create a mock source directory
        cls.undistorted_dir = os.path.join(cls.pytorch_ds_test_dir,
                                           'undistorted')
        os.mkdir(cls.undistorted_dir)
        # Create the file path to the metadata
        cls.test_metadata_path = os.path.join(cls.pytorch_ds_test_dir,
                                              'metadata.csv')
        # Save the metadata
        df_TEST_metadata.to_csv(cls.test_metadata_path, index=False)
        # Create mock sub-directories
        A_dir = os.path.join(cls.undistorted_dir, 'A')
        notA_dir = os.path.join(cls.undistorted_dir, 'notA')
        U_dir = os.path.join(cls.undistorted_dir, 'U')
        os.mkdir(A_dir)
        os.mkdir(notA_dir)
        os.mkdir(U_dir)

        # Write mock data to appropriate files
        canonical_class_labels = sorted(os.listdir(cls.undistorted_dir))
        rows_idx = 0
        for i, canonical_class in enumerate(canonical_class_labels):
            num_files = sum([1 for i
                             in rows
                             if i[5] == canonical_class])
            for j, test_file in enumerate(range(num_files)):
                if rows[rows_idx + j][7] is not True:
                    file_name = f"{canonical_class}-{test_file}.txt"
                    file_path = os.path.join(cls.undistorted_dir,
                                             canonical_class,
                                             file_name)
                    with open(file_path, 'w') as f:
                        f.write(cls.dataset[rows_idx + j])
            rows_idx += num_files

        # Send all Tensor operations to mps if available, and cpu if not
        # TODO: Get this to work on GPU systems as well (CUDA)
        cls.device = "mps" if torch.backends.mps.is_available() else "cpu"
        if cls.device == 'mps':
            torch.mps.empty_cache()

        # Insantiate PyTorch dataset object with mock data
        # and toy parameters for testing
        cls.cnk_size = 5
        cls.num_pairs = 10
        cls.seed = 1
        cls.ds = LILADataset(cls.undistorted_dir,
                             cls.test_metadata_path,
                             cnk_size=cls.cnk_size,
                             num_pairs=cls.num_pairs,
                             device=cls.device,
                             seed=cls.seed)

    @classmethod
    def teardown_class(cls):
        """
        Clean up test data.
        """

        # Remove everything after tests have run
        shutil.rmtree(cls.pytorch_ds_test_dir, ignore_errors=True)

    def test_dataset_exists(cls):
        """
        Trivial test to ensure LILADataset can instantiate
        """

        assert isinstance(cls.ds, LILADataset)

    def test_metadata_exists(cls):
        """
        Trivial test to ensure the metadata file was created
        """
        assert os.path.exists(cls.test_metadata_path)

    def test_dataset_subdirs_exist(cls):
        """
        Ensure the LILADataset points to the currect directories and that
        they exist.
        """
        assert os.path.exists(cls.ds._data_dir)
        assert os.path.exists(cls.ds._A_dir)
        assert os.path.exists(cls.ds._notA_dir)
        assert os.path.exists(cls.ds._U_dir)
        assert os.path.exists(cls.ds._metadata_path)

    def test_docs_contents(cls):
        """
        Test that the contents of the LILADataset docs are the same as the
        mock data passed in.
        """
        assert cls.ds._A_docs == [(0, 'lorem ipsum'),
                                  (1, 'consectetur lorem'),
                                  (2, '42 adipiscing ipsum 42 s9t amet'),
                                  (3, 'adipiscing elit amet')]
        assert cls.ds._U_docs == [(4, 'lorem 42 consectetur amet'
                                   ' consectetur amet s9t'),
                                  (5, 'lorem'),
                                  (6, '42 ipsum s9t'),
                                  (7, 'ipsum lorem')]
        assert cls.ds._notA_docs == [(8, '42 42 lorem s9t foo foo foo foo'
                                      ' foo foo foo foo'),
                                     (9, 'ipsum lorem s9t foo'),
                                     (10, 'ipsum ipsum foo foo foo foo'
                                      ' foo foo foo foo foo foo foo foo'
                                      ' foo'),
                                     (11, 'lorem foo foo')]

    def test_docs_vs_metadata(cls):
        """
        Test whether the docs lists belonging LILADataset agree with its
        metadata.
        """

        # A docs
        for doc in cls.ds._A_docs:
            idx = doc[0]
            meta_data = cls.ds._metadata.loc[idx]

            # Test file field
            assert meta_data['file'] == f'A-{idx % 4}.txt'
            # Test author field
            assert meta_data['author'] == 'A author'
            # Test genre field
            if idx < 3:
                assert meta_data['genre'] == 'mock genre 1'
            else:
                assert meta_data['genre'] == 'mock genre 2'
            # Test imposter_for field
            assert isinstance(meta_data['imposter_for'], type(np.nan))
            # Test canonical_class_label field
            assert meta_data['canonical_class_label'] == 'A'
            # Test class field
            assert meta_data['class'] == 1
            # Test num_words field
            assert meta_data['num_words'] ==\
                len(cls.ds._A_docs[idx % 4][1].split())

        # U docs
        for doc in cls.ds._U_docs:
            idx = doc[0]
            meta_data = cls.ds._metadata.loc[idx]

            # Test file field
            assert meta_data['file'] == f'U-{idx % 4}.txt'
            # Test author field
            assert isinstance(meta_data['author'],
                              type(np.nan))
            # Test genre field
            assert meta_data['genre'] == 'mock genre 3'
            # Test imposter_for field
            assert isinstance(meta_data['imposter_for'], type(np.nan))
            # Test canonical_class_label field
            assert meta_data['canonical_class_label'] == 'U'
            # Test class field
            assert isinstance(meta_data['class'], type(np.nan))
            # Test num_words field
            assert meta_data['num_words'] ==\
                len(cls.ds._U_docs[idx % 4][1].split())

        # notA docs
        for doc in cls.ds._notA_docs:
            idx = doc[0]
            meta_data = cls.ds._metadata.loc[idx]

            # Test file field
            assert meta_data['file'] == f'notA-{idx % 4}.txt'
            # Test author field
            assert meta_data['author'] == 'Imposter author'
            # Test genre field
            if (idx % 2) == 0:
                assert meta_data['genre'] == 'mock genre 1'
            else:
                assert meta_data['genre'] == 'mock genre 2'
            # Test imposter_for field
            if (idx % 4) < 2:
                assert meta_data['imposter_for'] == 'John'
            else:
                assert meta_data['imposter_for'] == 'Jane'
            # Test canonical_class_label field
            assert meta_data['canonical_class_label'] == 'notA'
            # Test class field
            assert meta_data['class'] == 0
            # Test num_words field
            assert meta_data['num_words'] ==\
                len(cls.ds._notA_docs[idx % 4][1].split())

    def test_tokenization(cls):
        """
        Test that the LILADataset tokenizes the mock data correctly. While
        this is not an exhaustive test, it tests boundary conditions and
        for 'sane' tokenized data characteristics.
        """

        # Check that the root lists length is the same as the original
        # number of documents processed for each class
        assert len(cls.ds._A_docs_tokenized) == 4
        assert len(cls.ds._U_docs_tokenized) == 4
        assert len(cls.ds._notA_docs_tokenized) == 4

        # Get the input_ids from the PyTorch embedding for the first
        # document in each class
        At_ids = cls.ds._A_docs_tokenized[0][1].input_ids.tolist()[0]
        nAt_ids = cls.ds._notA_docs_tokenized[0][1].input_ids.tolist()[0]
        Ut_ids = cls.ds._U_docs_tokenized[0][1].input_ids.tolist()[0]

        # test for special BERT [CLS] and [SEP] tokens
        assert At_ids[0] == 101
        assert Ut_ids[0] == 101
        assert nAt_ids[0] == 101
        assert At_ids[-1] == 102
        assert Ut_ids[-1] == 102
        assert nAt_ids[-1] == 102
        # test or 'sane' middle values
        assert At_ids[1] > 102
        assert Ut_ids[1] > 102
        assert nAt_ids[1] > 102
        # ensur nothing else screwy happened
        assert len(At_ids) >= 4
        assert len(Ut_ids) >= 9
        assert len(nAt_ids) >= 6
        assert At_ids != nAt_ids
        assert At_ids != Ut_ids
        assert nAt_ids != Ut_ids

    def helper_test_cnking(cls, docs_tokenized, docs_cnked):
        """
        A generalized routine for the `test_cnking` test.
        This routine does that actual work of testing the chunking
        mechanism in LILADataset. It is parameterized to allow testing for
        different class based collections of tokenized docs (A, U, notA).

        :param docs_tokenized: <Required> List of tuples representing
        belonging to the A, U, or notA canonical classes in the LILA
        dataset, where the first tuple element is the associated index for
        the given file in the `LILADataset._metadata` member, and the
        second element is a PyTorch embedding of the tokenized contents.
        :type docs_tokenized: list

        :param docs_cnked: <Required> The actual collection of chunked
        and tokenized documents belonging to the LILADataset (A, U, notA).
        A list of tuples, where the first tuple element is the associated
        index for a given file in the `LILADataset._metadata` member, and
        the second element is a list of PyTorch embeddings of the chunks
        generated from that file.
        :type A_docs_cnked: list
        """
        for i, (doc_idx, doc) in enumerate(docs_tokenized):
            ts = doc.input_ids[0, 1:-1]
            length = ts.size()[0]
            cnk_len = cls.cnk_size - 2
            real_cnks = docs_cnked[i][1]
            # Create tensors for special tokens
            cls_token = torch.tensor([cls.ds.tokenizer.cls_token_id])
            sep_token = torch.tensor([cls.ds.tokenizer.sep_token_id])
            for cnk_start in range(0, length, cnk_len):
                expected_cnk_ids = ts[cnk_start:cnk_start + cnk_len]
                # Add CLS token at start
                expected_cnk_ids = torch.cat([cls_token,
                                              expected_cnk_ids,
                                              sep_token],
                                             dim=0).to(cls.device)
                short = (cnk_len + 2) - expected_cnk_ids.size()[0]
                if short > 0:
                    # # Adapted from:
                    # # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html  # noqa: E501
                    # expected_cnk_ids = F.pad(expected_cnk_ids,
                    #                          (0, short),
                    #                          "constant",
                    #                          0)  # effectively 0 padding
                    # Throw away for now
                    continue
                expected_cnk_ids = expected_cnk_ids.unsqueeze(dim=0)
                real_cnk_index = int(cnk_start/cnk_len)
                real_cnk_ids = real_cnks[real_cnk_index]['input_ids']
                # Adapted from:
                # https://stackoverflow.com/a/54187453
                assert torch.equal(real_cnk_ids, expected_cnk_ids)

    def test_cnking(cls):
        """
        This is a test of the chunking mechanism in the LILADataset. It
        calls `helper_test_cnking()` method which does that actual
        end-testing.
        It also trivially checks for correct lengths of the collections.
        """
        cls.helper_test_cnking(cls.ds._A_docs_tokenized,
                               cls.ds._A_docs_cnked)
        cls.helper_test_cnking(cls.ds._U_docs_tokenized,
                               cls.ds._U_docs_cnked)
        cls.helper_test_cnking(cls.ds._notA_docs_tokenized,
                               cls.ds._notA_docs_cnked)

        assert len(cls.ds._A_docs_cnked) == 4
        assert len(cls.ds._U_docs_cnked) == 4
        assert len(cls.ds._notA_docs_cnked) == 4

    def test_pair_creation(cls):
        """
        Test that the LILADataset creates pairs correctly. While this is
        not an exhaustive test, it tests boundary conditions and for
        'sane' pair creations characteristics.
        """

        # Check that the correct number of pairs were generated
        assert len(cls.ds._pairs) == cls.num_pairs

        # Calculate the balanced numbr of same/diff pairs to generate
        # When unable to balance perfectly we give the difference to the
        # different-author pairs. This code matches the internals of
        # LILADataset but checks for correct processing downstream
        metadata = cls.ds._metadata
        class_mask = metadata['class'] == 0
        n_imp_types = len(metadata[class_mask]['imposter_for'].unique())
        n_diff_pairs_by_imp_type = math.ceil((cls.num_pairs/2) /
                                             n_imp_types)
        n_diff_pairs_total = n_diff_pairs_by_imp_type * n_imp_types
        num_same_pairs = cls.num_pairs - n_diff_pairs_total

        # Check that the first block of pairs are all same-author
        for pair in cls.ds._pairs[:num_same_pairs]:
            assert pair[2] == 1
        # Check that the last block of pairs are all different-author
        for pair in cls.ds._pairs[num_same_pairs:]:
            assert pair[2] == 0

        # Check that the first and second tuple elements are of the right
        # shape for PyTorch model ingestion in the first and last pair
        assert isinstance(cls.ds._pairs[0][0], dict)
        assert 'input_ids' in cls.ds._pairs[0][0]
        assert 'attention_mask' in cls.ds._pairs[0][0]
        assert isinstance(cls.ds._pairs[0][1], dict)
        assert 'input_ids' in cls.ds._pairs[0][1]
        assert 'attention_mask' in cls.ds._pairs[0][1]
        assert isinstance(cls.ds._pairs[-1][0], dict)
        assert 'input_ids' in cls.ds._pairs[-1][0]
        assert 'attention_mask' in cls.ds._pairs[-1][0]
        assert isinstance(cls.ds._pairs[-1][1], dict)
        assert 'input_ids' in cls.ds._pairs[-1][1]
        assert 'attention_mask' in cls.ds._pairs[-1][1]