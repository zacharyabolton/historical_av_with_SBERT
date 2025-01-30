"""
This test suite tests the project's `train` function, which runs the
primary training loop, not included 'views' iteration.
"""
import os
import sys
from constants import ROOT_DIR
import shutil
import re

# Add src directory to sys.path
# Adapted from Taras Alenin's answer on StackOverflow at:
# https://stackoverflow.com/a/55623567
src_path = os.path.join(ROOT_DIR, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import custom modules
from logger import Logger  # noqa: E402
from generate_test_data import generate_test_data, destroy_test_data  # noqa: E402
from train import train  # noqa: E402


class TestTrain:
    """
    A unified class to allow for easy setup and teardown of global and
    reused data and objects, and sharing of common methods. See
    `setup_class`, `teardown_class`.
    """
    @classmethod
    def setup_class(cls):
        doc_length = 64
        cls.dataset = [
            ' '.join(['foo' for i in range(doc_length)]),  # A-0
            ' '.join(['foo' for i in range(doc_length)]),  # A-1
            ' '.join(['foo' for i in range(doc_length)]),  # A-2
            ' '.join(['foo' for i in range(doc_length)]),  # A-3
            ' '.join(['foo' for i in range(doc_length)]),  # U-0
            ' '.join(['foo' for i in range(doc_length)]),  # U-1
            ' '.join(['foo' for i in range(doc_length)]),  # U-2
            ' '.join(['foo' for i in range(doc_length)]),  # U-3
            ' '.join(['foo' for i in range(doc_length)]),  # notA-0
            ' '.join(['foo' for i in range(doc_length)]),  # notA-1
            ' '.join(['foo' for i in range(doc_length)]),  # notA-2
            ' '.join(['foo' for i in range(doc_length)])   # notA-3
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
                              'mock genre 1', None, 'A', 1, False,
                              len(cls.dataset[3].split())],

                             ['U-0.txt', None, None, 'mock genre 1',
                              None, 'U', None, False,
                              len(cls.dataset[4].split())],
                             ['U-1.txt', None, None, 'mock genre 1',
                              None, 'U', None, False,
                              len(cls.dataset[5].split())],
                             ['U-2.txt', None, None, 'mock genre 1',
                              None, 'U', None, False,
                              len(cls.dataset[6].split())],
                             ['U-3.txt', None, None, 'mock genre 1',
                              None, 'U', None, False,
                              len(cls.dataset[7].split())],

                             ['notA-0.txt', 'naauth', 'Imposter author',
                              'mock genre 1', 'John', 'notA', 0, False,
                              len(cls.dataset[8].split())],
                             ['notA-1.txt', 'naauth', 'Imposter author',
                              'mock genre 1', 'John', 'notA', 0, False,
                              len(cls.dataset[9].split())],
                             ['notA-2.txt', 'naauth', 'Imposter author',
                              'mock genre 1', 'John', 'notA', 0, False,
                              len(cls.dataset[10].split())],
                             ['notA-3.txt', 'naauth', 'Imposter author',
                              'mock genre 1', 'John', 'notA', 0, False,
                              len(cls.dataset[11].split())]]

        # Set name of directory where all test data for this test run will
        # be placed.
        cls.test_data_directory = 'train_test_dir'

        # Generate the test data and relevent paths
        cls.paths, cls.canonical_class_labels = generate_test_data(
            cls.test_data_directory, cls.dataset, cls.metadata_rows)

        cls.test_hyperparameters = {
            'batch_size': 8,
            'accumulation_steps': 1,
            'chunk_size': 8,
            'margin': 0.5,
            'epsilon': 1e-06,
            'num_pairs': 64,
            'num_folds': 2,
            'num_epochs': 1,
            'initial_lr': 2e-05,
            'experiment_name': 'test_test_experiment'
        }
        cls.test_output_directory = '../data/test/model_out'
        cls.logger = Logger(cls.paths['normalized_dir'],
                            cls.test_output_directory,
                            cls.test_hyperparameters)

        view_path = os.path.join(cls.paths['normalized_dir'],
                                 'undistorted')
        metadata_path = cls.paths['test_metadata_path']

        train(view_path,
              metadata_path,
              cls.test_hyperparameters['batch_size'],
              cls.test_hyperparameters['accumulation_steps'],
              cls.test_hyperparameters['chunk_size'],
              cls.test_hyperparameters['margin'],
              cls.test_hyperparameters['epsilon'],
              cls.test_hyperparameters['num_pairs'],
              cls.test_hyperparameters['num_folds'],
              cls.test_hyperparameters['num_epochs'],
              cls.logger,
              cls.test_hyperparameters['initial_lr'])


    @classmethod
    def teardown_class(cls):
        """
        Clean up test data.
        """

        # Remove everything after tests have run
        destroy_test_data(cls.test_data_directory)
        shutil.rmtree(cls.test_output_directory, ignore_errors=True)

    def test_losses_saved(cls):
        """
        Test that the model saves losses to disk in the format expected.
        """

        assert os.path.exists(cls.test_output_directory)
        sub_dirs = os.listdir(cls.test_output_directory)
        assert len(sub_dirs) == 1
        assert sub_dirs[0] == cls.test_hyperparameters['experiment_name']
        expected_run_out_path = os.path.join(cls.test_output_directory,
                                             sub_dirs[0])
        assert os.path.exists(expected_run_out_path)
        expected_view_out_dir = os.path.join(expected_run_out_path,
                                             'undistorted')
        assert os.path.exists(expected_view_out_dir)
        for k in range(cls.test_hyperparameters['num_folds']):
            exp_name = cls.test_hyperparameters['experiment_name']
            expected_fold_out_file = (f"{exp_name}"
                                      f"_fold_{k}_train_losses.png")
            expected_fold_out_dir = os.path.join(
                expected_view_out_dir,
                expected_fold_out_file
                )
            assert os.path.exists(expected_fold_out_dir)