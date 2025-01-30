"""
This test suite tests this project's Logger class which is responsible
for logging train/eval results to disk and stdout.
"""
import os
import sys
from constants import ROOT_DIR
import shutil
import json
import torch
import pandas as pd

# Add src directory to sys.path
# Adapted from Taras Alenin's answer on StackOverflow at:
# https://stackoverflow.com/a/55623567
src_path = os.path.join(ROOT_DIR, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import custom modules
from logger import Logger  # noqa: E402
from generate_test_data import generate_test_data, destroy_test_data  # noqa: E402


class TestLogger:
    """
    A unified class to allow for easy setup and teardown of global and
    reused data and objects, and sharing of common methods. See
    `setup_class`, `teardown_class`.
    """
    @classmethod
    def setup_class(cls):
        cls.dataset = [
            'foo bar',  # A-0
            'foo bar',  # A-1
            'foo bar',  # A-2
            'foo bar',  # A-3
            'foo bar',  # U-0
            'foo bar',  # U-1
            'foo bar',  # U-2
            'foo bar',  # U-3
            'foo bar',  # notA-0
            'foo bar',  # notA-1
            'foo bar',  # notA-2
            'foo bar'   # notA-3
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
        cls.test_data_directory = 'logger_test_dir'

        # Generate the test data and relevent paths
        cls.paths, cls.canonical_class_labels = generate_test_data(
            cls.test_data_directory, cls.dataset, cls.metadata_rows)

        cls.expirement_name = 'test_test_experiment'
        cls.test_hyperparameters = {
            'foo': 'bar',
            'experiment_name': cls.expirement_name}
        cls.test_output_directory = '../data/test/model_out'
        cls.logger = Logger(cls.paths['normalized_dir'],
                            cls.test_output_directory,
                            cls.test_hyperparameters)

    @classmethod
    def teardown_class(cls):
        """
        Clean up test data.
        """

        # Remove everything after tests have run
        destroy_test_data(cls.test_data_directory)
        shutil.rmtree(cls.test_output_directory, ignore_errors=True)

    def test_logger_exists(cls):
        """
        Trivial test to ensure Logger can instantiate
        """

        assert isinstance(cls.logger, Logger)

    def test_output_structure(cls):
        """
        Test that the structure of the output directory created by Logger
        is correct.
        """
        assert os.path.exists(cls.test_output_directory)
        assert os.path.exists(cls.logger.out_path)
        assert os.path.exists(cls.logger.run_out_path)

    def test_hyperparameters(cls):
        """
        Test that the hyperparameters.json file is created with correct
        contents.
        """
        hyperparameters_path = os.path.join(cls.logger.run_out_path,
                                            'hyperparameters.json')
        assert os.path.exists(hyperparameters_path)
        with open(hyperparameters_path, 'r') as f:
            jsoncontent = json.loads(f.read())

        cls.test_hyperparameters['out_path'] = cls.logger.run_out_path

        assert jsoncontent == cls.test_hyperparameters

    def test_logging(cls):
        """
        Test the core logging functionality
        """
        # Test that the file can be updated with a first round of losses
        view_path = 'undistorted'
        batch_idx = 3
        epoch_losses = torch.Tensor([float(i)
                                     for i in range((batch_idx + 1))])
        epoch_idx = 0
        fold_idx = 0
        durations = [42 for i in range(batch_idx + 1)]

        cls.logger.log_train_results(view_path, epoch_losses, batch_idx,
                                     epoch_idx, fold_idx, durations, 8)

        path_to_losses_png = os.path.join(cls.logger.run_out_path,
                                          view_path,
                                           (f"{cls.expirement_name}"
                                            f"_fold_{fold_idx}"
                                            "_train_losses.png"))

        assert os.path.exists(path_to_losses_png)

        assert len(cls.logger.folds_losses) == 1
        assert len(cls.logger.folds_losses[0]) == len(epoch_losses)
        assert cls.logger.folds_losses[0] == [tensor.item()
                                              for tensor in epoch_losses]

        # Test that a second round of losses, on the same fold and epoch
        # can be updated
        batch_idx = 7
        epoch_losses = torch.Tensor([float(i)
                                     for i in range(batch_idx + 1)])
        durations = [42 for i in range(batch_idx + 1)]
        cls.logger.log_train_results(view_path, epoch_losses, batch_idx,
                                     epoch_idx, fold_idx, durations, 8)

        assert len(cls.logger.folds_losses) == 1
        assert len(cls.logger.folds_losses[0]) == len(epoch_losses)
        assert cls.logger.folds_losses[0] == [tensor.item()
                                              for tensor in epoch_losses]

        # Test that a third round of losses, on the same fold but new
        # epoch can be updated
        batch_idx = 3
        prev_epoch_losses = torch.Tensor([float(i) for i in range(8)])
        fold_losses = torch.cat(
            (prev_epoch_losses,
             torch.Tensor([float(i) for i in range(batch_idx + 1)])), 0)
        epoch_idx = 1
        durations = [42 for i in range(len(fold_losses))]
        cls.logger.log_train_results(view_path, fold_losses, batch_idx,
                                     epoch_idx, fold_idx, durations, 8)

        assert len(cls.logger.folds_losses) == 1
        assert cls.logger.folds_losses[0] == [tensor.item()
                                              for tensor
                                              in fold_losses]

        # Test that a forth round of losses, on the previous fold and
        # epoch can be updated
        batch_idx = 7
        prev_epoch_losses = torch.Tensor([float(i) for i in range(8)])
        fold_losses = torch.cat(
            (prev_epoch_losses,
             torch.Tensor([float(i) for i in range(batch_idx + 1)])), 0)
        epoch_idx = 1
        durations = [42 for i in range(len(fold_losses))]
        cls.logger.log_train_results(view_path, fold_losses, batch_idx,
                                     epoch_idx, fold_idx, durations, 8)

        assert len(cls.logger.folds_losses) == 1
        assert cls.logger.folds_losses[0] == [tensor.item()
                                              for tensor
                                              in fold_losses]

        # Test that a fifth round of losses, on the a new fold can be
        # updated
        prev_fold_losses = fold_losses
        batch_idx = 3
        fold_losses = torch.Tensor([float(i)
                                    for i in range(batch_idx + 1)])
        epoch_idx = 0
        fold_idx = 1
        durations = [42 for i in range(batch_idx + 1)]
        cls.logger.log_train_results(view_path, fold_losses, batch_idx,
                                     epoch_idx, fold_idx, durations, 8)

        path_to_losses_png = os.path\
            .join(cls.logger.run_out_path, 'undistorted',
                  (f"{cls.expirement_name}"
                   f"_fold_{fold_idx}_train_losses.png"))
        assert os.path.exists(path_to_losses_png)

        assert len(cls.logger.folds_losses) == 2
        assert cls.logger.folds_losses[0] == [tensor.item()
                                              for tensor
                                              in prev_fold_losses]
        assert cls.logger.folds_losses[1] == [tensor.item()
                                              for tensor
                                              in fold_losses]

        # Test that a sixth round of losses, on the the second fold can be
        # updated
        batch_idx = 7
        epoch_losses = torch.Tensor([float(i)
                                     for i in range(batch_idx + 1)])
        epoch_idx = 0
        durations = [42 for i in range(batch_idx + 1)]
        cls.logger.log_train_results(view_path, epoch_losses, batch_idx,
                                     epoch_idx, fold_idx, durations, 8)

        assert len(cls.logger.folds_losses) == 2
        assert cls.logger.folds_losses[0] == [tensor.item()
                                              for tensor
                                              in prev_fold_losses]
        assert cls.logger.folds_losses[1] == [tensor.item()
                                              for tensor in epoch_losses]

        # Test that a summary plot can be created
        prev_epoch_losses = epoch_losses
        batch_idx = 3
        epoch_losses = torch.Tensor([float(i)
                                     for i in range(batch_idx + 1)])
        epoch_idx = 1
        durations = [42 for i in range(batch_idx + 1)]
        cls.logger.log_train_results(view_path, epoch_losses, batch_idx,
                                     epoch_idx, fold_idx, durations, 8)
        batch_idx = 7
        epoch_losses = torch.Tensor([float(i)
                                     for i in range(batch_idx + 1)])
        epoch_losses = torch.cat((prev_epoch_losses, epoch_losses), 0)
        epoch_idx = 1
        durations = [42 for i in range(batch_idx + 1)]
        cls.logger.log_train_results(view_path, epoch_losses, batch_idx,
                                     epoch_idx, fold_idx, durations, 8)
        cls.logger.gen_summary_loss_plot('undistorted', 2, 8)

        path_to_losses_png = os.path\
            .join(cls.logger.run_out_path, 'undistorted',
                  f'{cls.expirement_name}_all_train_losses.png')
        assert os.path.exists(path_to_losses_png)

        # Test eval logging
        cls.logger.log_val_results([0.9, 0.9, 0.1, 0.1],
                                   [1, 0, 1, 0],
                                   view_path,
                                   0)
        fold_0_cm_hedged_file = (f'{cls.expirement_name}_'
                                 'fold_0_eval_cm_hedged.png')
        fold_0_cm_true_file = (f'{cls.expirement_name}_'
                               'fold_0_eval_cm_true.png')
        fold_0_cm_hedged_path = os.path.join(cls.logger.run_out_path,
                                             'undistorted',
                                             fold_0_cm_hedged_file)
        fold_0_cm_true_path = os.path.join(cls.logger.run_out_path,
                                           'undistorted',
                                           fold_0_cm_true_file)
        assert os.path.exists(fold_0_cm_hedged_path)
        assert os.path.exists(fold_0_cm_true_path)
        hedged_scores_file = (f'{cls.expirement_name}'
                              '_all_eval_hedged_scores.csv')
        true_scores_file = (f'{cls.expirement_name}'
                            '_all_eval_true_scores.csv')
        hedged_scores_path = os.path.join(cls.logger.run_out_path,
                                          'undistorted',
                                          hedged_scores_file)
        true_scores_path = os.path.join(cls.logger.run_out_path,
                                        'undistorted',
                                        true_scores_file)
        assert os.path.exists(hedged_scores_path)
        assert os.path.exists(true_scores_path)
        hedged_scores_df = pd.read_csv(hedged_scores_path, index_col=None)
        true_scores_df = pd.read_csv(true_scores_path, index_col=None)
        rows, columns = hedged_scores_df.shape
        assert rows == 1
        assert columns == 9
        rows, columns = true_scores_df.shape
        assert len(true_scores_df) == 1
        assert len(true_scores_df.columns) == 9

        cls.logger.log_val_results([0.9, 0.9, 0.1, 0.1],
                                   [0, 0, 1, 1],
                                   view_path,
                                   1)
        fold_1_cm_hedged_file = (f'{cls.expirement_name}'
                                 '_fold_1_eval_cm_hedged.png')
        fold_1_cm_true_file = (f'{cls.expirement_name}'
                               '_fold_1_eval_cm_true.png')
        fold_1_cm_hedged_path = os.path.join(cls.logger.run_out_path,
                                             'undistorted',
                                             fold_1_cm_hedged_file)
        fold_1_cm_true_path = os.path.join(cls.logger.run_out_path,
                                           'undistorted',
                                           fold_1_cm_true_file)
        assert os.path.exists(fold_1_cm_hedged_path)
        assert os.path.exists(fold_1_cm_true_path)

        hedged_scores_df = pd.read_csv(hedged_scores_path, index_col=None)
        true_scores_df = pd.read_csv(true_scores_path, index_col=None)
        rows, columns = hedged_scores_df.shape
        assert rows == 2
        assert columns == 9
        rows, columns = true_scores_df.shape
        assert len(true_scores_df) == 2
        assert len(true_scores_df.columns) == 9

        # Check that summary scores are created
        cls.logger.gen_summary_eval_metrics(view_path)
        hedged_scores_df = pd.read_csv(hedged_scores_path, index_col=None)
        true_scores_df = pd.read_csv(true_scores_path, index_col=None)
        rows, columns = hedged_scores_df.shape
        assert rows == 3
        assert columns == 9
        rows, columns = true_scores_df.shape
        assert len(true_scores_df) == 3
        assert len(true_scores_df.columns) == 9