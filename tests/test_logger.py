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
import math
import numpy as np

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
        # Parallelization/Concurency
        # Use CUDA if available, else use MPS if available. Fallback is
        # CPU
        cls.device = torch.device("cuda" if torch.cuda.is_available()
                                  else (
                                    "mps"
                                    if torch.backends.mps.is_available()
                                    else "cpu"
                                  ))
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
        cls.experiment_name = cls.expirement_name
        cls.dataset_path = os.path.abspath(cls.paths['normalized_dir'])
        cls.batch_size = 4
        cls.accumulation_steps = 1
        cls.chunk_size = 4
        cls.margin_s = 0.5
        cls.margin_d = 0.5
        cls.epsilon = 0.000001
        cls.num_pairs = 32
        cls.num_folds = 2
        cls.num_epochs = 2
        cls.initial_lr = 0.00001
        cls.seed = 0
        cls.max_norm = None
        cls.base_out_path = '../data/test/model_out'
        cls.run_out_path = os.path.join(cls.base_out_path,
                                        cls.experiment_name)
        cls.test_hyperparameters = {
            'experiment_name': cls.expirement_name,
            'dataset_path': cls.dataset_path,
            'batch_size': cls.batch_size,
            'accumulation_steps': cls.accumulation_steps,
            'chunk_size': cls.chunk_size,
            'margin_s': cls.margin_s,
            'margin_d': cls.margin_d,
            'epsilon': cls.epsilon,
            'num_pairs': cls.num_pairs,
            'num_folds': cls.num_folds,
            'num_epochs': cls.num_epochs,
            'initial_lr': cls.initial_lr,
            'seed': cls.seed,
            'max_norm': cls.max_norm,
            'run_out_path': cls.run_out_path
        }
        cls.train_batch_duration = 100
        cls.val_batch_duration = 60
        cls.logger = Logger(cls.paths['normalized_dir'],
                            cls.base_out_path,
                            cls.test_hyperparameters)

    @classmethod
    def teardown_class(cls):
        """
        Clean up test data.
        """

        # Remove everything after tests have run
        destroy_test_data(cls.test_data_directory)
        shutil.rmtree(cls.base_out_path, ignore_errors=True)

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
        assert os.path.exists(cls.base_out_path)
        assert os.path.exists(cls.logger.base_out_path)
        assert os.path.exists(cls.run_out_path)
        assert os.path.exists(cls.logger.run_out_path)
        assert cls.base_out_path == cls.logger.base_out_path
        assert cls.run_out_path == cls.logger.run_out_path

    def test_hyperparameters(cls):
        """
        Test that the hyperparameters.json file is created with correct
        contents.
        """
        hyperparameters_path = os.path.join(cls.run_out_path,
                                            'hyperparameters.json')
        assert os.path.exists(hyperparameters_path)
        with open(hyperparameters_path, 'r') as f:
            jsoncontent = json.loads(f.read())

        assert jsoncontent == cls.test_hyperparameters

    def process_view(cls, view):
        """
        Run a full mock-'logging-loop' simulating the way training/eval
        occurs, in order to generate mock data to test on.

        :returns folds_losses: A nested list of losses by folds accrued
        during mock train/eval. This is used by some methods for testing
        as the logger clears it's internal folds_losses after generating
        summary evaluation stats.
        :type list:
        """
        folds_losses = []
        view_path = os.path.join(cls.dataset_path, view)
        train_durations = []
        val_durations = []
        for fold_idx in range(cls.num_folds):
            fold_losses = []
            for epoch_idx in range(cls.num_epochs):
                # Mock train
                num_train_pairs = round(cls.num_pairs *
                                        ((cls.num_folds - 1)/
                                         cls.num_folds))
                num_train_batches = math.ceil(num_train_pairs /
                                              cls.batch_size)
                num_val_pairs = cls.num_pairs - num_train_pairs
                num_val_batches = math.ceil(num_val_pairs /
                                            cls.batch_size)

                for batch_idx in range(num_train_batches):
                    total_batches = (num_train_batches +
                                     num_val_batches) * cls.num_epochs
                    fold_losses.append((epoch_idx +
                                        batch_idx + 1e-04)/total_batches)
                    train_durations.append(cls.train_batch_duration)
                    # Mock process batch
                    cls.logger.log_train_results(
                        view_path,
                        torch.tensor(fold_losses),
                        batch_idx,
                        epoch_idx,
                        fold_idx,
                        train_durations,
                        num_train_batches
                    )
                    cls.logger.echo_stats(
                        cls.batch_size,
                        cls.num_folds,
                        cls.num_epochs,
                        cls.num_pairs,
                        train_durations,
                        val_durations,
                        view_path,
                        fold_idx,
                        epoch_idx,
                        batch_idx,
                        True,
                        num_train_batches,
                        num_val_batches,
                        cls.device
                    )

                for batch in range(num_val_batches):
                    val_similarities = [1
                                        for i in range(batch + 1)]
                    val_truths = [i % 2
                                  for i in range(batch + 1)]
                    val_durations.append(cls.val_batch_duration)
                    # Mock eval
                    cls.logger.log_val_results(
                        val_similarities,
                        val_truths,
                        view_path,
                        fold_idx,
                        epoch_idx,
                        cls.num_epochs
                    )

                    cls.logger.echo_stats(
                        cls.batch_size,
                        cls.num_folds,
                        cls.num_epochs,
                        cls.num_pairs,
                        train_durations,
                        val_durations,
                        view_path,
                        fold_idx,
                        epoch_idx,
                        batch_idx,
                        False,
                        num_train_batches,
                        num_val_batches,
                        cls.device
                    )

            folds_losses.append(fold_losses)

        cls.logger.gen_summary_loss_plot(
            view_path,
            cls.num_epochs,
            num_train_batches
        )
        cls.logger.gen_summary_eval_metrics(
            view_path,
            cls.num_epochs
        )

        return folds_losses

    def test_view_dirs_created(cls):
        """
        Test that view based output directories are correctly created.
        """
        views = [
            'DV-MA-k-0',
            'DV-MA-k-2',
            'DV-MA-k-8',
            'DV-SA-k-0',
            'DV-SA-k-2',
            'DV-SA-k-8',
            'undistorted'
        ]

        for view_path in views:
            cls.process_view(view_path)

        view_dirs_created = []
        for view_dir in os.listdir(cls.run_out_path):
            if not view_dir.startswith('.') and\
               not view_dir.endswith('.json'):
                view_dirs_created.append(view_dir)

        view_dirs_created = sorted(view_dirs_created)

        assert view_dirs_created == views

    def test_loss_imgs_created(cls):
        """
        Test that the loss plot and confusion matrices are created
        correctly.
        """
        view_path = 'undistorted'
        cls.process_view('undistorted')
        for fold_idx in range(cls.num_folds):
            path_to_losses_png = os.path.join(cls.run_out_path,
                                              view_path,
                                              (f"{cls.expirement_name}"
                                               f"_fold_{fold_idx}"
                                               "_train_losses.png"))
        assert os.path.exists(path_to_losses_png)
        path_to_cm_hedged_png = os.path.join(cls.run_out_path,
                                             view_path,
                                             (f"{cls.expirement_name}"
                                              "_fold_0"
                                              "_eval_cm_hedged.png"))
        assert os.path.exists(path_to_cm_hedged_png)
        path_to_cm_true_png = os.path.join(cls.run_out_path,
                                           view_path,
                                           (f"{cls.expirement_name}"
                                            "_fold_0_eval_cm_true.png"))
        assert os.path.exists(path_to_cm_true_png)
        path_to_all_train_losses = os.path.join(
            cls.run_out_path, view_path,
            f"{cls.expirement_name}_all_train_losses.png")
        assert os.path.exists(path_to_all_train_losses)

    def test_numpy_losses_saved(cls):
        """
        Test that accrued losses are saved in full in numpy (.npy) files,
        and have the correct shape.
        """
        view_path = 'undistorted'
        folds_losses = cls.process_view('undistorted')
        path_to_numpy = os.path.join(
            cls.run_out_path, view_path,
            f"{cls.expirement_name}_raw_losses.npy")
        assert os.path.exists(path_to_numpy)
        loaded_losses = np.load(path_to_numpy)
        print("folds_losses")
        print(np.array(folds_losses))
        print("loaded_losses")
        print(loaded_losses)
        assert np.array(folds_losses).shape == loaded_losses.shape

    def test_eval_metrics_saved(cls):
        """
        Test that the evaluation metrics are saved correctly in .csv
        files.
        """
        view_path = 'undistorted'
        cls.process_view('undistorted')
        path_to_hedged_metrics = os.path.join(cls.run_out_path, view_path,
                                              (f"{cls.expirement_name}"
                                               "_all_eval_"
                                               "hedged_scores.csv"))
        path_to_true_metrics = os.path.join(cls.run_out_path, view_path,
                                            (f"{cls.expirement_name}"
                                             "_all_eval_"
                                             "true_scores.csv"))
        assert os.path.exists(path_to_hedged_metrics)
        assert os.path.exists(path_to_true_metrics)
        hedged_metrics = pd.read_csv(path_to_hedged_metrics,
                                     index_col=None)
        true_metrics = pd.read_csv(path_to_true_metrics, index_col=None)
        assert len(hedged_metrics) == (cls.num_folds * cls.num_epochs) + 1
        assert len(true_metrics) == (cls.num_folds * cls.num_epochs) + 1