"""
A logger class for logging results of `train.py`.
"""
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import json
from sklearn import metrics
from constants import ROOT_DIR, P1, P2
import pandas as pd
import time

# Add src directory to sys.path
# Adapted from Taras Alenin's answer on StackOverflow at:
# https://stackoverflow.com/a/55623567
baseline_path = os.path.join(ROOT_DIR, 'baseline')
if baseline_path not in sys.path:
    sys.path.insert(0, baseline_path)

from pan22_verif_evaluator import evaluate_all  # noqa: E402
from cngdist import correct_scores  # noqa: E402


class Logger():
    """
    A logging class for the `train.py` file to periodically save results
    of training runs to disk.
    """

    def __init__(self, dataset_path, out_path, hyperparameters):
        """
        The constructor for the logger, responsible for setting up
        members which the `log_train_results` and `log_eval_results`
        method will use, and writing initial directories and destination
        files to disk.

        :param dataset_path: <Required> Path to the source dataset which
        the calling routine will be training on.
        :type dataset_path: str

        :param out_path: <Required> Path to the root output directory,
        wherin Logger will create a sub-directory for storing the results
        of the given run.
        :type out_path: str

        :param hyperparameters: <Required> Dict of all hyperparameters for
        this run.
        :type hyperparameters: dict
        """
        self.dataset_path = dataset_path
        self.num_views = 0
        # Get the number of views to be processed for time calcs
        for view_dir in os.listdir(self.dataset_path):
            view_path = os.path.join(dataset_path, view_dir)
            if os.path.isdir(view_path) and view_dir[0] != '.':
                self.num_views += 1
        # Get instantiation time for time calcs
        self.start_time = time.time()
        self.out_path = out_path
        self.hyperparameters = hyperparameters
        self.experiment_name = self.hyperparameters['experiment_name']

        print("\n%%%%%%%%%%%%%  STARTING EXPERIMENT"
              f" {self.experiment_name}\n")

        self.folds_losses = []
        self.folds_hedged_scores = pd.DataFrame(columns=['experiment',
                                                         'fold',
                                                         'epoch',
                                                         'auc',
                                                         'c@1',
                                                         'f_05_u',
                                                         'F1',
                                                         'brier',
                                                         'overall',
                                                         'sub_overall'])
        self.folds_true_scores = pd.DataFrame(columns=['experiment',
                                                       'fold',
                                                       'epoch',
                                                       'auc',
                                                       'c@1',
                                                       'f_05_u',
                                                       'F1',
                                                       'brier',
                                                       'overall',
                                                       'sub_overall'])

        # SAVE RESULTS!
        # Setup directories to store the results of each run
        # Choose output directory and create if not already there
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        self.run_out_path = os.path.join(out_path, self.experiment_name)
        assert not os.path.exists(self.run_out_path), \
            (f"Experiment with name {self.experiment_name} already"
             " exists!")
        os.mkdir(self.run_out_path)

        self.hyperparameters["out_path"] = os.path.abspath(
            self.run_out_path)
        self.hyperparameters["dataset_path"] = os.path.abspath(
            self.dataset_path)

        # Save run hyperparameters
        params_file_path = os.path.join(self.run_out_path,
                                        "hyperparameters.json")
        with open(params_file_path, 'w') as f:
            f.write(json.dumps(hyperparameters))

    def reset_folds_losses(self):
        """
        Method to reset folds losses for a new 'view' run.
        """
        self.folds_losses = []

    def reset_val_scores(self):
        """
        Method to reset validation scores for a new 'view' run.
        """
        self.folds_hedged_scores = pd.DataFrame(columns=['fold',
                                                         'auc',
                                                         'c@1',
                                                         'f_05_u',
                                                         'F1',
                                                         'brier',
                                                         'overall',
                                                         'sub_overall'])
        self.folds_true_scores = pd.DataFrame(columns=['fold',
                                                       'auc',
                                                       'c@1',
                                                       'f_05_u',
                                                       'F1',
                                                       'brier',
                                                       'overall',
                                                       'sub_overall'])

    def log_train_results(self,
                          view_path,
                          fold_losses,
                          batch_idx,
                          epoch_idx,
                          fold_idx,
                          durations,
                          num_batches_per_epoch):
        """
        A method to save the results of a full k-folds cross-validation
        training run on the SiameseSBERT model using the LILADataset.

        :param view_path: <Required> Path to the source view data.
        :type view_path: str

        :param fold_losses: <Required> List of losses accumulated per
        batch for the fold thus far.
        :type fold_losses: list

        :param batch_idx: <Required> The batch index.
        :type batch_idx: int

        :param epoch_idx: <Required> The epoch index.
        :type epoch_idx: int

        :param fold_idx: <Required> The fold index.
        :type fold_idx: int

        :param durations: <Required> List of durations accumulagted per
        batch for the epoch thus far.
        :type fold_idx: list

        :param num_batches_per_epoch: <Required> Number of batches per
        epoch.
        :type num_batches_per_epoch: int
        """
        # Create view based directory
        # Adapted from:
        # https://stackoverflow.com/a/3925147
        view_out_dir = os.path.basename(os.path.normpath(view_path))
        view_out_path = os.path.join(self.run_out_path, view_out_dir)
        if not os.path.exists(view_out_path):
            os.mkdir(view_out_path)

        # Create fold based file
        fold_png = (f"{self.experiment_name}_fold_{fold_idx}"
                    "_train_losses.png")
        fold_png_path = os.path.join(view_out_path, fold_png)

        plain_losses = [tensor.item() for tensor in fold_losses]

        if len(self.folds_losses) == fold_idx + 1:
            # Same fold as prior, overwrite plot
            self.folds_losses[-1] = plain_losses
        else:
            # New fold, start new plot
            self.folds_losses.append(plain_losses)

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        batch_numbers = np.arange(len(plain_losses))
        self.plot_loss(ax,
                       batch_numbers,
                       plain_losses,
                       'Training Loss',
                       epoch_idx + 1,
                       num_batches_per_epoch,
                       'b', '-', 1.0)

        plt.grid(True)
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title(f"'{view_out_dir} (fold {fold_idx}) Mean Training"
                  " Loss'")
        plt.suptitle(f"Experiment: {self.experiment_name}")
        plt.legend()

        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)

        # Show the plot
        plt.tight_layout()
        plt.savefig(fold_png_path)
        plt.close()

    def get_exp_moving_average(self, durations, alpha=0.05):
        if len(durations) == 0:
            return 0
        # Disregard the first batch since its usually slower
        if len(durations) == 1:
            return durations[0]
        ema = durations[1]
        for d in durations[2:]:
            ema = alpha * d + (1 - alpha) * ema
        return ema

    def echo_stats(self,
                   batch_size,
                   num_folds,
                   num_epochs,
                   num_pairs,
                   train_durations,
                   val_durations,
                   view_path,
                   fold_idx,
                   epoch_idx,
                   batch_idx,
                   training,
                   num_train_batches,
                   num_val_batches):
        """
        Estimate remaining time for the full training run.

        Uses exponential moving average of batch times and accounts for
        current progress through folds/epochs/batches.

        :param batch_size: <Required> The size of the batches to train on.
        :type batch_size: int

        :param num_folds: <Required> The number of folds to run for
        K-Folds Cross-Validation.
        :type num_folds: int

        :param num_epochs: <Required> The number of epochs to train the
        entire train dataset for.
        :type num_epochs: int

        :param num_pairs: <Required> The total number of pairs requested
        by the user (train & val).
        :type num_pairs: int

        :param train_durations: <Required> A list of all batch training
        durations calculated thus far.
        :type train_durations: list

        :param val_durations: <Required> A list of durations per batch
        accumlated thus far for logging.
        :type val_durations: list

        :param view_path: <Required> The path the the specific distorted
        'view' dataset being trained on.
        :type view_path: str

        :param fold_idx: <Required> The index of the current fold being
        trained.
        :type fold_idx: int

        :param epoch_idx: <Required> The index of the current epoch being
        trained.
        :type epoch_idx: int

        :param batch_idx: <Required> The index of the current batch being
        processed.
        :type batch_idx: int

        :param training: <Required> Flag indicating if the K-Folds loop is
        currently in training (True) or validation (False) mode.
        :type training: bool

        :param num_train_batches: <Required> The number of batches in each
        train epoch.
        :type num_train_batches: int

        :param num_val_batches: <Required> The number of batches in each
        validation phase.
        :type num_val_batches: int
        """
        # Get view name for display
        view = os.path.basename(os.path.normpath(view_path))

        # Calc averages for train and validation
        train_avg = self.get_exp_moving_average(train_durations)
        if len(val_durations) > 0:
            val_avg = self.get_exp_moving_average(val_durations)
        else:
            # Val loop is ~5x faster than train
            val_avg = train_avg * 0.2

        # Estimate train time per epoch
        train_time_per_epoch = num_train_batches * train_avg
        # Estimate val time per epoch
        val_time_per_epoch = num_val_batches * val_avg

        # The estimates appear to be about 40% underestimated at the
        # start, approaching correct estimates gradually as the loop
        # progresses. Inflate the estimate by 40% on first output,
        # and gradually decrease this inflation to zero by the final
        # output.
        # Get the total number of batches
        total_batches = num_folds * num_epochs * (num_train_batches +
                                                  num_val_batches)
        batches_per_epoch = num_train_batches + num_val_batches
        batches_per_fold = batches_per_epoch * num_epochs
        # Get current absolute batch
        fold_batches_done = fold_idx * batches_per_fold
        epoch_batches_done = epoch_idx * batches_per_epoch
        current_abs_batch = fold_batches_done + epoch_batches_done

        if training:
            # Get remaining batches this epoch
            remaining_train_batches = num_train_batches - batch_idx
            remaining_val_batches = num_val_batches
            # Add remaining batches to abs batch
            current_abs_batch += batch_idx + 1

        else:
            # Get remaining batches this epoch
            remaining_train_batches = 0
            remaining_val_batches = num_val_batches - batch_idx
            # Add remaining batches to abs batch
            current_abs_batch += num_train_batches + batch_idx + 1

        # Estimate remaining train time in current epoch
        remaining_train_time_this_epoch = (remaining_train_batches *
                                           train_avg)
        remaining_val_time_this_epoch = (remaining_val_batches
                                         * val_avg)
        # Add val time
        remaining_time_this_epoch = (remaining_train_time_this_epoch
                                     + remaining_val_time_this_epoch)

        # Add remaining epochs in current fold
        remaining_epochs = num_epochs - (epoch_idx + 1)
        remaining_this_fold = remaining_epochs *\
            (train_time_per_epoch + val_time_per_epoch)

        # Add remaining folds
        remaining_folds = num_folds - (fold_idx + 1)
        time_per_fold = (num_epochs * num_train_batches * train_avg +
                         num_epochs * num_val_batches * val_avg)
        remaining_fold_time = remaining_folds * time_per_fold

        remain = (remaining_time_this_epoch +
                  remaining_this_fold +
                  remaining_fold_time)
        # Calculate the amount to inflate the estimation by
        inflation_amount = 1.6 * (1 - (current_abs_batch/total_batches))
        # Inflate
        remain *= inflation_amount

        # Calculate elapsed time since start
        elapsed = time.time() - self.start_time

        # Convert to human-readable times
        dys_elapsed = int(elapsed//86400)
        hrs_elapsed = int((elapsed % 86400)//3600)
        min_elapsed = int(((elapsed % 86400) % 3600)//60)
        sec_elapsed = int(((elapsed % 86400) % 3600) % 60)

        dys_remain = int(remain//86400)
        hrs_remain = int((remain % 86400)//3600)
        min_remain = int(((remain % 86400) % 3600)//60)
        sec_remain = int(((remain % 86400) % 3600) % 60)

        # Calculate average of last 10 losses
        avg_10_loss = 'und'
        i = len(self.folds_losses) - 1
        last_ten_losses = []
        while i >= 0 and len(last_ten_losses) < 10:
            to_collect = 10 - len(last_ten_losses)
            last_ten_losses.extend(self.folds_losses[i][-to_collect:])
            i -= 1
        if len(last_ten_losses) > 0:
            avg_10_loss = round(sum(last_ten_losses) /
                                len(last_ten_losses), 3)

        # Echo out
        if training:
            print(f"TRAIN: View {view:>13}  •  "
                  f"Fold {fold_idx + 1:>2}/{num_folds:<2}  •  "
                  f"Epoch {epoch_idx + 1:>2}/{num_epochs:<2}  •  "
                  f"Batch {batch_idx + 1:>4}/{num_train_batches:<4}  •  "
                  f"AVG10 Loss {avg_10_loss:<6}{' '*32}  •  "
                  f"Elapsed {dys_elapsed:>2}:{hrs_elapsed:02}:"
                  f"{min_elapsed:02}:{sec_elapsed:02}"
                  f" Remain {dys_remain:>2}:{hrs_remain:02}:"
                  f"{min_remain:02}:{sec_remain:02}")
        else:
            has_rows = len(self.folds_hedged_scores) > 0
            t_overall = self.folds_true_scores.iloc[-1]['overall']\
                if has_rows else 'und  '
            t_sub_overall = self.folds_true_scores.iloc[-1]['sub_overall']\
                if has_rows else 'und  '
            h_overall = self.folds_hedged_scores.iloc[-1]['overall']\
                if has_rows else 'und  '
            h_sub_overall = self.folds_hedged_scores.iloc[-1]['sub_overall']\
                if has_rows else 'und  '
            print(f"VAL:   View {view:>13}  •  "
                  f"Fold {fold_idx + 1:>2}/{num_folds:<2}  •  "
                  f"Epoch {epoch_idx + 1:>2}/{num_epochs:<2}  •  "
                  f"Batch {batch_idx + 1:>4}/{num_val_batches:<4}  •  "
                  f"TRUE O: {t_overall:<5}, SO {t_sub_overall:<5}"
                  f" HEDGED O: {h_overall:<5}, SO {h_sub_overall:<5}  •  "
                  f"Elapsed {dys_elapsed:>2}:{hrs_elapsed:02}:"
                  f"{min_elapsed:02}:{sec_elapsed:02}"
                  f" Remain {dys_remain:>2}:{hrs_remain:02}:"
                  f"{min_remain:02}:{sec_remain:02}")

    def gen_summary_loss_plot(self, view_path, num_epochs,
                              num_batches_per_epoch):
        """
        Method to generate single plot showing all loss plots for each
        K-Folds run in one plot.

        :param view_path: <Required> The name of the directory from which
        data is being ingested. This identifies the 'view' and is used
        to construct the output directory.
        :type view_path: str

        :param num_epochs: <Required> The number of epochs for the whole
        run for plotting.
        :type num_epochs: int

        :param num_batches_per_epoch: <Required> Number of batches per
        epoch.
        :type num_batches_per_epoch: int
        """
        # Get view based directory
        # Adapted from:
        # https://stackoverflow.com/a/3925147
        view_out_dir = os.path.basename(os.path.normpath(view_path))
        view_out_path = os.path.join(self.run_out_path, view_out_dir)
        fold_png = (f"{self.experiment_name}_all_train_losses.png")
        fold_png_path = os.path.join(view_out_path, fold_png)

        # Save the losses to disk.
        # Adapted from:
        # https://stackoverflow.com/a/28440249
        raw_losses_file = f'{self.experiment_name}_raw_losses.npy'
        raw_losses_path = os.path.join(view_out_path, raw_losses_file)
        np.save(raw_losses_path, np.array(self.folds_losses))

        # Example for loading
        # raw_losses_loaded = np.load(raw_losses_path)

        fig, ax = plt.subplots(figsize=(10, 6))
        # Given contastive-loss with cosine-similarity the expected range
        # of losses is 0 - 1
        ax.set_ylim(0.0001, 1.0)

        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        line_types = ['-', '--', '-.', ':']

        # Plot each fold
        # Save min length found for calculated the means
        min_len = float('inf')
        for i, fold_losses in enumerate(self.folds_losses):
            # Line style
            color = colors[i % len(colors)]
            line_type = line_types[(i // len(line_types)) %
                                   len(line_types)]
            label = f'Fold {i} Training Loss'
            # Plot loss line
            batch_numbers = np.arange(len(fold_losses))

            self.plot_loss(ax,
                           batch_numbers,
                           fold_losses,
                           label,
                           num_epochs,
                           num_batches_per_epoch,
                           color,
                           line_type,
                           1.0)

            # Updatae min length
            min_len = min(min_len, len(fold_losses))

        # Calculate and plot mean line up to minimum length
        truncated_losses = [losses[:min_len]
                            for losses in self.folds_losses]
        means = np.mean(np.array(truncated_losses), axis=0)
        self.plot_loss(ax,
                       np.arange(min_len),
                       means,
                       'Mean Training Loss',
                       num_epochs,
                       num_batches_per_epoch,
                       'k',
                       '-',
                       5.0)

        # Setup axes and labels
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        # Adapted from:
        # https://stackoverflow.com/a/27496811
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        plt.grid(True)
        plt.title(f"'{view_out_dir} (all folds) Mean Training Loss'")
        plt.suptitle(f"Experiment: {self.experiment_name}")
        plt.legend()
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        plt.savefig(fold_png_path)
        plt.close()

        self.reset_folds_losses()

    def plot_loss(self,
                  ax,
                  x,
                  y,
                  label,
                  num_epochs,
                  num_batches_per_epoch,
                  color,
                  line_type,
                  line_width):
        """
        Method to plot an individual loss curve.

        :param ax: <Required> Matplotlib axes to plot to
        :type ax: matplotlib.axes._axes.Axes

        :param x: <Required> X values to plot
        :type x: numpy.ndarray

        :param y: <Required> Y values to plot
        :type y: numpy.ndarray

        :param line_style: <Required> The line style to use for the plot.
        See https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html  # noqa: E501
        :type line_style: str

        :param label: <Required> The label to show for the plot in the
        legend.
        :type label: str

        :param num_epochs: <Required> The number of epochs for the whole
        run for plotting.
        :type num_epochs: int

        :param num_batches_per_epoch: <Required> Number of batches per
        epoch.
        :type num_batches_per_epoch: int

        :param color: The color string for matplot lib to apply to the
        plot.
        See https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html  # noqa: E501
        :type color: str

        :param color: The color string for matplotlib to apply to the
        plot.
        See https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html  # noqa: E501
        :type color: str

        :param line_type: The line type string for matplotlib to apply to
        the plot.
        See https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html  # noqa: E501
        :type line_type: str

        :param line_width: The line width for matplotlib to apply to the
        plot.
        See https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html  # noqa: E501
        :type line_width: float
        """
        line_style = f"{color}{line_type}"

        # Given contastive-loss with cosine-similarity the expected range
        # of losses is 0 - 1
        ax.set_ylim(0.0001, 1.0)

        ax.plot(x, y, line_style, alpha=0.3,
                label=label, linewidth=line_width)

        # Add epoch markers
        for epoch in range(num_epochs):
            batch_idx = epoch * num_batches_per_epoch
            if batch_idx < len(y):
                ax.scatter(batch_idx, y[batch_idx],
                           color=color, alpha=0.5, marker='o')

    def log_val_results(self,
                        val_similarities,
                        val_truths,
                        view_path,
                        fold_idx,
                        epoch_idx,
                        num_epochs):
        """
        Method for saving and logging similarities and truths from
        evaluation runs.

        :param val_similarities: <Required> The predicted similarities
        0 - 1.
        :type val_similarities: list

        :param val_truths: <Required> The ground truths 0 or 1.
        :type val_truths: list

        :param view_path: <Required> The name of the directory from which
        data is being ingested. This identifies the 'view' and is used
        to construct the output directory.
        :type view_path: str

        :param fold_idx: <Required> The fold index.
        :type fold_idx: int

        :param epoch_idx: <Required> The epoch index.
        :type epoch_idx: int

        :param num_epochs: <Required> The total number of epochs.
        :type epoch_idx: int
        """
        # Get view based directory
        # Adapted from:
        # https://stackoverflow.com/a/3925147
        view_out_dir = os.path.basename(os.path.normpath(view_path))
        view_out_path = os.path.join(self.run_out_path, view_out_dir)
        # EVALUATE MODEL WITH p1 = 0.45 and p2 = 0.54
        # Adapted from:
        # https://github.com/pan-webis-de/pan-code/blob/e0cfef01add9de4ac34fd20b337a5b5e1ac3b01e/clef22/authorship-verification/pan22_verif_evaluator.py  # noqa: E501
        hedged_similarities = np.array(list(
            correct_scores(val_similarities, p1=P1, p2=P2)))
        hedged_scores = evaluate_all(pred_y=hedged_similarities,
                                     true_y=val_truths)
        # Generate the overall score for only those metrics covered in
        # this study
        hedged_scores['sub_overall'] = round(((hedged_scores['auc'] +
                                               hedged_scores['c@1'] +
                                               hedged_scores['F1']) / 3),
                                             3)
        # Save the hedged scores
        hedged_scores_filename = (f"{self.experiment_name}"
                                  "_all_eval_hedged_scores.csv")
        hedged_scores_filepath = os.path.join(view_out_path,
                                              hedged_scores_filename)
        hedged_row = [self.experiment_name,
                      fold_idx,
                      epoch_idx,
                      hedged_scores['auc'],
                      hedged_scores['c@1'],
                      hedged_scores['f_05_u'],
                      hedged_scores['F1'],
                      hedged_scores['brier'],
                      hedged_scores['overall'],
                      hedged_scores['sub_overall']]
        if len(self.folds_hedged_scores) == (epoch_idx +
                                             (fold_idx * num_epochs)):
            # New epoch. Extend
            self.folds_hedged_scores.loc[len(self.folds_hedged_scores)] =\
                hedged_row
        else:
            # Processing. Overwrite
            self.folds_hedged_scores\
               .loc[len(self.folds_hedged_scores)-1] = hedged_row
        # Save
        self.folds_hedged_scores.to_csv(hedged_scores_filepath,
                                        index=False)

        # EVALUATE MODEL WITHOUT p1 and p2
        # Adapted from:
        # https://github.com/pan-webis-de/pan-code/blob/e0cfef01add9de4ac34fd20b337a5b5e1ac3b01e/clef22/authorship-verification/pan22_verif_evaluator.py  # noqa: E501
        true_scores = evaluate_all(pred_y=val_similarities,
                                   true_y=val_truths)
        # Generate the overall score for only those metrics covered in
        # this study
        true_scores['sub_overall'] = round(((true_scores['auc'] +
                                             true_scores['c@1'] +
                                             true_scores['F1']) / 3), 3)
        # Save the true scores
        true_scores_filename = (f"{self.experiment_name}"
                                "_all_eval_true_scores.csv")
        true_scores_filepath = os.path.join(view_out_path,
                                            true_scores_filename)
        true_row = [self.experiment_name,
                    fold_idx,
                    epoch_idx,
                    true_scores['auc'],
                    true_scores['c@1'],
                    true_scores['f_05_u'],
                    true_scores['F1'],
                    true_scores['brier'],
                    true_scores['overall'],
                    true_scores['sub_overall']]
        if len(self.folds_true_scores) == (epoch_idx +
                                           (fold_idx * num_epochs)):
            # New epoch. Extend
            self.folds_true_scores.loc[len(self.folds_true_scores)] =\
                true_row
        else:
            # Processing. Overwrite
            self.folds_true_scores\
               .loc[len(self.folds_true_scores)-1] = true_row
        # Save
        self.folds_true_scores.to_csv(true_scores_filepath,
                                      index=False)

        if epoch_idx == num_epochs - 1 and fold_idx == 0:
            # Only generate confusion matrices on the final epoch of the
            # last fold to make results compilation easier.

            # GENERATE HEDGED CONFUSION MATRICES
            # Adpated from:
            # https://www.w3schools.com/python/python_ml_confusion_matrix.asp
            # And:
            # https://github.com/pan-webis-de/pan-code/blob/e0cfef01add9de4ac34fd20b337a5b5e1ac3b01e/clef22/authorship-verification/pan22_verif_evaluator.py  # noqa: E501
            truths_filtered, sims_filtered = [], []

            for true, pred in zip(val_truths, hedged_similarities):
                if pred != 0.5:
                    truths_filtered.append(true)
                    sims_filtered.append(pred)

            sims_filtered = np.array(sims_filtered)
            sims_filtered = np.ma.fix_invalid(sims_filtered,
                                              fill_value=0.5)
            sims_filtered[sims_filtered >= 0.5] = 1
            sims_filtered[sims_filtered < 0.5] = 0
            truths_filtered = np.array(truths_filtered)
            truths_filtered = np.ma.fix_invalid(truths_filtered,
                                                fill_value=0.5)
            truths_filtered[truths_filtered >= 0.5] = 1
            truths_filtered[truths_filtered < 0.5] = 0
            hedged_confusion_matrix = metrics.confusion_matrix(
                truths_filtered, sims_filtered)
            cm_display = metrics.ConfusionMatrixDisplay(
                confusion_matrix=hedged_confusion_matrix,
                display_labels=[0, 1])
            cm_display.plot()
            # Save the hedged confusion matrix
            fold_png = (f"{self.experiment_name}_fold_{fold_idx}"
                        "_eval_cm_hedged.png")
            fold_png_path = os.path.join(view_out_path, fold_png)
            plt.title(f"'{view_out_dir} (fold {fold_idx}) Hedged Confusion"
                      " Matrix'")
            plt.suptitle(f"Experiment: {self.experiment_name}")
            plt.savefig(fold_png_path)
            plt.close()

            # GENERATE TRUE CONFUSION MATRICES
            # Adpated from:
            # https://www.w3schools.com/python/python_ml_confusion_matrix.asp
            # And:
            # https://github.com/pan-webis-de/pan-code/blob/e0cfef01add9de4ac34fd20b337a5b5e1ac3b01e/clef22/authorship-verification/pan22_verif_evaluator.py  # noqa: E501
            truths_filtered, sims_filtered = [], []

            for true, pred in zip(val_truths, val_similarities):
                if pred != 0.5:
                    truths_filtered.append(true)
                    sims_filtered.append(pred)

            sims_filtered = np.array(sims_filtered)
            sims_filtered = np.ma.fix_invalid(sims_filtered,
                                              fill_value=0.5)
            sims_filtered[sims_filtered >= 0.5] = 1
            sims_filtered[sims_filtered < 0.5] = 0
            truths_filtered = np.array(truths_filtered)
            truths_filtered = np.ma.fix_invalid(truths_filtered,
                                                fill_value=0.5)
            truths_filtered[truths_filtered >= 0.5] = 1
            truths_filtered[truths_filtered < 0.5] = 0
            true_confusion_matrix = metrics.confusion_matrix(
                truths_filtered, sims_filtered)
            cm_display = metrics.ConfusionMatrixDisplay(
                confusion_matrix=true_confusion_matrix,
                display_labels=[0, 1])
            cm_display.plot()
            # Save the true confusion matrix
            fold_png = (f"{self.experiment_name}"
                        f"_fold_{fold_idx}_eval_cm_true.png")
            fold_png_path = os.path.join(view_out_path, fold_png)
            plt.title(f"'{view_out_dir} (fold {fold_idx}) True Confusion"
                      " Matrix'")
            plt.suptitle(f"Experiment: {self.experiment_name}")
            plt.savefig(fold_png_path)
            plt.close()

    def gen_summary_eval_metrics(self, view_path, num_epochs):
        """
        Method for saving and logging similarities and truths from
        evaluation runs.

        :param view_path: <Required> The name of the directory from which
        data is being ingested. This identifies the 'view' and is used
        to construct the output directory.
        :type view_path: str
        """
        # Get view based directory
        # Adapted from:
        # https://stackoverflow.com/a/3925147
        # GENERATE HEDGED SCORES SUMMARY
        view_out_dir = os.path.basename(os.path.normpath(view_path))
        view_out_path = os.path.join(self.run_out_path, view_out_dir)

        mask = self.folds_hedged_scores['epoch'] == num_epochs - 1
        final_val_scores = self.folds_hedged_scores[mask]
        final_val_scores = final_val_scores.mean(0, numeric_only=True)
        final_val_scores = final_val_scores.round(3)

        hedged_summary_row = [
            self.experiment_name,
            "AVG",
            "N/A",
            final_val_scores['auc'],
            final_val_scores['c@1'],
            final_val_scores['f_05_u'],
            final_val_scores['F1'],
            final_val_scores['brier'],
            final_val_scores['overall'],
            final_val_scores['sub_overall']
        ]
        self.folds_hedged_scores.loc[len(self.folds_hedged_scores)] =\
            hedged_summary_row
        # Save the hedged scores
        hedged_scores_filename = (f"{self.experiment_name}"
                                  "_all_eval_hedged_scores.csv")
        hedged_scores_filepath = os.path.join(view_out_path,
                                              hedged_scores_filename)
        self.folds_hedged_scores.to_csv(hedged_scores_filepath,
                                        index=False)

        # GENERATE TRUE SCORES SUMMARY
        view_out_dir = os.path.basename(os.path.normpath(view_path))
        view_out_path = os.path.join(self.run_out_path, view_out_dir)

        mask = self.folds_true_scores['epoch'] == num_epochs - 1
        final_val_scores = self.folds_true_scores[mask]
        final_val_scores = final_val_scores.mean(0, numeric_only=True)
        final_val_scores = final_val_scores.round(3)

        true_summary_row = [
            self.experiment_name,
            "AVG",
            "N/A",
            final_val_scores['auc'],
            final_val_scores['c@1'],
            final_val_scores['f_05_u'],
            final_val_scores['F1'],
            final_val_scores['brier'],
            final_val_scores['overall'],
            final_val_scores['sub_overall']
        ]
        self.folds_true_scores.loc[len(self.folds_true_scores)] =\
            true_summary_row
        # Save the true scores
        true_scores_filename = (f"{self.experiment_name}"
                                "_all_eval_true_scores.csv")
        true_scores_filepath = os.path.join(view_out_path,
                                            true_scores_filename)
        self.folds_true_scores.to_csv(true_scores_filepath, index=False)

        self.reset_val_scores()