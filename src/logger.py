"""
A logger class for logging results of `train.py`.
"""
import os
from datetime import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker


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
        self.out_path = out_path
        self.hyperparameters = hyperparameters
        self.folds_losses = []

        # SAVE RESULTS!
        # Setup directories to store the results of each run
        # Choose output directory and create if not already there
        if not os.path.exists(out_path):
            print(f"No directory found at {out_path}.")
            os.mkdir(out_path)
            print(f"Created new {out_path} directory.")

        # Create unique run dirname
        current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        self.run_out_path = os.path.join(
            out_path, '_'.join(current_time.split()))
        if not os.path.exists(self.run_out_path):
            os.mkdir(self.run_out_path)
            print(f"Created new {self.run_out_path} directory to store"
                  " results of this run.")

        self.hyperparameters["out_path"] = self.run_out_path

        # Save run hyperparameters
        params_file_path = os.path.join(self.run_out_path,
                                        "hyperparameters.json")
        with open(params_file_path, 'w') as f:
            f.write(json.dumps(hyperparameters))
            print(f"Hyperparameters saved to {params_file_path}.")

    def log_train_results(self,
                          view_path,
                          epoch_losses,
                          batch_idx,
                          epoch_idx,
                          fold_idx,
                          durations):
        """
        A method to save the results of a full k-folds cross-validation
        training run on the SiameseSBERT model using the LILADataset.

        :param view_path: <Required> Path to the source view data.
        :type view_path: str

        :param epoch_losses: <Required> List of losses accumulated per
        batch for the epoch thus far.
        :type epoch_losses: list

        :param batch_idx: <Required> The batch index.
        :type batch_idx: int

        :param epoch_idx: <Required> The epoch index.
        :type epoch_idx: int

        :param fold_idx: <Required> The fold index.
        :type fold_idx: int

        :param durations: <Required> List of durations accumulagted per
        batch for the epoch thus far.
        :type fold_idx: list
        """
        # Create view based directory
        # Adapted from:
        # https://stackoverflow.com/a/3925147
        view_out_dir = os.path.basename(os.path.normpath(view_path))
        view_out_path = os.path.join(self.run_out_path, view_out_dir)
        if not os.path.exists(view_out_path):
            os.mkdir(view_out_path)
            print(f"Created sub-directory {view_out_path} for storing"
                  f" results of {view_out_dir[3:5]}:{view_out_dir[8:]}"
                  " training.")

        # Create fold based file
        fold_png = f"k_{fold_idx}_train_losses.png"
        fold_png_path = os.path.join(view_out_path, fold_png)

        plain_losses = [tensor.item() for tensor in epoch_losses]

        if len(self.folds_losses) == fold_idx + 1:
            # Same fold as prior, extend plot
            self.folds_losses[-1] = plain_losses
        else:
            # New fold, start new plot
            self.folds_losses.append(plain_losses)

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(plain_losses, 'b-', label='Training Loss')
        plt.grid(True)
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Time')
        plt.legend()

        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)

        # Show the plot
        plt.tight_layout()
        plt.savefig(fold_png_path)

        print(f"Losses saved to {fold_png_path}.")
        avg_duration = sum(durations[-10:]) /\
            min(len(durations), 10)
        avg_loss = sum(epoch_losses[-10:]) /\
            min(len(epoch_losses), 10)
        print(f"Fold {fold_idx} > Epoch {epoch_idx} >"
              f" Batch {batch_idx}")
        print(f'Loss: {epoch_losses[-1].item():.4f},'
              f' 10AVG Loss: {avg_loss:.4f},'
              f' 10AVG Time/Batch: {avg_duration:.2f}s')

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
        # Create view based directory
        # Adapted from:
        # https://stackoverflow.com/a/3925147
        view_out_dir = os.path.basename(os.path.normpath(view_path))
        view_out_path = os.path.join(self.run_out_path, view_out_dir)
        # Create fold based file
        fold_png = "k_all_train_losses.png"
        fold_png_path = os.path.join(view_out_path, fold_png)

        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        line_types = ['-', '--', '-.', ':']
        # Create the plot
        fig, batch_ax = plt.subplots(figsize=(10, 6))
        num_batches = num_epochs * num_batches_per_epoch
        batch_numbers = np.arange(num_batches)

        for i, fold_losses in enumerate(self.folds_losses):
            color = colors[i % len(colors)]
            line_type = line_types[(i // len(line_types)) %
                                   len(self.folds_losses)]
            line_style = f"{color}{line_type}"

            plt.plot(fold_losses, line_style, alpha=0.3,
                     label=f'Fold {i} Training Loss')
            # batch_ax.plot(batch_numbers, fold_losses, line_style,
            #               alpha=0.3, label=f'Fold {i} Training Loss')
        # Tick only ints
        batch_ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Create secondary axis for epochs
        epoch_ax = batch_ax.twiny()

        # Set epoch ticks
        epoch_ticks = np.arange(0, num_batches, num_batches_per_epoch)
        epoch_labels = [f'Epoch {i}' for i in range(num_epochs)]

        epoch_ax.set_xticks(epoch_ticks)
        epoch_ax.set_xticklabels(epoch_labels)
        epoch_ax.set_xlim(batch_ax.get_xlim())

        batch_ax.set_xlabel('Batch')
        batch_ax.set_ylabel('Loss')

        # Plot thick black line showing mean
        # Adapted from:
        # https://stackoverflow.com/q/42758897
        means = np.mean(np.array(self.folds_losses), axis=0)
        batch_ax.plot(batch_numbers, means, 'k', alpha=0.3,
                      label='Mean Training Loss', linewidth=5.0)

        plt.grid(True)
        plt.title('Training Loss over Time')
        plt.legend()

        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)

        # Show the plot
        plt.tight_layout()
        plt.savefig(fold_png_path)

        print(f"Summary loss plot saved to {fold_png_path}.")

    def log_val_results(self,
                        similarities,
                        truth,
                        view_path,
                        fold_idx):
        """
        Method for saving and logging similarities and truths from
        evaluation runs.

        :param similarites: <Required> The predicted similarities of a
        sample to a class. 0 different-author, 1 same-author. Value are
        between 0 and 1, inclusive.
        :type similarities: list

        :param truth: <Required> The actual labels of the pairs, 0 or 1.
        :type truth: str

        :param view_path: <Required> The name of the directory from which
        data is being ingested. This identifies the 'view' and is used
        to construct the output directory.
        :type view_path: str

        :param fold_idx: <Required> The index of which fold the algorithm
        is currently on the the K-Folds training loop. Starting from 0.
        :type fold_idx: int
        """
        # Create view based directory
        # Adapted from:
        # https://stackoverflow.com/a/3925147
        view_out_dir = os.path.basename(os.path.normpath(view_path))
        view_out_path = os.path.join(self.run_out_path, view_out_dir)
        assert os.path.exists(view_out_path), ("Something went wrong,"
                                               f" {view_out_path} does"
                                               " not exist.")

        df_s = pd.DataFrame([(i, s) for i, s in enumerate(similarities)],
                            columns=['id', 'similarity'])
        df_t = pd.DataFrame([(i, s) for i, s in enumerate(truth)],
                            columns=['id', 'same'])

        # Create fold based files
        fold_sims_file = f"k_{fold_idx}_val_similarities.jsonl"
        fold_truths_file = f"k_{fold_idx}_val_truths.jsonl"
        fold_sims_file_path = os.path.join(view_out_path, fold_sims_file)
        fold_truths_path = os.path.join(view_out_path, fold_truths_file)

        # Write to JSONL
        with open(fold_sims_file_path, 'w') as f:
            f.write(df_s.to_json(orient='records', lines=True))
            print(f"Wrote similarities to {fold_sims_file_path}")
        with open(fold_truths_path, 'w') as f:
            f.write(df_t.to_json(orient='records', lines=True))
            print(f"Wrote truths to {fold_truths_path}")