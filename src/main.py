"""
This is the main training loop for the Siamese SBERT model on the
LILADataset.
"""
import sys
import os
import argparse
from constants import ROOT_DIR

# Add src directory to sys.path
# Adapted from Taras Alenin's answer on StackOverflow at:
# https://stackoverflow.com/a/55623567
src_path = os.path.join(ROOT_DIR, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import custom modules
from logger import Logger  # noqa: E402
from train import train  # noqa: E402


def train_per_distorted_view(args):
    """
    Wrapper for main training loop of the SiameseSBERT model.

    :param args: argparse object coming from the CLI which bundles the
    `batch_size`, `accumulation_steps`, `chunk_size`, `margin`, `epsilon`,
    and `num_pairs` parameters for use in the training loop for the
    `SiameseSBERT` model.
    :type args: argparse.Namespace
    """
    experiment_name = args.experiment_name
    dataset_path = args.dataset_path
    batch_size = args.batch_size
    accumulation_steps = args.accumulation_steps
    chunk_size = args.chunk_size
    margin = args.margin
    epsilon = args.epsilon
    num_pairs = args.num_pairs
    num_folds = args.num_folds
    num_epochs = args.num_epochs
    seed = args.seed
    max_norm = args.max_norm
    initial_lr = args.initial_lr

    # Verify types
    assert type(experiment_name) is str
    assert type(dataset_path) is str
    assert type(batch_size) is int
    assert type(accumulation_steps) is int
    assert type(chunk_size) is int
    assert type(margin) is float
    assert type(epsilon) is float
    assert type(num_pairs) is int
    assert type(num_folds) is int
    assert type(num_epochs) is int
    assert type(initial_lr) is float
    assert type(seed) is int or seed is None
    assert type(max_norm) is float or max_norm is None

    # Verify some boundary conditions
    for i in experiment_name:
        assert i in 'abcdefghijklmnopqrstuvwxyz0123456789_', \
            ('Experiment names must consist only of lowercase letters,'
             ' numbers and underscores.')
    assert os.path.exists(dataset_path)
    assert batch_size >= 1, ("ERROR: You chose a batch size of"
                             f" {batch_size}.\n Batch sizes less than 1"
                             " are not possible.")
    assert chunk_size > 2, ("ERROR: You chose a chunk size of"
                            f" {chunk_size}.\n Chunk sizes of less than 3"
                            " will not contain any actual data, due to"
                            " the special BERT [CLS] and [SEP] tokens.")
    assert margin > 0, (f"ERROR: You chose a margin of {margin}.\nA"
                        " margin of zero or less means different-author"
                        " pairs will never contribute to learning.")
    assert epsilon < 1e-5, (f"ERROR: You chosen an epsilon of {epsilon}."
                            "\n Consider choosing a smaller value.")
    assert num_pairs > 2, (f"ERROR: You chose {num_pairs} pairs.\n"
                           " Consider greater a higher number of pairs.")
    assert num_pairs % 2 == 0, ("ERROR: You chose an odd numer of pairs."
                                "\n For balancing reasons, it is required"
                                " that `num_pairs` be even.")
    assert 0 < num_folds < 11, ("Please choose k-folds splits between 1"
                                " and 10. 0 would result in no dataset,"
                                " and greater than 10 would result in a"
                                " validation split of less than 10%.")
    assert num_epochs > 0, ("Epochs must be greater than zero."
                            f" {num_epochs} was passed.")
    if max_norm is not None:
        assert max_norm > 0, ("Clipping gradients to a norm of 0 or less"
                              " will result in no convergence. Please"
                              " a value greater than zero.")

    undistorted_path = os.path.join(dataset_path, 'undistorted')
    assert os.path.exists(undistorted_path), \
        (f"The provided path {dataset_path} containes no subdirectory"
         " named 'undistorted'. Please provide a path to a properly"
         " formatted dataset directory. See help: `python train.py -h`.")
    metadata_path = os.path.join(dataset_path, 'metadata.csv')
    assert os.path.exists(metadata_path), \
        (f"The provided metadata {metadata_path} does not exist. Please"
         " ensure the root data view directory contains a 'metadata.csv'"
         " file. See help: `python train.py -h`.")
    # Create list to store all views to process
    views = [undistorted_path]
    for view_dir in os.listdir(dataset_path):
        view_path = os.path.join(dataset_path, view_dir)
        if view_dir != 'undistorted' and\
           os.path.isdir(view_path) and\
           view_dir[0] != '.':
            assert view_dir[:8] == 'DV-SA-k-'\
                or view_dir[:8] == 'DV-MA-k-', ("Your view directories"
                                                " are not in the form"
                                                " DV-<<S|M>>A-k-<<k>>."
                                                " See help: `python"
                                                " train.py -h`.")
            views.append(view_path)

    # SAVE RESULTS!
    logger = Logger(dataset_path, '../model_out', args.__dict__)

    try:
        for view_path in sorted([undistorted_path]):
            # Run training
            train(
                view_path,
                metadata_path,
                batch_size,
                accumulation_steps,
                chunk_size,
                margin,
                epsilon,
                num_pairs,
                num_folds,
                num_epochs,
                logger,
                initial_lr,
                seed,
                max_norm
            )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")


if __name__ == '__main__':
    # Adapted from:
    # https://github.com/JacobTyo/Valla/blob/main/valla/methods/AA_MHC.py

    # get command line args
    parser = argparse.ArgumentParser(
        description='Train SiamesSBERT on LILADataset')

    parser.add_argument('experiment_name',
                        type=str,
                        help=("Set the name of the experiment for easy"
                              " identification of experiment outputs."
                              " Must consist only of lowercase letters,"
                              " numbers, and underscores."))

    # Set the path to the dataset. Expects structure such as:
    # .
    # ├── undistorted
    # │   ├── A
    # │   ├── notA
    # │   └── U
    # ├── DV-MA-k-<<k>>
    # ├── DV-SA-k-<<k>>
    # ├── DV- etc...
    # └── metadata.csv
    parser.add_argument('dataset_path',
                        type=str,
                        help=("Set the path to the dataset. Expects a"
                              " path to a directory whith at least one"
                              " subdirectory named 'undistorted'"
                              " containing 'A', 'notA', and 'U'"
                              " sub-subirectories. Any further"
                              " subdirectories must start 'DV-MA-' or"
                              " 'DV-SA-' and and end 'k-<<k>>` where"
                              " <<k>> is an integer greater than one."))

    # Set the batch size
    parser.add_argument('batch_size',
                        type=int,
                        help=('<Required> [int] The actual batch size. If'
                              ' --accumulation_steps = 1, then this is'
                              ' also the effective batch size'))
    # Set the accumulation steps
    parser.add_argument('accumulation_steps',
                        type=int,
                        help=('<Required> [int] The number of batches to'
                              ' wait before updating model weights.'
                              ' `accumulation_steps * batch_size` renders'
                              ' the effective batch size.'))
    # Set the chunk size
    parser.add_argument('chunk_size',
                        type=int,
                        help=('<Required> [int] The size of the chunks to'
                              ' generate, in tokens, from the'
                              ' LILADataset, including special BERT [CLS]'
                              ' and [SEP] tokens.'))
    # Set the margin
    parser.add_argument('margin',
                        type=float,
                        help=('<Required> [float] The margin to pass to'
                              ' the contrastive loss function, setting'
                              ' the loss above which to stop \'pushing\''
                              ' different author pairs appart.'))
    # Set the margin
    parser.add_argument('epsilon',
                        type=float,
                        help=('<Required> [float] A very small number to'
                              ' pass to the optimizer for numerical'
                              ' stability.'))
    # Set the number of pairs to generate
    parser.add_argument('num_pairs',
                        type=int,
                        help=('<Required> [int] ** must be an even number'
                              ' ** The number of same/different-author'
                              ' pairs to generate from the LILADataset,'
                              ' in total (train & validate).'))

    # Set the number of k-folds splits
    parser.add_argument('num_folds',
                        type=int,
                        help=('<Required> [int] Number of divisions of'
                              ' of the dataset for k-folds'
                              ' cross-validation.'))

    # Set the number of epochs
    parser.add_argument('num_epochs',
                        type=int,
                        help=('<Required> [int] Number of times to train'
                              ' the model on the full dataset.'))

    # Set the number initial learning rate
    parser.add_argument('initial_lr',
                        type=float,
                        help=('<Required> [int] The initial learning rate'
                              ' to start the learning reate scheduler at'
                              '(LinearLR).'))

    # Set an optional seed for reproducable results
    parser.add_argument('-s',
                        '--seed',
                        type=int,
                        help=('<Optional> [int] Seed to make experiments'
                              ' reproducible.'))

    # Set an optional seed for gradient clipping
    parser.add_argument('-m',
                        '--max_norm',
                        default=None,
                        type=float,
                        help=('<Optional> [int] Max Norm value to clip'
                              ' gradients at. If None no gradient'
                              ' clipping will occur'))

    args = parser.parse_args()

    train_per_distorted_view(args)