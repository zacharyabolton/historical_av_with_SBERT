"""
This is the main training loop for the Siamese SBERT model on the
LILADataset.
"""
import json
import sys
import os
import torch
from torch.utils.data import DataLoader
import time
import argparse
from datetime import datetime
from constants import ROOT_DIR, MODEL, LEARNING_RATE

# Add src directory to sys.path
# Adapted from Taras Alenin's answer on StackOverflow at:
# https://stackoverflow.com/a/55623567
src_path = os.path.join(ROOT_DIR, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import custom modules
from siamese_sbert import SiameseSBERT, ContrastiveLoss  # noqa: E402
from lila_dataset import LILADataset, collate_fn  # noqa: E402

# Parallelization/Concurency
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

if device == 'mps':
    torch.mps.empty_cache()


def save_results(run_dir, k, losses, args):
    """
    A function to save the results of a full k-folds cross-validation
    training run on the SiameseSBERT model using the LILADataset.

    :param run_dir: <Required> The directory to save the results files for
    this entire k-folds run for.
    :type run_dir: str

    :param k: <Required> The fold index data to save.
    :type k: int

    :param losses: <Required> A list of batch losses per k fold run.
    :type losses: list

    :param args: <Required> A dictionary mapping hyperparameters to their
    values for a given k-fold run.
    :type args: dict
    """
    if k == 0:
        # Save run metadata
        filename = "hyperparameters.txt"
        filepath = os.path.join(run_dir, filename)
        print('filepath:', filepath)
        with open(filepath, 'w') as f:
            f.write(json.dumps(args))
            print(f"Hyperparameters saved to {filepath}.")

    # Create k based filename
    filename = f"k_{k}_losses.csv"
    filepath = os.path.join(run_dir, filename)

    # CSV-ify losses and save to output dir
    string_losses = [str(tensor.item()) for tensor in losses]
    with open(filepath, 'w') as f:
        f.write(','.join(string_losses))
        print(f"Losses saved to {filepath}.")


def train(batch_size,
          accumulation_steps,
          chunk_size,
          margin,
          epsilon,
          num_pairs,
          n_splits,
          epochs):
    """
    The main training loop for SiameseSBERT on LILADataset.

    :param batch_size: <Required> The actual batch size, before
    considering gradient accumulation.
    I.e. `batch_size` = 'effective batch size' / accumulation_steps
    :type batch_size: int

    :param accumulation_steps: <Required> Number of times to divide and
    average the effective batch size before updating weights with gradient
    descent.
    :type accumulation_steps: int

    :param chunk_size: <Required> The length, in tokens, to divide the
    train/eval data into (including the special BERT [CLS] and [SEP]
    tokens.
    :type chunk_size: int

    :param margin: <Required> The margin for the contrastive loss
    function. This is the loss value above which to no longer 'push'
    different-author embeddings appart.
    :type margin: float

    :param epsilon: <Required> This is a very small number to pass to the
    optimizer for numerical stability.
    :type epsilon: float

    :param num_pairs: <Required> The number of desired pairs to generate
    in total, for both training and evaluation.
    :type num_pairs: int

    :param n_splits: <Required> The number of k-folds splits to divide the
    dataset into for k-fold cross-validation.
    :type n_splits: int

    :param epochs: <Required> The number of times to train the
    SiameseSBERT model on the entire dataset.
    """

    # SAVE RESULTS!
    # Setup directories to store the results of each run
    # Choose output directory and create if not already there
    output_dir = "../model_out"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Create unique run dirname
    current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    run_dir = os.path.join(output_dir, '_'.join(current_time.split()))
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    # Instantiate custom Siamese SBERT model and move to device
    model = SiameseSBERT(MODEL).to(device)

    # Instantiate custom contrastive loss function
    # TODO: Consider implementing 'modified contrastive loss' from
    # https://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    # [18] and Tyo Et. Al (2021) [15]
    loss_function = ContrastiveLoss(margin=margin)

    # Instantiate Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 eps=epsilon)

    # Reset any existing splits
    LILADataset.reset_splits()

    print("""
    IGNORE FOLLOWING WARNING:

    > Token indices sequence length is longer than the specified maximum
    > sequence length for this model (21972 > 512). Running this sequence
    > through the model will result in indexing errors

    We are pre-tokenizing the whole dataset and then chunking in lengths
    512 or less, rather than feeding this single payload through the
    model, as this warning is assuming.
    """)

    # Instantiate the full LILA dataset
    full_dataset = LILADataset('../data/normalized/undistorted',
                               '../data/normalized/metadata.csv',
                               cnk_size=chunk_size,
                               num_pairs=num_pairs,
                               n_splits=n_splits)

    try:
        # Create list to store batch processing durations for
        # computational requirements assesment
        durations = []
        start_time = time.time()

        # Perform a full training cycle `n_splits` times, changing the
        # train/validation sets each time (K-folds cross-validation)
        for fold_idx in range(n_splits):
            # Extract the train and validation datasets
            train_dataset, val_dataset = full_dataset\
                .get_train_val_datasets(fold_idx)

            # Instantiate the dataloader for the train_dataset
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
            )

            # TRAINING LOOP
            ##############################################################

            # Set model to training mode
            model.train()

            # Create list to store batch losses
            losses = []

            # Gradient Accumulation Implementation:
            # Adapted from
            # https://stackoverflow.com/a/78619879 [37]
            # Initialize running total for gradients
            optimizer.zero_grad()

            # Iterate over the train_dataloader one batch at a time
            for batch_idx, (batch_anchor,
                            batch_other,
                            labels) in enumerate(train_dataloader):
                # batch_content is a tuple containing three elements
                # coming from the PyTorch `DataLoader` object:
                # - batch_anchor at index 0 - the batch tensor of chunks
                #   to be fed through the 'left' side of the Siamese
                #   network.
                # - batch_other at index 1 - the batch tensor of chunks to
                #   be fed through the 'right' side of the Siamese
                #   network.
                # - labels at index 2 - the ground truths for the pairs:
                #     - 1 = same-author
                #     - 0 = different-author

                # Move batch to device (MPS/CPU)
                batch_anchor = {k: v.to(device)
                                for k, v in batch_anchor.items()}
                batch_other = {k: v.to(device)
                               for k, v in batch_other.items()}
                labels = labels.to(device)

                # Forward pass with error checking
                anchor_embedding, other_embedding = model(
                    batch_anchor['input_ids'],
                    batch_anchor['attention_mask'],
                    batch_other['input_ids'],
                    batch_other['attention_mask']
                )

                # Calculate the contrastive loss of this batch and normalize
                # by accumulation steps
                loss = loss_function(anchor_embedding, other_embedding,
                                     labels) / accumulation_steps
                # Save the batch loss
                # unnormalized loss for reporting
                losses.append(loss * accumulation_steps)

                # Gradient Accumulation Implementation:
                # Adapted from
                # https://stackoverflow.com/a/78619879 [37]
                # Backpropogation pass
                loss.backward()

                # Step the optimizer every accumulation_steps batches
                if (batch_idx + 1) % accumulation_steps == 0:
                    # Gradient Accumulation Implementation:
                    # Adapted from
                    # https://stackoverflow.com/a/78619879 [37]

                    # Update weights using calculated gradients from Adam
                    # optimizer
                    optimizer.step()
                    # Clear out any existing gradients
                    optimizer.zero_grad()

                    # Clear MPS cache periodically
                    if device == 'mps':
                        torch.mps.empty_cache()

                # Calculate when the batch finished processing
                end_time = time.time()
                # Calculate the duration
                duration = end_time - start_time
                # Save the result
                durations.append(duration)

                if batch_idx % 10 == 0:
                    avg_duration = sum(durations[-10:]) /\
                        min(len(durations), 10)
                    avg_loss = sum(losses[-10:]) / min(len(losses), 10)
                    print(f'Batch {batch_idx}, '
                          f'Loss: {loss.item():.4f}, '
                          f'10-rolling avg Loss: {avg_loss:.4f}, '
                          '10-rolling avg Time/Batch:'
                          f' {avg_duration:.2f}s')

                start_time = time.time()

            # SAVE RESULTS!
            save_results(run_dir,
                         fold_idx,
                         losses,
                         {
                             "batch_size": batch_size,
                             "accumulation_steps": accumulation_steps,
                             "chunk_size": chunk_size,
                             "margin": margin,
                             "epsilon": epsilon,
                             "num_pairs": num_pairs,
                             "n_splits": n_splits,
                             "epochs": epochs,
                         })

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    finally:
        if durations:
            # Do not calculate the first batch as setup adds overhead time
            # which inflates the influence of the first batch on the
            # average batch processing calculation innacurately.
            avg_duration = sum(durations[1:]) / max(len(durations) - 1, 1)
            print(f'\nTraining complete.'
                  f' Average time per batch: {avg_duration:.2f}s')

            # Calculate effective batch statistics
            total_samples = len(train_dataset)
            effective_batch_size = batch_size * accumulation_steps
            # estimated_total_batches = total_samples / effective_batch_size
            # estimated_total_time = (avg_duration *
            #                         estimated_total_batches *
            #                         accumulation_steps)
            # print(f'Total samples: {total_samples}')
            # print(f'Effective batch size: {effective_batch_size}')
            # print(f'Estimated total batches: {estimated_total_batches}')
            # print('Estimated total time for one epoch:'
            #       f' {estimated_total_time:.2f} seconds')


def run_training_loop(args):
    """
    Wrapper for main training loop of the SiameseSBERT model.

    :param args: argparse object coming from the CLI which bundles the
    `batch_size`, `accumulation_steps`, `chunk_size`, `margin`, `epsilon`,
    and `num_pairs` parameters for use in the training loop for the
    `SiameseSBERT` model.
    :type args: argparse.Namespace
    """
    batch_size = args.batch_size
    accumulation_steps = args.accumulation_steps
    chunk_size = args.chunk_size
    margin = args.margin
    epsilon = args.epsilon
    num_pairs = args.num_pairs
    n_splits = args.n_splits
    epochs = args.epochs

    # Verify types
    assert type(batch_size) is int
    assert type(accumulation_steps) is int
    assert type(chunk_size) is int
    assert type(margin) is float
    assert type(epsilon) is float
    assert type(num_pairs) is int
    assert type(n_splits) is int
    assert type(epochs) is int

    # Verify some boundary conditions
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
    assert 0 < n_splits < 11, ("Please choose k-folds splits between 1"
                               " and 10. 0 would result in no dataset,"
                               " and greater than 10 would result in a"
                               " validation split of less than 10%.")
    assert epochs > 0, ("Epochs must be greater than zero."
                        f" {epochs} was passed.")

    # Run training
    train(
        batch_size,
        accumulation_steps,
        chunk_size,
        margin,
        epsilon,
        num_pairs,
        n_splits,
        epochs
    )


if __name__ == '__main__':
    # Adapted from:
    # https://github.com/JacobTyo/Valla/blob/main/valla/methods/AA_MHC.py

    # get command line args
    parser = argparse.ArgumentParser(
        description='Train SiamesSBERT on LILADataset')

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
    parser.add_argument('n_splits',
                        type=int,
                        help=('<Required> [int] Number of divisions of'
                              ' of the dataset for k-folds'
                              ' cross-validation.'))

    # Set the number of epochs
    parser.add_argument('epochs',
                        type=int,
                        help=('<Required> [int] Number of times to train'
                              ' the model on the full dataset.'))

    args = parser.parse_args()

    run_training_loop(args)