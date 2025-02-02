"""
The primary training loop for `SiameseSBERT` on `LILADataset`.
"""
from constants import MODEL
import time
import torch
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
import numpy as np
from lila_dataset import LILADataset, collate_fn
from siamese_sbert import SiameseSBERT
from modified_contrastive_loss import ModifiedContrastiveLoss
from torch.cuda.amp import GradScaler, autocast

# Parallelization/Concurency
# Use CUDA if available, else use MPS if available. Fallback is CPU
device = torch.device("cuda" if torch.cuda.is_available()
                      else (
                        "mps"
                        if torch.backends.mps.is_available()
                        else "cpu"
                      ))
NUM_WORKERS = 0
PIN_MEMORY = False
# Clear cache initially
if device == "cuda":
    torch.cuda.empty_cache()
    # Use prefetching if cuda is available
    NUM_WORKERS = 4
    PIN_MEMORY = True
elif device == 'mps':
    torch.mps.empty_cache()
    if hasattr(torch.mps, 'synchronize'):
        torch.mps.synchronize()
    # Use prefetching if mps is available
    NUM_WORKERS = 4
    PIN_MEMORY = True


def train_epoch(model,
                train_dataloader,
                optimizer,
                scheduler,
                loss_function,
                accumulation_steps,
                device,
                fold_losses,
                train_durations,
                start_time,
                logger,
                batch_size,
                num_folds,
                num_epochs,
                num_pairs,
                val_durations,
                view_path,
                fold_idx,
                epoch_idx,
                num_val_batches,
                max_norm=None):
    """
    Train a single epoch

    :param model: <Required> The SiameseSBERT model to train.
    :type model: torch.nn.Module

    :param train_dataloader: <Required> The dataloader instantiated on the
    train dataset from `LILADataset` to use for training.
    :type train_dataloader: torch.utils.data.DataLoader

    :param optimizer: <Required> The gradient optimizer to use. Adam in
    this study's case.
    :type optimizer: torch.nn.Module

    :param loss_function: <Required> A loss function to use with the
    model. In the case of this study this is a custom Contrastive Loss
    sub-class of torch.nn.Module.
    :type loss_function: torch.nn.Module

    :param accumulation_steps: <Required> The number of accumulation steps
    to use for gradient accumulation.
    :type accumulation_steps: int

    :param device: <Required> The device to send batch tensor operations
    to for parallelization/concurrency optimizations.
    :type device: str

    :param fold_losses: <Required> A list of all losses calculated thus
    far during the training of the given fold.
    :type fold_losses: list

    :param train_durations: <Required> A list of all batch training
    durations calculated thus far.
    :type train_durations: list

    :param start_time: <Required> The time that the epoch's training
    started, in seconds.
    :type start_time: float

    :param logger: <Required> The logger class to save results of training
    and evaluation.
    :type logger: Logger

    :param batch_size: <Required> The size of the batches to train on.
    :type batch_size: int

    :param num_folds: <Required> The number of folds to run for K-Folds
    Cross-Validation.
    :type num_folds: int

    :param num_epochs: <Required> The number of epochs to train the entire
    train dataset for.
    :type num_epochs: int

    :param num_pairs: <Required> The total number of pairs requested by
    the user (train & val).
    :type num_pairs: int

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

    :param num_val_batches: <Required> The number of batches in each
    validation phase.
    :type num_val_batches: int

    :param max_norm: <Optional> Maximum normal vector to clip gradients to
    for numerical stability. If `None`, no gradient clipping will be
    applied.
    :type max_norm: float
    """
    torch.optim
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
        # - batch_anchor at index 0 - the batch tensor of
        #   chunks to be fed through the 'left' side of the
        #   Siamese network.
        # - batch_other at index 1 - the batch tensor of
        #   chunks to be fed through the 'right' side of the
        #   Siamese network.
        # - labels at index 2 - the ground truths for the
        #   pairs:
        #     - 1 = same-author
        #     - 0 = different-author

        # Move batches to device (MPS/CPU)
        batch_anchor = {k: v.to(device)
                        for k, v in batch_anchor.items()}
        batch_other = {k: v.to(device)
                       for k, v in batch_other.items()}
        labels = labels.to(device)
        
        # If cuda is available, use mixed precision training
        if device == "cuda":
            with autocast():
                # Forward pass
                anchor_embedding, other_embedding = model(
                    batch_anchor['input_ids'],
                    batch_anchor['attention_mask'],
                    batch_other['input_ids'],
                    batch_other['attention_mask']
                )
                # Calculate the contrastive loss of this batch and
                # normalize by accumulation steps
                loss = loss_function(anchor_embedding,
                                     other_embedding,
                                     labels) / accumulation_steps
        else:
            # Forward pass
            anchor_embedding, other_embedding = model(
                batch_anchor['input_ids'],
                batch_anchor['attention_mask'],
                batch_other['input_ids'],
                batch_other['attention_mask']
            )
            # Calculate the contrastive loss of this batch and
            # normalize by accumulation steps
            loss = loss_function(anchor_embedding,
                                 other_embedding,
                                 labels) / accumulation_steps
        # Save the batch loss
        # unnormalized loss for reporting
        fold_losses.append(loss * accumulation_steps)

        # If cuda is available, use mixed precision training
        if device == "cuda":
            # Backpropogation pass
            scaler.scale(loss).backward()
            # Clear gradients explicitly
            for param in model.parameters():
                param.grad = None
        else:
            loss.backward()

        # Clear MPS cache after each epoch
        if device == 'mps':
            torch.mps.empty_cache()
            if hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()

        # Step the optimizer every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping to help with observed dramatic loss
            # fluctuations.
            # Adapted from:
            # https://discuss.pytorch.org/t/gradient-clipping-and-gradient-accumulation-together/189753  # noqa: E501
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=max_norm)
            # Gradient Accumulation Implementation:
            # Adapted from
            # https://stackoverflow.com/a/78619879 [37]
            # If cuda is available, use mixed precision training
            if device == "cuda":
                # Replace optimizer.step() with:
                scaler.step(optimizer)
                scaler.update()
            else:
                # Update weights using calculated gradients from
                # Adam optimizer
                optimizer.step()
            # Clear out any existing gradients
            optimizer.zero_grad()

        # Calculate when the batch finished processing
        end_time = time.time()
        # Calculate the duration
        duration = end_time - start_time
        # Save the result
        train_durations.append(duration)
        # Reset clock for next calculation
        start_time = time.time()

        if batch_idx % 100 == 0 or\
           batch_idx == (len(train_dataloader) - 1):
            # LOG RESULTS!
            logger.log_train_results(view_path,
                                     fold_losses,
                                     batch_idx,
                                     epoch_idx,
                                     fold_idx,
                                     train_durations,
                                     len(train_dataloader))

            # Log relevent stats to CLI
            logger.echo_stats(batch_size,
                              num_folds,
                              num_epochs,
                              num_pairs,
                              train_durations,
                              val_durations,
                              view_path,
                              fold_idx,
                              epoch_idx,
                              batch_idx,
                              True,
                              len(train_dataloader),
                              num_val_batches,
                              device)

    # Adapted from:
    # https://machinelearningmastery.com/using-learning-rate-schedule-in-pytorch-training/  # noqa: E501
    # Update the learning rate ahead of new epoch
    scheduler.step()

    return fold_losses, start_time


def train(view_path,
          metadata_path,
          batch_size,
          accumulation_steps,
          chunk_size,
          margin_s,
          margin_d,
          epsilon,
          num_pairs,
          num_folds,
          num_epochs,
          logger,
          initial_lr,
          seed=None,
          max_norm=None):
    """
    The main training loop for SiameseSBERT on LILADataset.

    :param view_path: <Required> The path to the view's training data.
    Expects three subdirectories of the form `A`, `notA`, `U` to hold
    the authorial class' data.
    :type view_path: str

    :param metadata_path: <Required> The path to the 'metdata.csv' file
    on the same level as the view directory.
    :type metadata_path: str

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

    :param margin_s: <Required> The same-author margin for the contrastive
    loss function. This is the loss value below which to no longer 'pull'
    same-author embeddings together.
    :type margin_s: float

    :param margin_d: <Required> The different-author margin for the
    contrastive loss function. This is the loss value above which to no
    longer 'push' different-author embeddings appart.
    :type margin_d: float

    :param epsilon: <Required> This is a very small number to pass to the
    optimizer for numerical stability.
    :type epsilon: float

    :param num_pairs: <Required> The number of desired pairs to generate
    in total, for both training and evaluation.
    :type num_pairs: int

    :param num_folds: <Required> The number of k-folds splits to divide
    the dataset into for k-fold cross-validation.
    :type num_folds: int

    :param num_epochs: <Required> The number of times to train the
    SiameseSBERT model on the entire dataset.

    :param logger: <Required> `Logger` object from the `src/logger.py`
    module, responsible for setting up and logging run results to stdout
    and files on disk.
    :type logger: Logger

    :param initial_lr: <Required> Initial learning rate to set the
    learning rate scheduler at (LinearLR).
    :type initial_lr: float

    :param seed: <Optional> Seed to pass to random function such as KFolds
    and random.Random() for running reproducible expirements. Defautls to
    None which means the seed will be system clock.
    :type seed: int

    :param max_norm: <Optional> Maximum normal vector to clip gradients to
    for numerical stability. If `None`, no gradient clipping will be
    applied.
    :type max_norm: float
    """
    if seed is not None:
        print(f"\n--- Using seed {seed} to obtain reproducible"
              " results.\n")
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(0)
    if max_norm is not None:
        print(f"\n--- Clipping gradients to a max norm of {max_norm}.\n")

    if device == "cuda":
        # Setup gradient scaler for mixed precision training
        scaler = GradScaler()

    # Reset any existing splits
    LILADataset.reset_splits()

    print("""
    IGNORE FOLLOWING WARNING:
    We are pre-tokenizing the whole dataset and then chunking in lengths
    512 or less, rather than feeding this single payload through the
    model, as this warning is assuming.
    """)

    # Create list to store batch processing durations for
    # computational requirements assesment
    train_durations = []
    val_durations = []
    start_time = time.time()

    # Instantiate the full LILA dataset
    full_dataset = LILADataset(view_path,
                               metadata_path,
                               cnk_size=chunk_size,
                               num_pairs=num_pairs,
                               num_folds=num_folds,
                               seed=seed)

    # Perform a full training cycle `num_folds` times, changing the
    # train/validation sets each time (K-folds cross-validation)
    for fold_idx in range(num_folds):
        # Clear GPU cache before starting new fold
        if device == "cuda":
            torch.cuda.empty_cache()

        # Instantiate custom Siamese SBERT model and move to device
        model = SiameseSBERT(MODEL, device).to(device)

        # Instantiate custom contrastive loss fuction
        # 'modified contrastive loss'
        loss_function = ModifiedContrastiveLoss(margin_s=margin_s,
                                                margin_d=margin_d)

        # Instantiate Adam optimizer
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=initial_lr,
                                     eps=epsilon)

        # Ibrahim et al. (2023) [7:10] state:
        # > The list below includes the hyper-parameters considered,
        # > together with the values selected after fine tuning, using
        # > the validation set.
        # > • [...]
        # > • Scheduler: The learning rate scheduler was set to
        # >   ’warmuplinear’, which gradually increases the learning
        # >   rate during the warm-up phase and then linearly decays
        # >   it.
        # > • Warmup Steps: The warm-up steps were set to 0,
        # >   indicating that no warm-up phase was employed
        # > • Learning Rate: We set the initial learning rate to
        #     2e-05, which determines the step size during gradient
        #     descent optimization.
        #
        # In short, Ibrahim et al. used a linear learning rate
        # scheduler in their implementation.
        # Adapted from:
        # https://machinelearningmastery.com/using-learning-rate-schedule-in-pytorch-training/  # noqa: E501
        # Start at full learning rate (2e-05)
        # End at 10% of initial learning rate
        scheduler = LinearLR(optimizer, start_factor=1.0,
                             end_factor=0.1,
                             total_iters=num_epochs)

        # Create a list to save fold losses
        fold_losses = []

        # Extract the train and validation datasets
        train_dataset, val_dataset = full_dataset\
            .get_train_val_datasets(fold_idx)

        # Instantiate the dataloader for the train_dataset
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=PIN_MEMORY,
            num_workers=NUM_WORKERS
        )

        # Instantiate the dataloader for the val_dataset
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        for epoch_idx in range(num_epochs):
            # TRAIN
            ##############################################################
            # Set model to training mode
            model.train()

            fold_losses, start_time = train_epoch(model,
                                                  train_dataloader,
                                                  optimizer,
                                                  scheduler,
                                                  loss_function,
                                                  accumulation_steps,
                                                  device,
                                                  fold_losses,
                                                  train_durations,
                                                  start_time,
                                                  logger,
                                                  batch_size,
                                                  num_folds,
                                                  num_epochs,
                                                  num_pairs,
                                                  val_durations,
                                                  view_path,
                                                  fold_idx,
                                                  epoch_idx,
                                                  len(val_dataloader),
                                                  max_norm)
            # EVALUATE
            ##############################################################
            # Set model to evaluation mode
            model.eval()

            # Disable gradient calculation for inference
            with torch.no_grad():
                # Create a list to store similarity predictions for all
                # validation batches
                val_similarities = []
                # And a list to store the associated ground truth labels
                val_truths = []

                # Iterate over the val_dataloader one batch at a time
                for batch_idx, (batch_anchor,
                                batch_other,
                                labels) in enumerate(val_dataloader):
                    # Move batches to device (MPS/CPU)
                    batch_anchor = {k: v.to(device)
                                    for k, v in batch_anchor.items()}
                    batch_other = {k: v.to(device)
                                   for k, v in batch_other.items()}
                    labels = labels.to(device)

                    # Forward pass
                    anchor_embedding, other_embedding = model(
                        batch_anchor['input_ids'],
                        batch_anchor['attention_mask'],
                        batch_other['input_ids'],
                        batch_other['attention_mask']
                    )

                    # Calculate cosine similarity between embeddings
                    similarities = F.cosine_similarity(anchor_embedding,
                                                       other_embedding)

                    # Scale similarities from [-1,1] to [0,1] range
                    scaled_similarities = (similarities + 1) / 2

                    val_similarities.extend(scaled_similarities.tolist())
                    val_truths.extend(labels.tolist())

                    # Calculate when the batch finished processing
                    end_time = time.time()
                    # Calculate the duration
                    duration = end_time - start_time
                    # Save the result
                    val_durations.append(duration)
                    # Reset clock for next calculation
                    start_time = time.time()

                    if batch_idx % 100 == 0 or\
                       batch_idx == (len(val_dataloader) - 1):
                        # LOG RESULTS OF VALIDATION
                        logger.log_val_results(val_similarities,
                                               val_truths,
                                               view_path,
                                               fold_idx,
                                               epoch_idx,
                                               num_epochs)

                        metrics = {
                            'similarities': val_similarities,
                            'truths': val_truths,
                            'hedged_scores': logger.folds_hedged_scores
                                                   .iloc[-1].to_dict(),
                            'true_scores': logger.folds_true_scores
                                                 .iloc[-1].to_dict()
                        }

                        logger.echo_stats(batch_size,
                                          num_folds,
                                          num_epochs,
                                          num_pairs,
                                          train_durations,
                                          val_durations,
                                          view_path,
                                          fold_idx,
                                          epoch_idx,
                                          batch_idx,
                                          False,
                                          len(train_dataloader),
                                          len(val_dataloader),
                                          device)

    logger.gen_summary_loss_plot(view_path, num_epochs,
                                 len(train_dataloader))

    # Save model and all metrics for resumption/inference
    final_metrics = {
        'hedged_scores': logger.folds_hedged_scores[
            logger.folds_hedged_scores['fold'] == fold_idx
        ].to_dict('records'),
        'true_scores': logger.folds_true_scores[
            logger.folds_true_scores['fold'] == fold_idx
        ].to_dict('records')
    }

    logger.save_model(model,
                      optimizer,
                      fold_idx,
                      num_epochs-1,
                      fold_losses,
                      final_metrics,
                      view_path)

    # Log final memory stats if on cuda
    if device == "cuda":
        torch.cuda.synchronize()
        print(f"\nFinal GPU Memory Stats for {view_path}:")
        print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f}GB\n")
    logger.gen_summary_eval_metrics(view_path, num_epochs) 