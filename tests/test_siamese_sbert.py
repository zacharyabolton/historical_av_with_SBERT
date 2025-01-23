"""
This test suite tests this project's implementaion of the abstract PyTorch
`nn.Module` class.

https://pytorch.org/docs/stable/generated/torch.nn.Module.html
"""
import os
import sys
import torch
import math
from torch.utils.data import DataLoader
from transformers.models.bert.modeling_bert import BertModel
from constants import ROOT_DIR, MODEL, INITIAL_LEARNING_RATE
from transformers import AutoTokenizer


# Add src directory to sys.path
# Adapted from Taras Alenin's answer on StackOverflow at:
# https://stackoverflow.com/a/55623567
src_path = os.path.join(ROOT_DIR, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import custom modules
from lila_dataset import LILADataset, collate_fn  # noqa: E402
from siamese_sbert import SiameseSBERT, ContrastiveLoss  # noqa: E402
from generate_test_data import generate_test_data, destroy_test_data  # noqa: E402


class TestSiameseSBERT:
    """
    A unified class to allow for easy setup and teardown of global and
    reused data and objects, and sharing of common methods. See
    `setup_class`, `teardown_class`, and `check_chunks`.
    """
    @classmethod
    def setup_class(cls):
        """
        Setup global and reused test data and objects.
        """
        # The following dataset will have the following characteristics:
        # - all A texts will consist of equal length strings of the word
        #   'foo' repeated
        # - all U texts will consist of equal length strings of the word
        #   'foo' repeated
        # - all notA texts will consist of equal length strings of the
        #   word 'bar' repeated
        # Should should allow for niave checking of model's discriminatory
        # ability.
        doc_length = 2**9
        cls.dataset = [
            ' '.join(['foo' for i in range(doc_length)]),     # A-0
            ' '.join(['foo' for i in range(doc_length)]),     # A-1
            ' '.join(['foo' for i in range(doc_length)]),     # A-2
            ' '.join(['foo' for i in range(doc_length)]),     # A-3
            ' '.join(['foo' for i in range(doc_length)]),     # U-0
            ' '.join(['foo' for i in range(doc_length)]),     # U-1
            ' '.join(['foo' for i in range(doc_length)]),     # U-2
            ' '.join(['foo' for i in range(doc_length)]),     # U-3
            ' '.join(['bar' for i in range(doc_length)]),     # notA-0
            ' '.join(['bar' for i in range(doc_length)]),     # notA-1
            ' '.join(['bar' for i in range(doc_length)]),     # notA-2
            ' '.join(['bar' for i in range(doc_length)]),     # notA-3
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
        cls.test_data_directory = 'sbert_test_data'

        # Generate the test data and relevent paths
        cls.paths, cls.canonical_class_labels = generate_test_data(
            cls.test_data_directory, cls.dataset, cls.metadata_rows)

        # Send all Tensor operations to mps if available, and cpu if not
        # TODO: Get this to work on GPU systems as well (CUDA)
        cls.device = "mps" if torch.backends.mps.is_available() else "cpu"
        if cls.device == 'mps':
            torch.mps.empty_cache()

        # Hyperparameters
        cls.batch_size = 32
        cls.cnk_size = 16
        cls.num_pairs = 10_000
        cls.seed = 1
        cls.epsilon = 1e-6  # Same as Ibrahim et al. (2023) [7:10]
        cls.margin = 1

        # Instantiate test Siamese SBERT model and move to device
        cls.model = SiameseSBERT(MODEL).to(cls.device)

        # Instantiate custom contrastive loss function
        # TODO: Consider implementing 'modified contrastive loss' from
        # https://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf [18]
        # and
        # Tyo Et. Al (2021) [15]
        cls.loss_function = ContrastiveLoss(margin=cls.margin)

        # Instantiate Adam optimizer
        cls.optimizer = torch.optim.Adam(cls.model.parameters(),
                                         lr=INITIAL_LEARNING_RATE,
                                         eps=cls.epsilon)

        # Reset any existing splits
        LILADataset.reset_splits()

        # Insantiate PyTorch dataset object and dataloadeer with mock data
        # and toy parameters for testing
        cls.ds = LILADataset(cls.paths['undistorted_dir'],
                             cls.paths['test_metadata_path'],
                             cnk_size=cls.cnk_size,
                             num_pairs=cls.num_pairs,
                             seed=cls.seed)

    @classmethod
    def teardown_class(cls):
        """
        Clean up test data.
        """

        # Remove everything after tests have run
        destroy_test_data(cls.test_data_directory)

    def test_sbert_exists(cls):
        """
        Trivial test to ensure SiameseSBERT can instantiate
        """
        assert isinstance(cls.model, SiameseSBERT)
    def test_encoder_exists(cls):
        """
        Trivial test to ensure SBERT encoder exists on `SiameseSBERT`
        object.
        """
        assert isinstance(cls.model.encoder, BertModel)

    def check_chunks(cls, anchor_chunk, other_chunk, label):
        """
        Helper method that checks several 'sane' and boundary conditions
        on two chunks processed by the
        `LILADataset` -> `torch.utils.data.DataLoader` pipeline.
        It expects two tokenized chunks of length less than 512, with
        special BERT [CLS] and [SEP] tokens on either end. It also expects
        a label of 1 or 0 indicating same/different-author pair.

        :param anchor_chunk: <Required> A PyTorch tensor representing a
        tokenized and chunked sentence which is part of a same or
        different author pair.
        :type anchor_chunk: torch.Tensor
        :param other_chunk: <Required> A PyTorch tensor representing a
        tokenized and chunked sentence which is part of a same or
        different author pair.
        :type other_chunk: torch.Tensor
        :param label: <Required> The label indicating which type of pair
        the two input chunks belong to - 1 for same-author, 0 for
        different-author.
        :type label: int (0 or 1)
        """
        # Check length
        assert anchor_chunk.size(0) <= 512
        assert other_chunk.size(0) <= 512
        assert anchor_chunk.size(0) > 2
        assert other_chunk.size(0) > 2
        # Check that the a and b batch chunks start and end with
        # special BERT [CLS] and [SEP] tokesn (101, and 102)
        assert anchor_chunk[0] == torch.Tensor([101]).to(
            cls.device)
        assert anchor_chunk[-1] == torch.Tensor([102]).to(
            cls.device)
        assert other_chunk[0] == torch.Tensor([101]).to(
            cls.device)
        assert other_chunk[-1] == torch.Tensor([102]).to(
            cls.device)

        # Check that everything in between the special BERT tokens
        # per chunk is equal given our toy dataset
        anchor_chnk_input_ids = anchor_chunk[1:cls.cnk_size - 1]
        for i, input_id in enumerate(anchor_chnk_input_ids[:-1]):
            assert torch.equal(input_id,
                               anchor_chnk_input_ids[i+1])
        other_chnk_input_ids = other_chunk[1:cls.cnk_size - 1]
        for i, input_id in enumerate(other_chnk_input_ids[:-1]):
            assert torch.equal(input_id,
                               other_chnk_input_ids[i+1])

        # Since the toy dataset is just the same word repeated, differing
        # only in which word based on A/notA canonical class, check that
        # their input ids are equal (or different in the case of a batch
        # that has a first half of one class and a second half of another)
        if label == 0:
            assert torch.equal(anchor_chnk_input_ids,
                               other_chnk_input_ids) is False
        elif label == 1:
            assert torch.equal(anchor_chnk_input_ids,
                               other_chnk_input_ids) is True
        else:
            assert False, "Error: Label is something other than 0 or 1."

    def test_dataloader(cls):
        """
        Test that the instantiated `torch.utils.data.DataLoader` behaves
        as expected and outputs correct batches containing correctly
        labeled and grouped tokenized chunks.
        """
        dataloader = DataLoader(
            cls.ds,
            batch_size=cls.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        actual_num_batches = len(dataloader)

        # Test that the length of the dataloader is correct
        assert actual_num_batches == math.ceil(
            cls.num_pairs/cls.batch_size)

        for i, (batch_anchor,
                batch_other, labels) in enumerate(dataloader):
            if i != (len(dataloader) - 1):
                # Test that the batch sizes are correct
                assert len(batch_anchor['input_ids']) == cls.batch_size
                assert len(batch_other['input_ids']) == cls.batch_size
                assert len(labels) == cls.batch_size

            # Without shuffling, the first half of the labels seen should
            # all be same-author (1)
            if i == actual_num_batches//2 and actual_num_batches % 2 != 0:
                # Odd number of batches, so the first segement of each
                # batch should be same-auth, and the final segment should
                # be diff-auth. However, because of internal balancing in
                # the `LILADataset` object, it is not trivial to determine
                # what proportion of this batch each class shares.
                # Therefore we will niavely check the first and last
                # values for labels of 1 and 0 respectively.
                first_label = torch.Tensor([labels[0]])
                expected_first_label = torch.Tensor([1])
                last_label = torch.Tensor([labels[-1]])
                expected_last_label = torch.Tensor([0])
                # Adapted from:
                # https://stackoverflow.com/a/54187453
                assert torch.equal(first_label, expected_first_label)
                assert torch.equal(last_label, expected_last_label)

            elif i < actual_num_batches/2:
                # If first half of batches all should be same-author
                expected = torch.Tensor(
                    [1 for label in range(cls.batch_size)])
                assert torch.equal(labels, expected)
            else:
                # If second half of batches (but not final) all should be
                # diff-author
                expected = torch.Tensor(
                    [0 for label in range(cls.batch_size)])
                if i == (len(dataloader) - 1):
                    # Were on the last batch, so it could be shorter
                    # don't check
                    continue
                assert torch.equal(labels, expected)

            for (chunk_anchor,
                 chunk_other,
                 label) in zip(batch_anchor['input_ids'],
                               batch_other['input_ids'], labels):
                # Check the contents of the a and b batch chunks are
                # correct
                cls.check_chunks(chunk_anchor, chunk_other, label)

    def test_train_predict_cycle(cls):
        """
        Test 'sane' values for a forward training pass and prediction run.

        This test performs a niave training on chunks of toy data from the
        authorial class (A) and non-authorial class (notA) and then makes
        a single prediction on strings that are identical to the training
        data in all ways but length, trivially ensuring that the
        predictions on the same class is greater than the diff class.
        """
        # Make results reproducable
        # Adapted from: https://pytorch.org/docs/stable/notes/randomness.html
        torch.manual_seed(0)

        dataloader = DataLoader(
            cls.ds,
            batch_size=cls.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        # Set model to training mode
        cls.model.train()

        # Create list to store batch losses
        losses = []

        # Iterate over the dataloader for only one batch
        for batch_idx, (batch_anchor, batch_other, labels) in enumerate(
          dataloader):
            # Move batch to device (MPS/CPU)
            batch_anchor = {k: v.to(cls.device)
                            for k, v in batch_anchor.items()}
            batch_other = {k: v.to(cls.device)
                           for k, v in batch_other.items()}
            labels = labels.to(cls.device)

            # Initialize running total for gradients
            cls.optimizer.zero_grad()

            # Forward pass with error checking
            anchor_embeddings, other_embeddings = cls.model(
                batch_anchor['input_ids'],
                batch_anchor['attention_mask'],
                batch_other['input_ids'],
                batch_other['attention_mask']
            )

            # Calculate the contrastive loss of this batch and normalize
            # by accumulation steps
            loss = cls.loss_function(anchor_embeddings, other_embeddings,
                                     labels)
            # Save the batch loss
            # unnormalized loss for reporting
            losses.append(loss)

            # Backpropogation pass
            loss.backward()

            # Update weights using calculated gradients from Adam
            # optimizer
            cls.optimizer.step()

        # Sentences we want sentence embeddings for
        sentence_pairs = [['foo foo', 'foo foo'],
                          ['foo foo', 'bar bar']]

        # Tokenize and batch our toy data for a prediction pass through
        # the model
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        tokenized_pairs = []
        for pair in sentence_pairs:
            tokenized_pair = []
            tokenized_pair.append(tokenizer(
                pair[0],
                return_tensors="pt",
                padding=False,
                truncation=False))
            tokenized_pair.append(tokenizer(
                pair[1],
                return_tensors="pt",
                padding=False,
                truncation=False))
            tokenized_pairs.append(tokenized_pair)
        anchor_batched = {
            'input_ids': torch.cat(
                [tokenized_pairs[0][0]['input_ids'],
                 tokenized_pairs[1][0]['input_ids']],
                dim=0).to(cls.device),
            'attention_mask': torch.cat(
                [tokenized_pairs[0][0]['attention_mask'],
                 tokenized_pairs[1][0]['attention_mask']],
                dim=0).to(cls.device)
        }
        other_batched = {
            'input_ids': torch.cat(
                [tokenized_pairs[0][1]['input_ids'],
                 tokenized_pairs[1][1]['input_ids']],
                dim=0).to(cls.device),
            'attention_mask': torch.cat(
                [tokenized_pairs[0][1]['attention_mask'],
                 tokenized_pairs[1][1]['attention_mask']],
                dim=0).to(cls.device)
        }

        # Forward pass through the model
        (anchor_embedding,
         other_embedding) = cls.model(anchor_batched['input_ids'],
                                      anchor_batched['attention_mask'],
                                      other_batched['input_ids'],
                                      other_batched['attention_mask'])

        # Test that the embeddings have the correct size
        assert anchor_embedding.size(0) == 2
        assert other_embedding.size(0) == 2
        assert anchor_embedding.size(1) == 384
        assert other_embedding.size(1) == 384

        # Calculate cosine similarity between embeddings
        similarities = torch.nn.functional.\
            cosine_similarity(anchor_embedding, other_embedding)

        # Test that the similarities are between -1 and 1 as is expected
        # for cosine similarity measures
        assert similarities[similarities < -1].size(0) == 0
        assert similarities[similarities > 1].size(0) == 0

        # Scale similarities from [-1,1] to [0,1] range
        scaled_similarities = (similarities + 1) / 2
        # Test scaling
        assert scaled_similarities[scaled_similarities < 0].size(0) == 0
        assert scaled_similarities[scaled_similarities > 1].size(0) == 0

        # Test that the similarities are of the right size and some
        # trivial convergence is happening
        assert scaled_similarities.size(0) == 2
        assert scaled_similarities[0] > scaled_similarities[1]