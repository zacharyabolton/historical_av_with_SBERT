"""
Dataset functions and classes.

Adapted for this project from the PyTorch (https://pytorch.org/)
documentation at:
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files  # noqa: E501
"""
import random
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
from constants import MODEL
import torch
import pandas as pd
import math
import copy
# import torch.nn.functional as F  # Only if preserving ending chunks


def collate_fn(batch):
    """
    Collate function to properly batch the paired inputs.
    """
    # Separate the batch components
    cnk_a = [item[0] for item in batch]
    cnk_b = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch])

    # Combine input_ids and attention_masks
    batched_a = {
        'input_ids': torch.cat([x['input_ids'] for x in cnk_a]),
        'attention_mask': torch.cat([x['attention_mask']
                                     for x in cnk_a])
    }
    batched_b = {
        'input_ids': torch.cat([x['input_ids'] for x in cnk_b]),
        'attention_mask': torch.cat([x['attention_mask']
                                     for x in cnk_b])
    }

    return batched_a, batched_b, labels


class LILADataset(Dataset):
    """
    Custom torch dataset.

    :param data_dir: The directory of the data files.
    :type data_dir: string
    """

    def __init__(self,
                 data_dir,
                 metadata_path,
                 cnk_size,
                 num_pairs,
                 device,
                 seed=None):
        """
        Constructor for setting up data members needed for the
        LILADataset.

        :param data_dir: <Required> The path to the data preprocessed data
        for ingestion. Expects a directory with subdirectories `A`,
        `notA`, and `U`, each containing `.txt` files.
        :type data_dir: str

        :param metadata_path: <Required> The path to the metadata file.
        Expects a path to a `.csv` file with columns ['file',
        'author_short', 'author', 'genre', 'imposter_for',
        'canonical_class_label', 'class', 'omit', 'num_words'] and rows
        containing unique 'file' values matching at least every `.txt`
        file in the subdirectories, and 'canonical_class_labels' matching
        that file's parent directory name: `A`, `notA`, or `U`.
        :type metadata_path: str

        :param cnk_size: <Required> Maximum length, in tokens, of each
        chunk generated, including the special BERT [CLS] and [SEP]
        tokens.
        Currently all chunks have this length as ending chunks that could
        be padded are being thrown out.
        :type cnk_size: int

        :param num_pairs: <Required> The total number of same-author and
        different-author pairs, combined, to generate. Must be an even
        number for data-balancing purposes.
        :type num_pairs: int

        :param device: <Required> The device to run tensor operations on.
        For use in paralelization/cuncurrency to speed up training.
        :type device: str

        :param seed: <Optional> An integer to pass in to the `random`
        module as a seed, for reproducibility. Defaults to `None` which
        causes the `random` module to use the system clock as a seed.
        :type seed: int
        """

        assert (cnk_size > 2), ("Your chunk size is too small."
                                " Please increase.")
        assert (num_pairs % 2) == 0, ("Please use an even number of"
                                      " pairs for data balancing.")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)

        self._data_dir = data_dir
        self._metadata_path = metadata_path
        self._cnk_size = cnk_size
        self._num_pairs = num_pairs
        self._device = device
        self._seed = seed

        self._metadata = pd.read_csv(self._metadata_path)
        # Filter out 'omitted' works
        omitted_mask = self._metadata['omit'] == False  # noqa: E712
        self._metadata = self._metadata[omitted_mask]
        self._metadata.reset_index(inplace=True)

        self._A_dir = os.path.join(self._data_dir, "A")
        self._U_dir = os.path.join(self._data_dir, "U")
        self._notA_dir = os.path.join(self._data_dir, "notA")

        self._A_docs = self._get_docs(self._A_dir)
        self._U_docs = self._get_docs(self._U_dir)
        self._notA_docs = self._get_docs(self._notA_dir)

        self._A_docs_tokenized = self._tokenize(self._A_docs)
        self._U_docs_tokenized = self._tokenize(self._U_docs)
        self._notA_docs_tokenized = self._tokenize(self._notA_docs)

        self._A_docs_cnked = self._cnk_tokens(self._A_docs_tokenized)
        self._U_docs_cnked = self._cnk_tokens(self._U_docs_tokenized)
        self._notA_docs_cnked = self._cnk_tokens(
            self._notA_docs_tokenized)

        self._pairs = self._create_pairs(self._A_docs_cnked,
                                         self._notA_docs_cnked)

    def __len__(self):
        """
        Returns the total number of pairs in the dataset.
        """
        return len(self._pairs)

    def __getitem__(self, idx: int):
        """
        Get a single pair of text chunks with their label.
        """
        return self._pairs[idx]

    def _get_docs(self, dir):
        """
        Get contents from all `.txt` files in a given directory.
        Preserves document boundaries.

        :param dir: <Required> The directory to get docs from.
        :type dir: string

        :returns: list of tuples, where the first tuple element is the
        associated index for the given file in the `self._metadata`
        member, and the second element is the raw contents of the file.
        :rtype: list
        """
        docs = []
        dir_contents = os.listdir(dir)
        dir_contents_sorted = sorted(dir_contents)
        for file in dir_contents_sorted:
            if file.endswith(".txt"):
                if file in self._metadata['file'].unique():
                    file_meta = self._metadata[self._metadata['file'] == file]
                    with open(os.path.join(dir, file), 'r') as file:
                        contents = file.read()
                        idx = int(file_meta.index[0])
                        docs.append((idx, contents))
        return docs

    def _tokenize(self, docs):
        """
        Tokenize input docs.
        Preserves document boundaries.

        :param docs: <Required> List of tuples representing the documents
        in a given class (A, notA, U), where the first tuple element is
        the associated index for the given file in the `self._metadata`
        member, and the second element is the raw contents of the file.
        :type docs: list

        :returns: list of tuples, where the first tuple element is the
        associated index for the given file in the `self._metadata`
        member, and the second element is a PyTorch embedding of the
        tokenized contents.
        :rtype: list
        """

        tokenized_docs = []
        for doc in docs:
            # Tokenize sentences
            encoded_input = self.tokenizer(
                doc[1],
                return_tensors="pt",
                padding=False,
                truncation=False
            )
            tokenized_docs.append((doc[0], encoded_input))

        return tokenized_docs

    def _cnk_tokens(self, tokenized_docs):
        """
        Chunk tokenized documents.

        Break tokenized data into chunks with lengths
        `self._cnk_length` - 2, before adding in special BERT [CLS] and
        [SEP] tokens bringing the lenght to `self._cnk_length`.
        Currently this method throws out final chunks if they are less
        than `self._cnk_length` after adding special BERT [CLS] and
        [SEP] tokens.

        :param tokenized_docs: <Required> List of tuples, where the first
        tuple element is the associated index for the given file in the
        `self._metadata` member, and the second element is a PyTorch
        embedding of the tokenized contents.
        :type tokenized_docs: list

        :returns: A list of tuples, where the first tuple element is the
        associated index for the given file in the `self._metadata`
        member, and the second element is a list of PyTorch embeddings of
        the chunks generated from that file.
        :rtype: list
        """

        # Our effective chunk size is two less to make room to add [CLS]
        # and [SEP] tokens back in to our resultant chunks
        cnk_size_reduced = self._cnk_size - 2

        cnked_encodings = []

        for i, embedding in tokenized_docs:
            # Get num of tokens minues special BERT [CLS] and [SEP] tokens
            tokens_length = embedding['input_ids'].size()[1] - 2

            # Remove special BERT [CLS] and [SEP] tokens
            # We will add them back in to our chunks in a further step
            input_ids = torch.narrow(embedding['input_ids'],
                                     1, 1, tokens_length)
            attention_mask = torch.narrow(embedding['attention_mask'],
                                          1, 1, tokens_length)

            # Create bare chunks by slicing the tensor in strides of size
            # cnk_size_reduced
            cnks_input_ids = input_ids.split(cnk_size_reduced, dim=1)
            cnks_attention_mask = attention_mask.split(cnk_size_reduced,
                                                       dim=1)

            # Create tensors for special tokens
            cls_token = torch.tensor([[self.tokenizer.cls_token_id]],
                                     device=input_ids.device)
            sep_token = torch.tensor([[self.tokenizer.sep_token_id]],
                                     device=input_ids.device)
            special_attention = torch.tensor([[1]],
                                             device=attention_mask.device)

            # Process each chunk to add special tokens
            processed_cnks = []

            for cnk_ids, cnk_mask in zip(cnks_input_ids,
                                         cnks_attention_mask):
                # Add CLS token at start
                cnk_ids = torch.cat([cls_token,
                                     cnk_ids,
                                     sep_token],
                                    dim=1).to(self._device)
                cnk_mask = torch.cat([special_attention,
                                      cnk_mask,
                                      special_attention],
                                     dim=1).to(self._device)

                # Pad final chunk if shorter than chunk size
                short = self._cnk_size - cnk_ids.size(1)
                if short > 0:
                    # # Adapted from:
                    # # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html  # noqa: E501
                    # cnk_ids = F.pad(cnk_ids,
                    #                   (0, short),
                    #                   "constant",
                    #                   0)  # effectively zero padding
                    # cnk_mask = F.pad(cnk_mask,
                    #                    (0, short),
                    #                    "constant",
                    #                    0)  # effectively zero padding
                    # Throw away for now
                    continue

                processed_cnks.append({
                    'input_ids': cnk_ids,
                    'attention_mask': cnk_mask
                })

            cnked_encodings.append((i, processed_cnks))

        return cnked_encodings

    def _create_pairs(self, A_docs_cnked, notA_docs_cnked):
        """
        Create training pairs.
        Balance training pairs to 50% same-author and 50% different author
        pairs.
        Within the same-author split, balance the distribution of chunks
        per genre to be equal to the same genre distribution found in the
        source dataset `A`.
        Within the different-author split, balance so that there is equal
        number of pairs for each imposter type ('LaSalle Imposters',
        'George Imposters').
        Within each imposter type subsplit, balance the distribution of
        chunks per genre, per imposter type, to to be equal to the same
        genre distribution found in the source dataset, `A`.
        _NOTE_: The returned list does not shuffle based on class
        (same/different-author), genre, or imposter type, placing class 1
        at the start followed by class 0, and genres/imposter types
        partitioned in the order they were encountered. It _does_ however,
        shuffle the chunks within each genre or imposter-genre segment to
        randomize document representation.

        :param A_docs_cnked: <Required> A list of tuples, where the first
        tuple element is the associated index for a given `A` (LaSalle)
        file in the `self._metadata` member, and the second element is a
        list of PyTorch embeddings of the chunks generated from that file.
        :type A_docs_cnked: list

        :param notA_docs_cnked: <Required> A list of tuples, where the
        first tuple element is the associated index for a given `notA`
        (Imposter) file in the `self._metadata` member, and the second
        element is a list of PyTorch embeddings of the chunks generated
        from that file.
        :type notA_docs_cnked: list

        :returns: A list of 3-tuples, where the first and second elements
        are PyTorch embeddings of chunks of either `A` or `notA` docs
        (both `A` in the case of same-author pais, one `A` and one `notA`
        in the case of different-author pairs). The third tuple element is
        a class label of 1 (for same-author pairs) or 0 (for
        different-author pairs). The first segment of the list, up to
        `n_same_pairs` (roughly half, depending on rounding imprecision
        - see below) are all of class 1, and the remaining segement are
        all of class 0.
        :rtype: list
        """
        # Save metadata locally for shorter code lines and readability
        meta_data = self._metadata

        # Get genres in A only, as A is what we want to balance on
        A_mask = meta_data['canonical_class_label'] == 'A'
        genres = meta_data[A_mask]['genre'].unique()

        # Create a genre holder
        genre_dict = {k: [] for k in genres}

        # Create dict to hold A chunks by genre
        A_cnks_by_genre = copy.deepcopy(genre_dict)

        # Get imposters by type
        notA_mask = meta_data['canonical_class_label'] == 'notA'
        imposter_types = meta_data[notA_mask]['imposter_for'].unique()

        # Create dict to hold notA chunks by genre by imposter type
        notA_cnks_nested = {k: copy.deepcopy(genre_dict)
                            for k in imposter_types}

        # Fill dicts with appropriate chunks
        for (doc_idx, cnks) in A_docs_cnked:
            doc_genre = meta_data.loc[doc_idx]['genre']
            A_cnks_by_genre[doc_genre].extend(cnks)

        for (doc_idx, cnks) in notA_docs_cnked:
            doc_imposter_type = meta_data.loc[doc_idx]['imposter_for']
            doc_genre = meta_data.loc[doc_idx]['genre']
            notA_cnks_nested[doc_imposter_type][doc_genre].extend(cnks)

        # Determine the genre ratios
        n_A_cnks = sum([len(cnks) for cnks in A_cnks_by_genre.values()])
        genre_ratios = {k: len(A_cnks_by_genre[k])/n_A_cnks
                        for k in A_cnks_by_genre}

        # Calculate the balanced number of same/diff pairs to generate
        # When unable to balance perfectly we give the difference to the
        # different-author pairs
        n_imp_types = len(notA_cnks_nested.keys())
        n_diff_pairs_by_imp_type = math.ceil((self._num_pairs/2) /
                                             n_imp_types)
        n_diff_pairs_total = n_diff_pairs_by_imp_type * n_imp_types
        n_same_pairs = self._num_pairs - n_diff_pairs_total

        # Create positive pairs (same author)
        same_auth_pairs = []
        # Loop through genres
        for genre in genre_ratios:
            # Get the chunks for this genre
            cnks = A_cnks_by_genre[genre]
            # Determine how many pairs are needed in this given genre
            p_need = math.ceil(n_same_pairs * genre_ratios[genre])
            c_have = len(cnks)
            # Make sure we have enough
            # Since we're creating same pairs we need enough chunks to
            # allow us to create `p_need` pairs by picking 2 from `cnks`
            # without replacement.
            # To check that we have sufficient chunks we need to solve
            # p_need = (c_need * (c_need - 1))) / 2
            # for `c_need`
            c_need = math.ceil((1 + math.sqrt(1 + 8 * p_need)) / 2)
            assert (c_have >= c_need), ("The requested number of same"
                                        " pairs requires more chunks than"
                                        " can be generated from this"
                                        f" population: {c_have} < "
                                        f"{c_need}")
            # First, create all possible positive pairs (same author)
            generated_pairs = []
            for i in range(len(cnks)):
                # Note: i + 1 prevents self-pairs
                for j in range(i + 1, len(cnks)):
                    generated_pairs.append((cnks[i], cnks[j], 1))
            # Randomly sample from all possible pairs without replacement.
            # Adapted from:
            # https://stackoverflow.com/a/6494519
            same_auth_pairs.extend(random.Random(self._seed).sample(
                generated_pairs, p_need))

        # if rounding errors resulted in more pairs than needed, remove
        # the excess randomly
        same_auth_pairs = random.Random(self._seed).sample(
            same_auth_pairs, n_same_pairs)

        # Create negative pairs (different author)
        # preserving equal number of pairs between imposter types
        # and genre balance within those
        diff_auth_pairs = []
        # Loop through the imposter types
        for imposter in notA_cnks_nested:
            # Loop through genres
            for genre in notA_cnks_nested[imposter]:
                # Get the notA chunks for this genre and imposter type
                notA_cnks = notA_cnks_nested[imposter][genre]
                # Get the A chunks for this genre
                A_cnks = A_cnks_by_genre[genre]
                # Determine how many pairs are needed in this given
                # genre per imposter split
                p_need = round(n_diff_pairs_by_imp_type *
                               genre_ratios[genre])
                # Make sure we have enough
                # Since we're creating diff pairs we need at least the
                # square root of the number of needed pairs of needed
                # chunks to generate sufficient pairs without duplication.
                c_notA_need = math.ceil(math.sqrt(p_need))
                c_A_need = c_notA_need
                c_notA_have = len(notA_cnks)
                c_A_have = len(A_cnks)
                assert (c_A_have >= c_A_need), ("The requested number of"
                                                " diff pairs requires"
                                                " more A chunks than can"
                                                " be generated from this"
                                                " population:"
                                                f" {c_A_have} <"
                                                f" {c_A_need}")
                assert (c_notA_have >= c_notA_need), ("The requested"
                                                      " number of diff"
                                                      " pairs requires"
                                                      " more notA chunks"
                                                      " than can be"
                                                      " generated from"
                                                      " this population:"
                                                      f" {c_notA_have} < "
                                                      f"{c_notA_need}")
                # Create all possible negative pairs (different authors)
                generated_pairs = []
                for A_cnk in A_cnks:
                    for notA_cnk in notA_cnks:
                        generated_pairs.append((A_cnk, notA_cnk, 0))
                # Randomly sample from all possible pairs without
                # replacement.
                # Adapted from:
                # https://stackoverflow.com/a/6494519
                diff_auth_pairs.extend(random.Random(
                    self._seed).sample(generated_pairs,
                                       p_need))

        # if rounding errors resulted in more pairs than needed, remove
        # the excess randomly
        diff_auth_pairs = random.Random(self._seed).sample(
            diff_auth_pairs, n_diff_pairs_total)

        # Combine
        all_pairs = same_auth_pairs + diff_auth_pairs

        return all_pairs