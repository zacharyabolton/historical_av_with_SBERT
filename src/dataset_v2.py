"""
Dataset functions and classes.
"""
import random
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
from constants import MODEL
import torch
import torch.nn.functional as F


def collate_fn(batch):
    """
    Collate function to properly batch the paired inputs.
    """
    # Separate the batch components
    chunk_a = [item[0] for item in batch]
    chunk_b = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch])

    # Combine input_ids and attention_masks
    batched_a = {
        'input_ids': torch.cat([x['input_ids'] for x in chunk_a]),
        'attention_mask': torch.cat([x['attention_mask'] for x in chunk_a])
    }
    batched_b = {
        'input_ids': torch.cat([x['input_ids'] for x in chunk_b]),
        'attention_mask': torch.cat([x['attention_mask'] for x in chunk_b])
    }

    return batched_a, batched_b, labels


class CustomDataset(Dataset):
    """
    Custom torch dataset.

    :param data_dir: The directory of the data files.
    :type data_dir: string
    """

    def __init__(self, data_dir, chunk_size):
        assert (chunk_size > 2), ("Your chunk size is too small."
                                  " Please increase.")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)

        self._data_dir = data_dir
        self._chunk_size = chunk_size

        self._A_dir = os.path.join(self._data_dir, "A")
        self._U_dir = os.path.join(self._data_dir, "U")
        self._notA_dir = os.path.join(self._data_dir, "notA")

        self._A_docs = self._get_docs(self._A_dir)
        self._U_docs = self._get_docs(self._U_dir)
        self._notA_docs = self._get_docs(self._notA_dir)

        self._A_docs_tokenized = self._tokenize(self._A_docs)
        self._U_docs_tokenized = self._tokenize(self._U_docs)
        self._notA_docs_tokenized = self._tokenize(self._notA_docs)

        self._A_chunks = self._chunk_tokens(self._A_docs_tokenized)
        # self._U_chunks = self._chunk_tokens(self._U_docs_tokenized)
        # self._notA_chunks = self._chunk_tokens(self._notA_docs_tokenized)

        # self._pairs = self._create_pairs(self._A_chunks,
        #                                  self._notA_chunks)

    def __len__(self):
        """
        Returns the total number of pairs in the dataset.
        """
        # return len(self._pairs)
        pass

    def __getitem__(self, idx: int):
        """
        Get a single pair of text chunks with their label.
        """
        # return self._pairs[idx]
        pass

    def _get_docs(self, dir):
        """
        Concatenates the contents from all `.txt` files in a given directory.

        :param dir: The directory to get docs from.
        :type dir: string
        :rtype: list
        """
        docs = []
        dir_contents = os.listdir(dir)
        dir_contents_sorted = sorted(dir_contents)
        for file in dir_contents_sorted:
            if file.endswith(".txt"):
                with open(os.path.join(dir, file), 'r') as file:
                    docs.append(file.read())
        return docs

    def _tokenize(self, docs):
        """
        Tokenize input docs.

        :param docs: List of strigs representing the documents in a given
        class (A, notA, U)
        :type docs: list
        :rtype: list
        """

        tokenized_docs = []
        for doc in docs:
            # Tokenize sentences
            encoded_input = self.tokenizer(
                doc,
                return_tensors="pt",
                padding=False,
                truncation=False
            )
            tokenized_docs.append(encoded_input)

        return tokenized_docs

    def _chunk_tokens(self, encodings):
        """
        Chunk input tokens.

        Break tokenized data into 512 token chunks, padding final chunks if
        necessary.

        NOTE: This is a change from Ibrahim et al.
        Since the input size of the model is 512 we're using it.
        Ibrahim et al. used 256
        I'm assuming they used this because most of the documentation assumes
        usage as a cross-encoder (512 divided between two documents = 256
        per document) rather than Siamese (giving the full 512 for each
        instance).

        :param encodings: List of dictionaries of 'input_ids',
        'token_type_ids', and 'attention_mask'
        :type encodings: list
        :rtype: list
        """

        # Our effective chunk size is two less to make room to add [CLS]
        # and [SEP] tokens back in to our resultant chunks
        chunk_size_reduced = self._chunk_size - 2

        chunked_encodings = []

        for encoding in encodings:
            # Get num of tokens minues special BERT [CLS] and [SEP] tokens
            tokens_length = encoding['input_ids'].size()[1] - 2

            # Remove special BERT [CLS] and [SEP] tokens
            # We will add them back in to our chunks in a further step
            input_ids = torch.narrow(encoding['input_ids'],
                                     1, 1, tokens_length)
            attention_mask = torch.narrow(encoding['attention_mask'],
                                          1, 1, tokens_length)

            chunks_input_ids = input_ids.split(chunk_size_reduced, dim=1)
            chunks_attention_mask = attention_mask.split(chunk_size_reduced,
                                                         dim=1)

            # Create tensors for special tokens
            cls_token = torch.tensor([[self.tokenizer.cls_token_id]],
                                     device=input_ids.device)
            sep_token = torch.tensor([[self.tokenizer.sep_token_id]],
                                     device=input_ids.device)
            special_attention = torch.tensor([[1]], device=attention_mask.device)

            # Process each chunk to add special tokens
            processed_chunks = []

            for chunk_ids, chunk_mask in zip(chunks_input_ids,
                                             chunks_attention_mask):
                # Add CLS token at start
                chunk_ids = torch.cat([cls_token,
                                       chunk_ids,
                                       sep_token], dim=1)
                chunk_mask = torch.cat([special_attention,
                                        chunk_mask,
                                        special_attention], dim=1)

                # Pad final chunk if shorter than chunk size
                short = self._chunk_size - chunk_ids.size(1)
                if short > 0:
                    # Adapted from:
                    # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
                    chunk_ids = F.pad(chunk_ids,
                                      (0, short),
                                      "constant",
                                      0)  # effectively zero padding
                    chunk_mask = F.pad(chunk_mask,
                                       (0, short),
                                       "constant",
                                       0)  # effectively zero padding

                processed_chunks.append({
                    'input_ids': chunk_ids,
                    'attention_mask': chunk_mask
                })

            chunked_encodings.append(processed_chunks)

        return chunked_encodings

    def _create_pairs(self, A_chunks, notA_chunks):
        """
        Create training pairs as DataFrame.

        :returns: Training pairs as a Pandas DataFrame.
        :rtype: pandas.core.frame.DataFrame
        """
        # Create positive pairs (same author)
        same_auth_pairs = []
        for i in range(len(A_chunks)):
            # Note: i + 1 prevents self-pairs
            for j in range(i + 1, len(A_chunks)):
                same_auth_pairs.append((A_chunks[i], A_chunks[j], 1))

        # Create negative pairs (different authors)
        diff_auth_pairs = []
        for A_chunk in A_chunks:
            for notA_chunk in notA_chunks:
                diff_auth_pairs.append((A_chunk, notA_chunk, 0))

        # Balance positive and negative samples
        min_samples = min(len(same_auth_pairs), len(diff_auth_pairs))
        same_auth_pairs = random.sample(same_auth_pairs, min_samples)
        diff_auth_pairs = random.sample(diff_auth_pairs, min_samples)

        # Combine
        all_pairs = same_auth_pairs + diff_auth_pairs

        return all_pairs