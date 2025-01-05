"""
Dataset functions and classes.
"""
import random
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
from constants import ROOT_DIR, MODEL
import torch


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

    def __init__(self, data_dir, evaluate=False):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.evaluate = evaluate

        project_root = ROOT_DIR
        self._data_dir = os.path.join(project_root, data_dir)

        self._A_dir = os.path.join(self._data_dir, "A")
        self._notA_dir = os.path.join(self._data_dir, "notA")
        self._U_dir = os.path.join(self._data_dir, "U")

        self._A_text = self._get_text(self._A_dir)
        self._notA_text = self._get_text(self._notA_dir)
        self._U_text = self._get_text(self._U_dir)

        self._A_tokens = self._tokenize(self._A_text)
        self._notA_tokens = self._tokenize(self._notA_text)
        self._U_tokens = self._tokenize(self._U_text)

        self._A_chunks = self._chunk_tokens(self._A_tokens,
                                            self.evaluate)
        self._notA_chunks = self._chunk_tokens(self._notA_tokens,
                                               self.evaluate)
        self._U_chunks = self._chunk_tokens(self._U_tokens,
                                            self.evaluate)

        self._pairs = self._create_pairs(self._A_chunks,
                                         self._notA_chunks)

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

    def _get_text(self, dir):
        """
        Concatenates the contents from all `.txt` files in a given directory.

        :param dir: The directory to get text from.
        :type dir: string
        :rtype: string
        """
        text = ""
        dir_contents = os.listdir(dir)
        dir_contents_sorted = sorted(dir_contents)
        for file in dir_contents_sorted:
            if file.endswith(".txt"):
                with open(os.path.join(dir, file), 'r') as file:
                    text += file.read() + " "
        return text[0:-1]

    def _tokenize(self, text):
        """
        Tokenize input text.

        :param text:
        :type text: string
        :rtype: transformers.tokenization_utils_base.BatchEncoding
        """

        # Tokenize sentences
        encoded_input = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=False
        )

        return encoded_input

    def _chunk_tokens(self, encoding, evaluate):
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

        :param encoding: Dictionary of 'input_ids', 'token_type_ids', and
        'attention_mask'
        :type encoding: transformers.tokenization_utils_base.BatchEncoding
        :rtype List[transformers.tokenization_utils_base.BatchEncoding]
        """

        # Hard code chunk size to maximum all-MiniLM-L12-v2 input size.
        chunk_size = 512

        # Make sure the input encoding is only for one document
        # as we will implement chunking downstream
        assert len(encoding['input_ids']) == 1

        # Get num of tokens minues special BERT [CLS] and [SEP] tokens
        tokens_length = encoding['input_ids'].size()[1] - 2

        # Remove special BERT [CLS] and [SEP] tokens
        # We will add them back in to our chunks in a further step
        input_ids = torch.narrow(encoding['input_ids'], 1,
                                 1, tokens_length)
        attention_mask = torch.narrow(encoding['attention_mask'], 1,
                                      1, tokens_length)

        # Our effective chunk size is two less to make room to add [CLS] and
        # [SEP] tokens back in to our resultant chunks
        chunk_size_reduced = chunk_size - 2

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

        # TODO fix this hacky prototype train/eval splitting logic
        # For the prototype we are simply taking the first 64 chunks for
        # training and the last 64 for evaluation.
        print(len(chunks_input_ids))
        print(len(chunks_input_ids[0]))
        assert len(chunks_input_ids) >= 64 * 2

        if evaluate:
            start_idx = 0
            end_idx = 64
        else:
            start_idx = len(chunks_input_ids) - 64
            end_idx = len(chunks_input_ids)

        for chunk_ids, chunk_mask in zip(chunks_input_ids[start_idx:end_idx],
                                         chunks_attention_mask[start_idx:end_idx]):
            # Add CLS token at start
            chunk_ids = torch.cat([cls_token,
                                   chunk_ids,
                                   sep_token], dim=1)
            chunk_mask = torch.cat([special_attention,
                                    chunk_mask,
                                    special_attention], dim=1)

            # Throw away final chunk if shorter than chunk size
            if chunk_ids.size(1) < chunk_size:
                continue

            processed_chunks.append({
                'input_ids': chunk_ids,
                'attention_mask': chunk_mask
            })

        return processed_chunks

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