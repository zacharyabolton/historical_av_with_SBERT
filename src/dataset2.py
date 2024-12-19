from torch.utils.data import Dataset
import torch
from typing import List, Dict, Tuple, Optional
import random

class AuthorshipVerificationDataset(Dataset):
    """
    Custom PyTorch Dataset for Authorship Verification tasks.
    Handles text chunking and pairs generation for Siamese BERT architecture.
    """

    def __init__(
        self,
        data_director: str,
        chunk_size: int = 256,
    ):
        """
        Initialize the dataset.

        Args:
            texts_by_author: Dictionary mapping author IDs to lists of their texts
            chunk_size: Maximum size of each text chunk (including special tokens)
            pairs_per_text: Number of pairs to generate per text
            max_chunks_per_text: Maximum number of chunks to use per text
        """
        self.data_directory = data_directory
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.chunk_size = chunk_size

        # Pre-process texts and generate pairs
        self.text_pairs = self._generate_pairs()

    def __len__(self) -> int:
        """
        Returns the total number of pairs in the dataset.
        Each text generates pairs_per_text positive and negative pairs.
        """
        return len(self.text_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single pair of text chunks with their label.

        Args:
            idx: Index of the pair to retrieve

        Returns:
            Dictionary containing:
                - text_a_input_ids: Input IDs for first text
                - text_a_attention_mask: Attention mask for first text
                - text_b_input_ids: Input IDs for second text
                - text_b_attention_mask: Attention mask for second text
                - label: Binary label (1 for same author, 0 for different)
        """
        pair = self.text_pairs[idx]
        text_a_chunks = self._tokenize_and_chunk(pair['text_a'])
        text_b_chunks = self._tokenize_and_chunk(pair['text_b'])

        # Randomly select one chunk from each text
        chunk_a = random.choice(text_a_chunks)
        chunk_b = random.choice(text_b_chunks)

        return {
            'text_a_input_ids': chunk_a['input_ids'],
            'text_a_attention_mask': chunk_a['attention_mask'],
            'text_b_input_ids': chunk_b['input_ids'],
            'text_b_attention_mask': chunk_b['attention_mask'],
            'label': torch.tensor(pair['label'], dtype=torch.long)
        }

    def _tokenize_and_chunk(self, text: str) -> List[Dict[str, torch.Tensor]]:
        """
        Tokenize text and split into chunks of specified size.

        Args:
            text: Input text to tokenize and chunk

        Returns:
            List of dictionaries containing input_ids and attention_mask for each chunk
        """
        # Tokenize the full text
        encoded = self.tokenizer(
            text,
            truncation=False,
            padding=False,
            return_tensors='pt'
        )

        # Account for special tokens in chunk size
        chunk_size_reduced = self.chunk_size - 2  # For [CLS] and [SEP]

        # Split into chunks
        input_ids = encoded['input_ids'][0]
        attention_mask = encoded['attention_mask'][0]

        chunks_input_ids = input_ids.split(chunk_size_reduced)
        chunks_attention_masks = attention_mask.split(chunk_size_reduced)

        # Process chunks
        processed_chunks = []
        for chunk_ids, chunk_mask in zip(chunks_input_ids[:8],
                                         chunks_attention_masks[:8]):
            # Add special tokens
            chunk_ids = torch.cat([
                torch.tensor([self.tokenizer.cls_token_id]),
                chunk_ids,
                torch.tensor([self.tokenizer.sep_token_id])
            ])

            chunk_mask = torch.cat([
                torch.tensor([1]),
                chunk_mask,
                torch.tensor([1])
            ])

            # Pad if necessary
            if chunk_ids.size(0) < self.chunk_size:
                padding_length = self.chunk_size - chunk_ids.size(0)
                chunk_ids = torch.cat([
                    chunk_ids,
                    torch.zeros(padding_length, dtype=torch.long)
                ])
                chunk_mask = torch.cat([
                    chunk_mask,
                    torch.zeros(padding_length, dtype=torch.long)
                ])

            processed_chunks.append({
                'input_ids': chunk_ids,
                'attention_mask': chunk_mask
            })

            if (self.max_chunks_per_text is not None and 
                len(processed_chunks) >= self.max_chunks_per_text):
                break

        return processed_chunks

    def _generate_pairs(self) -> List[Dict[str, str]]:
        """
        Generate positive and negative pairs for training.

        Returns:
            List of dictionaries containing text pairs and their labels
        """
        pairs = []
        authors = list(self.texts_by_author.keys())

        # Generate pairs for each author
        for author_id in authors:
            author_texts = self.texts_by_author[author_id]

            # Generate positive pairs (same author)
            for i, text_a in enumerate(author_texts):
                for _ in range(self.pairs_per_text):
                    # Randomly select another text from the same author
                    text_b = random.choice(author_texts[:i] + author_texts[i+1:])
                    pairs.append({
                        'text_a': text_a,
                        'text_b': text_b,
                        'label': 1
                    })

            # Generate negative pairs (different authors)
            other_authors = [a for a in authors if a != author_id]
            for text_a in author_texts:
                for _ in range(self.pairs_per_text):
                    # Randomly select text from different author
                    other_author = random.choice(other_authors)
                    text_b = random.choice(self.texts_by_author[other_author])
                    pairs.append({
                        'text_a': text_a,
                        'text_b': text_b,
                        'label': 0
                    })

        return pairs