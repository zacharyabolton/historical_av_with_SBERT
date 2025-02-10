"""
This is a dataloader collate function for use with VALLA and PyTorch
Dataloaders. It takes batches as lists of InputExample from
`sentence-transformers` and transforms them into batches of token
embeddings for ingestion into SiameseSBERT sublcassling
`torch.nn.Module`.
"""
from transformers import AutoTokenizer
import torch
import os
import sys

# Add src directory to sys.path
# Adapted from Taras Alenin's answer on StackOverflow at:
# https://stackoverflow.com/a/55623567
src_path = os.path.join('..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import custom modules
from constants import MODEL


def valla_collate_fn(batch):
    """
    Collate function to properly batch the paired inputs.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    labels = torch.tensor([item.label for item in batch])
    anchor_texts = [tokenizer(item.texts[0],
                              return_tensors="pt",
                              padding='max_length',
                              truncation=True,
                              max_length=512) for item in batch]
    other_texts = [tokenizer(item.texts[1],
                             return_tensors="pt",
                             padding='max_length',
                             truncation=True,
                             max_length=512) for item in batch]

    # Combine input_ids and attention_masks
    batched_a = {
        'input_ids': torch.cat([x['input_ids'] for x in anchor_texts]),
        'attention_mask': torch.cat([x['attention_mask']
                                     for x in anchor_texts])
    }
    batched_o = {
        'input_ids': torch.cat([x['input_ids'] for x in other_texts]),
        'attention_mask': torch.cat([x['attention_mask']
                                     for x in other_texts])
    }

    return batched_a, batched_o, labels
