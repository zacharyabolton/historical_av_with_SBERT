"""
This file was adapted from the Jacob Tyo's distribution of VALLA in their
VALLA repo on GitHub:
https://github.com/JacobTyo/Valla/blob/main/valla/methods/SiameseBert.py

[60] Jacob Tyo. 2022. SiameseBert.py (Version f5022b8). Valla repository
on GitHub (October 19, 2022). Retrieved from
https://github.com/JacobTyo/Valla/blob/f5022b8f90d909d530ddb205f6e35228e6f35cda/valla/methods/SiameseBert.py  # noqa: E501
Accessed: February 6, 2025.
"""
# import torch
from torch.utils.data import Dataset
from loaders import get_av_dataset, get_aa_dataset
from sentence_transformers.readers import InputExample
from dataset_utils import list_dset_to_dict
import random


def get_random_substring(txt, substr_len=512*5):
    if len(txt) > substr_len + 1:
        idx = random.randint(0, len(txt) - substr_len + 1)
        txt = txt[idx:idx+substr_len]
    return txt


class VALLAAVTrainDataset(Dataset):
    def __init__(self, data_path):
        super(VALLAAVTrainDataset, self).__init__()
        self.data = list_dset_to_dict(get_aa_dataset(data_path))
        self.data_len = sum([len(x) for x in self.data.values()])
        # build a map to uniquely identify texts
        self.idx_to_txt_map = {}
        i = 0
        for auth, texts in self.data.items():
            for text_loc, _ in enumerate(texts):
                self.idx_to_txt_map[i] = {
                    'auth_id': auth,
                    'text_id': text_loc
                }
                i += 1
        self.author_list = list(self.data.keys())

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        auth_id = self.idx_to_txt_map[item]['auth_id']
        txt_num = self.idx_to_txt_map[item]['text_id']
        text0 = self.data[auth_id][txt_num]
        if random.random() < 0.5:
            # different author sample
            label = 0
            auth2 = random.choice(self.author_list)
            while auth2 == auth_id:
                auth2 = random.choice(self.author_list)
            text1 = random.choice(self.data[auth2])
        else:
            # same author sample
            label = 1
            text1 = random.choice(self.data[auth_id])
        # now pick a random ~512 words from each text to send
        text0 = get_random_substring(text0)
        text1 = get_random_substring(text1)

        return InputExample(texts=[text0, text1], label=label)

    def listify_text(self, txt, chunk_len=512, max_txt_len=100000):
        txt = txt[:max_txt_len]
        chunked = []
        for i in range(0, len(txt), chunk_len):
            chunked.append(txt[i:i + chunk_len])
        return chunked


class VALLAAVValDataset(Dataset):

    def __init__(self, data_path):
        # , char_vocab=None, tok_vocab=None, char_to_id=None,
        # tok_to_id=None, **kwargs):
        super(VALLAAVValDataset, self).__init__()  # char_vocab,
        # tok_vocab, char_to_id, tok_to_id, **kwargs)

        _data = get_av_dataset(data_path)
        self.data = []
        self.raw_data = []
        for label, text0, text1 in _data:
            self.data.append(InputExample(label=label,
                                          texts=[text0, text1]))
            self.raw_data.append([label, text0, text1])
        self.data_len = len(self.data)

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        return self.data[item]

    def get_chunked(self, item):
        # chunk each evaluation text so we can evaluate many pairs per
        # document pair
        chunk_len = 256 * 5
        if isinstance(item, slice):

            samples = self.raw_data[item]
            chunked_samples = []
            for label, text0, text1 in samples:
                chunked_samples.append(
                    self.break_sample_into_chunks(label, text0, text1,
                                                  chunk_len))
            return chunked_samples
        else:
            label, text0, text1 = self.raw_data[item]
            return self.break_sample_into_chunks(label, text0, text1,
                                                 chunk_len)

    @staticmethod
    def break_sample_into_chunks(lbl, txt0, txt1, chunk_len,
                                 max_txt_len=100000):
        txt0 = txt0[:max_txt_len]
        txt1 = txt1[:max_txt_len]
        chunked = []
        min_len = min(len(txt0), len(txt1))
        for i in range(0, min_len, chunk_len):
            chunked.append(
                InputExample(label=lbl,
                             texts=[txt0[i:i + chunk_len],
                                    txt1[i:i + chunk_len]]))
        return chunked