"""
Siamese SBERT architecture inspired by Ibrahim et al. (2023) [7]:
[7] Momen Ibrahim, Ahmed Akram, Mohammed Radwan, Rana Ayman, Mustafa
Abd-El-Hameed, Nagwa M. El-Makky, and Marwan Torki. 2023. Enhancing
Authorship Verification using Sentence-Transformers. In _Conference and
Labs of the Evaluation Forum_. Retrieved
from https://api.semanticscholar.org/CorpusID:264441704

The model architecture at training time is as follows:
0. The `SiameseSBERT` class instantiates the parameterized pre-trained
   model, 'all-MiniLM-L12-v2' SBERT Model [^1] in this study, as a data
   member, `self.encoder`.
1. > Batch `anchor` and Batch `other` of tokenized, chunked, text of
        same-author or different-author pairs are ingested into the
        forward pass ---------------------------------------------------->
2. -> Batches are fed through SBERT in isolation from one another ------->
3. -> Resultant SBERT embeddings for `anchor` & `other` are fed through
        mean pooling layers, in isolation, which convert multiple word
        embeddings representing the whole sentence into a single embedding
        by averaging the embeddings for all tokens in the sentence
        embedding. ------------------------------------------------------>
4. -> The pooled embeddings for `anchor` & `other` are fed to a
        contrastive loss function, which calculates the loss as the
        squared difference between embeddings when the label = 1
        (similar pairs). When label = 0 (dissimilar pairs), the loss grows
        quadratically as the distance decreases and is clamped at zero
        above the value of the `margin` input. -------------------------->
5. -> Backpropagation adjusts the model's weights per the loss function.

The model architecture at inference time is as follows:
0-3. > Steps 0 through 3 are the same as those for training ------------->
4. -> The cosine-similarity for the pooled embeddings for `anchor` &
        `other` is calculated to produce a similarity score - intended to
        be evaluated against a decision threshold (to be implemented) to
        determine the label, 0, or 1.

[^1] https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
"""
import torch
import torch.nn as nn
from transformers import AutoModel
# import torch.nn.functional as F


class SiameseSBERT(nn.Module):
    """
    A Siamese SBERT model based on the pre-trained all-MiniLM-L12-v2 model
    from: https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2.

    Inherits from `nn.Module`, the base class for all neural networks from
    PyTorch.
    """

    def __init__(self, model_name, device):
        """
        Initializes a Siamese SBERT model based on a pre-trained model
        from the `transformers` library. In this case, this pre-trained
        model is `sentence-transformers/all-MiniLM-L12-v2`.

        :param model_name: <Required> The name of the pre-trained model
        from the `transformers` library.
        :type model_name: string

        :param device: <Required> The device to run tensor operations on
        for parallelization/concurrency utlization: 'cuda', 'mps', 'cpu'.
        :type device: str
        """

        super(SiameseSBERT, self).__init__()
        self._device = device
        self.encoder = AutoModel.from_pretrained(model_name)
        if device == 'cuda' or device == 'mps':
           self.encoder.config.use_cache = True

    def forward(self,
                input_ids_anchor,
                attention_mask_anchor,
                input_ids_other,
                attention_mask_other):
        """
        The forward step feeds the data through the network.
        This feeds the data through the SBERT pre-trained model, followed
        by a mean pooling layer, which condenses sentences' token
        embeddings into a single embedding representing the entire
        sentence by averaging all token-based embeddings.

        :param input_ids_anchor, input_ids_other: <Required> Tensors of
        IDs indexing tokens in the pre-trained model's token vocabulary.
        :type input_ids_anchor, input_ids_other: torch.Tensor

        :param attention_mask_anchor, attention_mask_other: <Required>
        Tensors of binary values (0 or 1) that act as masks. They mask
        (zero out) embeddings representing padding tokens if the input
        tokenized sentences to the BERT model were padded to reach a
        uniform length.
        :type attention_mask_anchor, attention_mask_other: torch.Tensor

        :returns (embeddings_anchor, embeddings_other):
        """

        # Encode first input
        outputs_anchor = self.encoder(
            input_ids_anchor, attention_mask=attention_mask_anchor)
        embeddings_anchor = self.mean_pool(
            outputs_anchor.last_hidden_state, attention_mask_anchor)

        # Encode second input
        outputs_other = self.encoder(
            input_ids_other, attention_mask=attention_mask_other)
        embeddings_other = self.mean_pool(
            outputs_other.last_hidden_state, attention_mask_other)

        return embeddings_anchor, embeddings_other

    def mean_pool(self, token_embeddings, attention_mask):
        """
        The mean pooling layer which averages the token embeddings for the
        full sentence into one embedding. This does not include padding
        tokens.

        Adapted from...
        
        [31] Lewis Tunstall, Leandro Von Werra, and Thomas Wolf. 2022.
        Natural language processing with transformers. “ O’Reilly Media,
        Inc.” Retrieved from
        https://books.google.com.br/books?id=nzxbEAAAQBAJ&lpg=PP1&ots=sUqiDWs3MI&dq=Natural%20Language%20Processing%20with%20Transformers%20%5BRevised%20Edition  # noqa E501

        ...and...

        [36] Nils Reimers. 2021. Sentence-transformers/all-minilm-L12-V2.
        hugging face, sentence-transformers/all-MiniLM-L12-v2. Hugging
        Face. Available at:
        https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2.
        Accessed: December 04, 2024.


        :param token_embeddings: A tensor of embeddings representing the
        tokens of the tokenized input sentence.
        :type token_embeddings: torch.Tensor
        :param attention_mask: A binary (0 or 1) mask representing which
        tokens to attend to - used to mask padding tokens so they do not
        affect the final pooler embedding.
        :type attention_mask: torch.Tensor
        """

        # Calculate the mean of token embeddings, ignoring padding tokens
        # Expand the 1D input mask to fit the token_embeddings of dim
        # `tokens` * `token length`.
        # Adapted from Tunstall et al. (2022) [31:276]
        input_mask_expanded = attention_mask.unsqueeze(-1).\
            expand(token_embeddings.size()).float()

        # Sum the non-padding tokens
        sum_embeddings = torch.sum(token_embeddings *
                                   input_mask_expanded, dim=1)

        # Get the number of non-padding tokens
        # counting tiny floats (lt 1e-9) as zero
        # due to float imprecision
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        # Calculate the mean
        mean_embeddings = sum_embeddings / sum_mask

        return mean_embeddings