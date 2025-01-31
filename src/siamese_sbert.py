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
import torch.nn.functional as F


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


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss function

    Inherits from `nn.Module`, the base class for all neural networks from
    PyTorch.

    Adapted from Hadsell, Chopra, and LeCun (2006) [29]:
    [29] R. Hadsell, S. Chopra, and Y. LeCun. 2006. Dimensionality
    Reduction by Learning an Invariant Mapping. In 2006 IEEE Computer
    Society Conference on Computer Vision and Pattern Recognition
    (CVPR’06), 1735–1742. DOI:https://doi.org/10.1109/CVPR.2006.100


    And Shairoz. (2021) [31]:
    [31] ShairozS. 2021. losses.py. Gist 1a5e6953f0533cf19240ae1473eaedde.
    Retrieved November 4, 2024 from
    https://gist.github.com/ShairozS/1a5e6953f0533cf19240ae1473eaedde
    """

    def __init__(self, margin=1.0):
        """
        Initializes the Contrastive Loss model

        :param margin: The distance under which to push dissimilar pairs
        (label = 0) apart. Dissimilar pairs at a distance greater than the
        margin will have a loss of zero.
        :type margin: float
        """

        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, other, labels):
        """
        The forward step for contrastive loss. Calculates the contrastive
        loss between `anchor` and `other` given their labels `labels`.

        :param anchor, other: Final batched embeddings for input sentences
        A (achor) & B (other).
        :type anchor, other: torch.Tensor
        :param labels: Label indicating whether the sentences have
            different authors (0) or the same author (1).
        :type labels: int (0 or 1)
        """

        # Adapted from:
        # https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/ContrastiveLoss.py  # noqa E501

        # Calculate cosine similarity as per Ibrahim et al. (2023) [7:9]
        distances = (F.cosine_similarity(anchor, other) + 1) / 2

        # Loss: Similar pairs should have low distance;
        # dissimilar should have high
        # When label = 1 (same author), the loss is simply the first term,
        # `same_author_loss`, which is the squared distance, rewarding
        # closeness with lower # loss values.
        # When label = 0 (different author), the loss is the second term,
        # `diff_author_loss`, and grows quadratically the closer the pairs
        # are, starting at `margin` where the loss is zero.
        same_author_loss = labels.float() * distances.pow(2)
        diff_author_loss = (1 - labels).float() *\
            F.relu(self.margin - distances).pow(2)
        # Multiply by 0.5 to balance contribution of pos and neg pairs
        # and improve gradient stability
        losses = 0.5 * (same_author_loss + diff_author_loss)
        # Return the mean of all the losses
        return losses.mean()