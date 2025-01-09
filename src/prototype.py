"""
Prototype of Siamese SBERT inspired by Ibrahim et al. (2023) [7]:
[7] Momen Ibrahim, Ahmed Akram, Mohammed Radwan, Rana Ayman, Mustafa Abd-El-Hameed, Nagwa M. El-Makky, and Marwan Torki. 2023. Enhancing Authorship Verification using Sentence-Transformers. In _Conference and Labs of the Evaluation Forum_. Retrieved from https://api.semanticscholar.org/CorpusID:264441704  # noqa E501


This prototype does not handle any 'meta' operations of training/inference.
Including but not limited to:
- K-folds cross-validation
- Batching
- Hyperparamter search/tuning
- Data pre/post-processing
- Document chunking
- Decision thresholds
- Etc.


What this prototype _does_ is train the model on a tiny batch of toy data
in one forward and one backward pass. It then uses this minimally trained
model to run a single inference.

The model architecture at training time is as follows:
1. > Sentence A & Sentence B are tokenized -------------------------------->
2. -> Tokenized Sentences A & B are fed into SBERT using the
        all-MiniLM-L12-v2 pre-trained model [^1] -------------------------->
3. -> SBERT embeddings for A & B are fed through a mean pooling layer, which
        converts multiple word embeddings representing the whole sentence
        into a single embedding by averaging the embeddings for all tokens
        in the sentence embedding. ---------------------------------------->
4. -> The pooled embeddings for A & B are fed to a contrastive loss
        function, which calculates the loss as the squared difference
        between embeddings when the label = 1 (similar pairs).
        When label = 0 (dissimilar pairs), the loss grows quadratically as
        the distance decreases and is clamped at zero above the value of the
        `margin` input. --------------------------------------------------->
5. -> Backpropagation adjusts the model's weights per the loss function.

The model architecture at inference time is as follows:
1-3. > Steps 1 through 3 are the same as those for training --------------->
4. -> The cosine-similarity for the pooled embeddings for A & B is
        calculated to produce a similarity score - intended to be evaluated
        against a decision threshold (to be implemented) to determine the
        label, 0, or 1.

[^1] https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
"""
import torch
import torch.nn as nn
from transformers import AutoModel

class SiameseBERT(nn.Module):
    """
    A Siamese SBERT model based on the pre-trained all-MiniLM-L12-v2 model
    from: https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2.

    Inherits from `nn.Module`, the base class for all neural networks from
    PyTorch.
    """

    def __init__(self, model_name):
        """
        Initializes a Siamese SBERT model based on a pre-trained model from
        the `transformers` library. In this case, this pre-trained model is
        `sentence-transformers/all-MiniLM-L12-v2`.

        :param model_name: The name of the pre-trained model from the
            `transformers` library.
        :type model_name: string
        """

        super(SiameseBERT, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        # Use DataParallel
        if torch.cuda.device_count() > 1:
            self.encoder = nn.DataParallel(self.encoder)

    def forward(self,
                input_ids_a,
                attention_mask_a,
                input_ids_b,
                attention_mask_b):
        """
        The forward step feeds the data through the network.
        This feeds the data through the SBERT pre-trained model, followed by
        a mean pooling layer, which condenses sentences' token embeddings
        into a single embedding representing the entire sentence by
        averaging all token-based embeddings.

        :param input_ids_a, input_ids_b: Ids indexing tokens in the
            pre-trained model's token vocabulary in the order they appear in
            the tokenized input sentences A and B.
        :type input_ids_a, input_ids_b: torch.Tensor
        :param attention_mask_a, attention_mask_b: Tensors of binary values
            (0 or 1) that act as masks. They mask (zero out) embeddings
            representing padding tokens if the input tokenized sentences to
            the BERT model were padded to reach a uniform length.
        :type attention_mask_a, attention_mask_b: torch.Tensor
        """

        # # Encode both inputs in parallel
        # outputs = self.encoder(
        #     torch.cat([input_ids_a, input_ids_b]),
        #     attention_mask=torch.cat([attention_mask_a, attention_mask_b])
        # )

        # # Split the outputs back
        # batch_size = input_ids_a.size(0)
        # embeddings = self.mean_pool(outputs.last_hidden_state,
        #                             torch.cat([attention_mask_a,
        #                                        attention_mask_b]))
        # embeddings_a, embeddings_b = torch.split(embeddings, batch_size)

        # return embeddings_a, embeddings_b
        # Encode first input
        outputs_a = self.encoder(input_ids_a,
                                 attention_mask=attention_mask_a)
        embeddings_a = self.mean_pool(outputs_a.last_hidden_state,
                                      attention_mask_a)

        # Encode second input
        outputs_b = self.encoder(input_ids_b,
                                 attention_mask=attention_mask_b)
        embeddings_b = self.mean_pool(outputs_b.last_hidden_state,
                                      attention_mask_b)

        return embeddings_a, embeddings_b

    def mean_pool(self, token_embeddings, attention_mask):
        """
        The mean pooling layer which averages the token embeddings for the
        full sentence into one embedding. This does not include padding
        tokens.

        Adapted from Tunstall et al. (2022) [30:276]:
        [30] Lewis Tunstall, Leandro Von Werra, and Thomas Wolf. 2022. Natural language processing with transformers. “ O’Reilly Media, Inc.” Retrieved from https://books.google.com.br/books?id=nzxbEAAAQBAJ&lpg=PP1&ots=sUqiDWs3MI&dq=Natural%20Language%20Processing%20with%20Transformers%20%5BRevised%20Edition%5D%201098136799%2C%209781098136796%2C%209781098103248&lr&pg=PA224#v=onepage&q&f=false  # noqa E501


        :param token_embeddings: A tensor of embeddings representing the
            tokens of the tokenized input sentence.
        :type token_embeddings: torch.Tensor
        :param attention_mask: A binary (0 or 1) mask representing which
            tokens to attend to - used to mask padding tokens so they do
            not affect the final pooler embedding.
        :type attention_mask: torch.Tensor
        """

        # Calculate the mean of token embeddings, ignoring padding tokens
        # Expand the 1D input mask to fit the token_embeddings of dim
        # `tokens` * `token length`
        input_mask_expanded = attention_mask.unsqueeze(-1).\
            expand(token_embeddings.size()).float()

        # Sum the non-padding tokens
        sum_embeddings = torch.sum(token_embeddings *
                                   input_mask_expanded, dim=1)

        # Get the number of non-padding tokens
        # counting tiny floats (lt 1e-9) as zero
        # due to float imprecision
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        # Calculate mean and normalize
        mean_embeddings = sum_embeddings / sum_mask

        # # TODO: Consider normalizing embeddings
        # embeddings_norm = torch.norm(mean_embeddings, p=2, dim=1, keepdim=True)
        # normalized_embeddings = mean_embeddings / torch.clamp(embeddings_norm, min=1e-9)

        return mean_embeddings


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss function

    Inherits from `nn.Module`, the base class for all neural networks from
    PyTorch.

    Adapted from Hadsell, Chopra, and LeCun (2006) [29]:
    [29] R. Hadsell, S. Chopra, and Y. LeCun. 2006. Dimensionality Reduction by Learning an Invariant Mapping. In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’06), 1735–1742. DOI:https://doi.org/10.1109/CVPR.2006.100  # noqa E501


    And Shairoz. (2021) [31]:
    [31] ShairozS. 2021. losses.py. Gist 1a5e6953f0533cf19240ae1473eaedde. Retrieved November 4, 2024 from https://gist.github.com/ShairozS/1a5e6953f0533cf19240ae1473eaedde  # noqa E501
    """

    def __init__(self, margin=1.0, eps=1e-6):
        """
        Initializes the Contrastive Loss model

        :param margin: The distance under which to push dissimilar pairs
            (label = 0) apart. Dissimilar pairs at a distance greater than
            the margin will have a loss of zero.
        :type margin: float
        """

        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = eps  # Small constant to prevent numerical instability

    def forward(self, u, v, label):
        """
        The forward step for contrastive loss. Calculates the contrastive
        loss between `u` and `v` given their label `label`.

        :param u, v: Final embeddings for input sentences A & B.
        :type u, v: torch.Tensor
        :param label: Label indicating whether the sentences have different
            authors (0) or the same author (1).
        :type label: int (0 or 1)
        """

        # Normalize embeddings
        u_norm = u / (torch.norm(u, dim=1, keepdim=True) + self.eps)
        v_norm = v / (torch.norm(v, dim=1, keepdim=True) + self.eps)

        # Calculate Euclidean distance
        distance = torch.norm(u_norm - v_norm, dim=1)

        # Clamp distance for numerical stability
        distance = torch.clamp(distance, min=self.eps)

        # Loss: Similar pairs should have low distance;
        # dissimilar should have high
        # When label = 1 (same author), the loss is simply the first term,
        # which is the squared distance, rewarding closeness with lower
        # loss values.
        # When label = 0 (different author), the loss grows quadratically
        # the closer the pairs are, starting at `margin` where the loss is
        # zero.
        positive_loss = label * distance.pow(2)
        negative_loss = (1 - label) * torch.clamp(self.margin - distance, min=0).pow(2)

        loss = positive_loss + negative_loss
        return torch.mean(loss)