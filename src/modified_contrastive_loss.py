"""
This is this project's implementation of Modified Contrastive Loss
described in Hadsell et al. (2006) [30], and implemented by Jacob Tyo
(2022) [52].

[30] R. Hadsell, S. Chopra, and Y. LeCun. 2006. Dimensionality Reduction
by Learning an Invariant Mapping. In 2006 IEEE Computer Society Conference
on Computer Vision and Pattern Recognition (CVPR’06), 1735–1742.
DOI:https://doi.org/10.1109/CVPR.2006.100

[52] Jacob Tyo. 2022. Losses.py (Version d063365). Valla repository on
GitHub (October 19, 2022). Retrieved from https://github.com/JacobTyo/Valla/blob/d063365a2adeb1d0f49d73e71fabd43a00bd52e1/valla/utils/Losses.py.  # noqa: E501
Accessed: January 31, 2025.
"""
import torch.nn as nn
import torch.nn.functional as F


class ModifiedContrastiveLoss(nn.Module):
    """
    Modified Contrastive Loss function

    Inherits from `nn.Module`, the base class for all neural networks from
    PyTorch.

    Adapted from:
    
    Hadsell, Chopra, and LeCun (2006) [30]:
    [29] R. Hadsell, S. Chopra, and Y. LeCun. 2006. Dimensionality
    Reduction by Learning an Invariant Mapping. In 2006 IEEE Computer
    Society Conference on Computer Vision and Pattern Recognition
    (CVPR’06), 1735–1742. DOI:https://doi.org/10.1109/CVPR.2006.100

    Shairoz. (2021) [32]:
    [31] ShairozS. 2021. losses.py. Gist 1a5e6953f0533cf19240ae1473eaedde.
    Retrieved November 4, 2024 from
    https://gist.github.com/ShairozS/1a5e6953f0533cf19240ae1473eaedde

    Tyo (2022) [52]:
    [52] Jacob Tyo. 2022. Losses.py (Version d063365). Valla repository on
    GitHub (October 19, 2022). Retrieved from https://github.com/JacobTyo/Valla/blob/d063365a2adeb1d0f49d73e71fabd43a00bd52e1/valla/utils/Losses.py.  # noqa: E501
    Accessed: January 31, 2025.
    """

    def __init__(self, margin_s=0.5, margin_d=0.5):
        """
        Initializes the Contrastive Loss model

        :param margin_s: The distance over which to pull similar pairs
        (label = 1) together. Similar pairs at a distance lower than the
        margin will have a loss of zero.
        :type margin: float

        :param margin_d: The distance under which to push dissimilar pairs
        (label = 0) apart. Dissimilar pairs at a distance greater than the
        margin will have a loss of zero.
        :type margin_d: float
        """

        super(ModifiedContrastiveLoss, self).__init__()
        self.margin_s = margin_s
        self.margin_d = margin_d

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

        # Calculate cosine similarity as per Ibrahim et al. (2023) [8:9]
        similarities = (F.cosine_similarity(anchor, other) + 1) / 2

        # Modified loss with two margins: Similar pairs should have low
        # distance; dissimilar should have high.
        # When label = 1 (same author), the loss is the first term of
        # `losses`, `same_author_loss`, and grows quadratically the
        # further the pairs are, starting at `similarity == margin_s`
        # before which the loss is zeroed by the `relu` function.
        # The second term, `diff_author_loss`, is zeroed out by
        # multiplication with its label: (0 * loss).
        # When label = 0 (different author), the loss is the second term,
        # `diff_author_loss`, and grows quadratically the closer the pairs
        # are, starting at `similarity == margin_d` before which the loss
        # is zeroed by the `relu` function.
        # The first term, `same_author_loss`, is zeroed out by
        # multiplication with one minus its label: ((1 - 1) * loss).
        same_author_loss = labels.float() *\
            F.relu(self.margin_s - similarities).pow(2)
        diff_author_loss = (1 - labels).float() *\
            F.relu(similarities - self.margin_d).pow(2)
        # Multiply by 0.5 to balance contribution of pos and neg pairs
        # and improve gradient stability
        losses = 0.5 * (same_author_loss + diff_author_loss)
        # Return the mean of all the losses
        return losses.mean()