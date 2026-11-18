"""The rank-token encoder shared by the training and embedding nodes."""

import torch
from torch import nn


class RankEncoder(nn.Module):
    """Gene and rank-position embeddings, a transformer encoder, output weights tied to the gene embedding.

    Token ids: 0 to vocab_size - 1 are genes, vocab_size is the mask, vocab_size + 1 is padding.
    """

    def __init__(self, vocab_size: int, seq_len: int, width: int, layers: int, heads: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_id = vocab_size
        self.pad_id = vocab_size + 1
        self.gene = nn.Embedding(vocab_size + 2, width)
        self.rank = nn.Embedding(seq_len, width)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        nn.init.normal_(self.gene.weight, std=0.02)
        nn.init.normal_(self.rank.weight, std=0.02)

    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        pad = tokens == self.pad_id
        pad[pad.all(dim=1), 0] = False  # a cell with no genes still needs one position to attend to
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        states = self.gene(tokens) + self.rank(positions)
        return self.norm(self.encoder(states, src_key_padding_mask=pad if pad.any() else None))

    def gene_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.gene.weight[: self.vocab_size].T

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mean of the hidden states over the gene positions of each cell."""
        keep = (tokens != self.pad_id).unsqueeze(-1)
        return (self.hidden(tokens) * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)


def trim(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Cut a batch to the length of its longest cell."""
    longest = int((tokens != pad_id).sum(dim=1).max())
    return tokens[:, : max(longest, 1)]
