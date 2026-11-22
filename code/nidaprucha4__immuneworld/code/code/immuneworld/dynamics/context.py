import torch
from torch import Tensor, nn


class PerturbationEncoder(nn.Module):
    def __init__(self, genes: int, width: int, types: int, shared_embedding: nn.Parameter) -> None:
        super().__init__()
        self.genes = genes
        self.shared_embedding = shared_embedding
        self.type_embedding = nn.Embedding(types, width)
        self.dose_projection = nn.Linear(1, width, bias=False)
        self.output_norm = nn.LayerNorm(width)

    def forward(
        self,
        gene_ids: Tensor,
        doses: Tensor,
        type_ids: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        genes = torch.nn.functional.embedding(gene_ids, self.shared_embedding)
        context = genes + self.type_embedding(type_ids) + self.dose_projection(doses.unsqueeze(-1))
        if mask is None:
            return self.output_norm(context.mean(dim=-2, keepdim=True))
        weights = mask.to(context.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=-2, keepdim=True).clamp_min(1.0)
        return self.output_norm((context * weights).sum(dim=-2, keepdim=True) / denominator)


class PerturbationInjector(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout, batch_first=True)

    def forward(self, state: Tensor, context: Tensor) -> Tensor:
        normalized = self.norm(state)
        injected, _ = self.cross_attention(normalized, context, context, need_weights=False)
        return state + injected
