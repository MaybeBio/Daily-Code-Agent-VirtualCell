import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ContinuousTimeEncoding(nn.Module):
    def __init__(self, width: int, base: float = 10000.0) -> None:
        super().__init__()
        if width % 2:
            raise ValueError("width must be even")
        frequencies = torch.exp(-math.log(base) * torch.arange(0, width, 2) / width)
        self.register_buffer("frequencies", frequencies, persistent=True)

    def forward(self, time: Tensor) -> Tensor:
        phase = time.unsqueeze(-1) * self.frequencies
        return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, width: int, expansion: int, dropout: float) -> None:
        super().__init__()
        hidden = width * expansion
        self.value = nn.Linear(width, hidden, bias=False)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.output = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, state: Tensor) -> Tensor:
        return self.dropout(self.output(self.value(state) * F.silu(self.gate(state))))


class TrajectoryAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.raw_decay = nn.Parameter(torch.zeros(heads))
        self.dropout = nn.Dropout(dropout)

    def split(self, tensor: Tensor) -> Tensor:
        batch, steps, _ = tensor.shape
        return tensor.view(batch, steps, self.heads, self.head_width).transpose(1, 2)

    def forward(self, state: Tensor, time: Tensor) -> Tensor:
        query = self.split(self.query(state))
        key = self.split(self.key(state))
        value = self.split(self.value(state))
        scores = query @ key.transpose(-1, -2) / math.sqrt(self.head_width)
        distance = (time.unsqueeze(-1) - time.unsqueeze(-2)).abs().unsqueeze(1)
        decay = F.softplus(self.raw_decay).view(1, self.heads, 1, 1)
        scores = scores - decay * distance
        steps = state.shape[1]
        causal = torch.ones(steps, steps, dtype=torch.bool, device=state.device).triu(1)
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        weights = self.dropout(scores.softmax(dim=-1))
        mixed = weights @ value
        mixed = mixed.transpose(1, 2).contiguous().view(state.shape)
        return self.output(mixed)


class TrajectoryLayer(nn.Module):
    def __init__(self, width: int, heads: int, expansion: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.feedforward_norm = nn.LayerNorm(width)
        self.attention = TrajectoryAttention(width, heads, dropout)
        self.feedforward = SwiGLU(width, expansion, dropout)

    def forward(self, state: Tensor, time: Tensor) -> Tensor:
        state = state + self.attention(self.attention_norm(state), time)
        return state + self.feedforward(self.feedforward_norm(state))


class TrajectoryStack(nn.Module):
    def __init__(self, width: int, heads: int, layers: int, expansion: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TrajectoryLayer(width, heads, expansion, dropout) for _ in range(layers)]
        )
        self.output_norm = nn.LayerNorm(width)

    def forward(self, state: Tensor, time: Tensor) -> Tensor:
        for layer in self.layers:
            state = layer(state, time)
        return self.output_norm(state)
