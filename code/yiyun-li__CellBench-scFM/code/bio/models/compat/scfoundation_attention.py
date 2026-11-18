"""Bound peak attention memory without dropping genes or changing weights.

The pinned upstream uses torch TransformerEncoderLayer. PyTorch 2.2's fused
eval path can allocate the full heads x genes x genes matrix. Evaluate query
blocks against ALL keys instead; each softmax still covers the full sequence.
"""
from types import MethodType

import torch
from torch.nn import functional as F


def _query_block_attention(self, x, attn_mask, key_padding_mask, is_causal=False):
    if self.training or is_causal or attn_mask is not None:
        raise ValueError('scFoundation query-block attention supports frozen noncausal inference only')
    mha = self.self_attn
    batch, length, dim = x.shape
    qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
    qkv = qkv.reshape(batch, length, 3, mha.num_heads, dim // mha.num_heads)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    bias = None
    if key_padding_mask is not None and torch.any(key_padding_mask):
        if key_padding_mask.dtype == torch.bool:
            bias = torch.zeros_like(key_padding_mask, dtype=x.dtype).masked_fill(
                key_padding_mask, float('-inf'))
        else:
            bias = key_padding_mask.to(dtype=x.dtype)
        bias = bias[:, None, None, :]
    output = None
    # Single-cell native inference has no padding. Use a tiled CUDA kernel for
    # all queries when available, explicitly prohibiting the quadratic math
    # fallback. Older devices/kernels still have the bounded query-block path.
    if x.is_cuda and bias is None:
        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False,
                                                 enable_mem_efficient=True):
                output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        except RuntimeError as exc:
            if 'No available kernel' not in str(exc):
                raise
    if output is None:
        blocks = []
        for start in range(0, length, self.attention_query_chunk_size):
            blocks.append(F.scaled_dot_product_attention(
                q[:, :, start:start + self.attention_query_chunk_size], k, v,
                attn_mask=bias, dropout_p=0.0, is_causal=False))
        output = torch.cat(blocks, dim=2)
    output = output.transpose(1, 2).reshape(batch, length, dim)
    return self.dropout1(F.linear(output, mha.out_proj.weight, mha.out_proj.bias))


def configure_query_blocks(encoder, chunk_size=256):
    if int(chunk_size) < 1:
        raise ValueError('attention_query_chunk_size must be positive')
    for layer in encoder.transformer_encoder:
        # This is PyTorch's dispatch flag, not its activation function. A zero
        # flag selects the ordinary residual/norm/FFN forward, bypassing fused
        # attention; layer.activation and every parameter remain unchanged.
        layer.activation_relu_or_gelu = 0
        layer.attention_query_chunk_size = int(chunk_size)
        layer._sa_block = MethodType(_query_block_attention, layer)
