"""Load a Thalamus-Decision checkpoint and read the option-code logits at each prompt's answer position, exactly.

The checkpoint has the file format of Thalamus-Reasoning (``config.json`` with ``llm_config``, ``omics_input_dim`` and
``omics_input_layout``; weights ``omics_projection.*`` and ``language_model.*``). It is loaded here without remote code:
the language model is the Qwen3.5 class of ``transformers`` and the omics projection is one linear layer.

The readout is the one the model was calibrated with:

* each profile is one token: the prompt carries ``<|vision_start|><|image_pad|><|vision_end|>`` where the state had an
  ``<omics>`` placeholder, and the projected profile replaces the ``<|image_pad|>`` embedding;
* Qwen3.5 mixes Gated DeltaNet (recurrent) and full-attention layers, so prompts are never padded on the left. A batch
  is packed into one sequence whose position ids restart at 0 for every prompt: one varlen forward (cu_seqlens for
  FlashAttention 2, seq_idx for the DeltaNet kernels). Without FlashAttention or on CPU every prompt runs on its own;
* batch invariance: every packed forward is padded to at least ``MIN_PACK`` tokens with an isolated filler, every
  profile is projected on its own and the option-code logits are computed in float32, so an answer does not depend on
  what else is in the batch;
* only the answer positions go through the output layer.
"""
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn as nn

OMICS = "<omics>"
OMICS_SPAN = "<|vision_start|><|image_pad|><|vision_end|>"
MIN_PACK = 4096
# Blocks of a profile vector as Thalamus-Reasoning's tr-embed writes it (20,029 values).
PROFILE_LAYOUT = {"scale_indicator": (0, 1), "expression": (1, 19261), "geneformer_v2_104m": (19261, 20029)}
PROFILE_DIM = 20029


class OmicsProjection(nn.Module):
    """Linear map from a profile vector to one token embedding (optionally reading only some ``[start, end)`` ranges)."""

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True, slices=None):
        super().__init__()
        self.input_dim = input_dim
        self.slices = [(int(s), int(e)) for s, e in (slices or [(0, input_dim)])]
        self.linear = nn.Linear(sum(e - s for s, e in self.slices), output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [x[..., s:e] for s, e in self.slices]
        x = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        return self.linear(x.to(self.linear.weight.dtype))


class DecisionModel(nn.Module):
    def __init__(self, config: dict, language_model: nn.Module, projection: OmicsProjection, packed: bool):
        super().__init__()
        self.config = config
        self.language_model = language_model
        self.omics_projection = projection
        self.packed = packed
        self.image_token_index = int(config["image_token_index"])
        self.pad_token_id = int(config.get("pad_token_id") or 0)
        self.input_dim = int(config["omics_input_dim"])
        self._fill = profile_filler(config)

    def model_inputs(self, vectors: np.ndarray) -> np.ndarray:
        """The projection's input: the 20,029-value profiles as tr-embed writes them."""
        return self._fill(vectors)


def profile_filler(config: dict):
    """Check that the checkpoint reads the 20,029-value profiles tr-embed writes, block by block; returns the identity."""
    layout = config.get("omics_input_layout")
    dim = int(config["omics_input_dim"])
    if dim != PROFILE_DIM or (layout is not None and {k: tuple(v) for k, v in layout.items()} != PROFILE_LAYOUT):
        raise ValueError(f"the checkpoint takes {dim}-value profiles with blocks {list(layout or [])}; this engine serves "
                         f"the {PROFILE_DIM:,}-value profiles tr-embed writes {list(PROFILE_LAYOUT)}")
    return lambda v: np.asarray(v, np.float32)


def load_config(path) -> dict:
    return json.loads((Path(path) / "config.json").read_text())


def flash_available() -> bool:
    try:
        import flash_attn  # noqa: F401
        import fla  # noqa: F401
        import causal_conv1d  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def load_model(path, device="cuda", dtype=torch.bfloat16, packed=None) -> DecisionModel:
    """The checkpoint in ``path`` on ``device``. ``packed`` (default: when FlashAttention 2, flash-linear-attention and
    causal-conv1d are installed and a GPU is used) selects the packed varlen forward of the calibration readout."""
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.initialization import no_init_weights

    path = Path(path)
    cfg = load_config(path)
    packed = (flash_available() and str(device).startswith("cuda")) if packed is None else packed
    llm = dict(cfg["llm_config"])
    llm_cfg = AutoConfig.for_model(llm.get("model_type", "qwen3_5_text")).from_dict(llm)
    attn = "flash_attention_2" if packed else "sdpa"
    with no_init_weights():
        lm = AutoModelForCausalLM.from_config(llm_cfg, dtype=dtype, attn_implementation=attn)
    files = sorted(path.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no model*.safetensors in {path}")
    proj_state, lm_state = {}, {}
    for f in files:
        with safe_open(f, "pt") as h:
            for k in h.keys():
                if k.startswith("omics_projection."):
                    proj_state[k[len("omics_projection."):]] = h.get_tensor(k)
                elif k.startswith("language_model."):
                    lm_state[k[len("language_model."):]] = h.get_tensor(k)
    lm.load_state_dict(lm_state, strict=True)
    proj = OmicsProjection(int(cfg["omics_input_dim"]), int(llm["hidden_size"]), bool(cfg.get("projection_bias", True)),
                           cfg.get("omics_input_slices"))
    proj.load_state_dict(proj_state, strict=True)
    if packed and llm.get("model_type") == "qwen3_5_text":
        for layer in lm.model.layers:
            if layer.layer_type == "linear_attention":
                layer.forward = MethodType(_packed_decoder_forward, layer)
    return DecisionModel(cfg, lm, proj, packed).to(device=device, dtype=dtype).eval()


# ---- packed (varlen) forward of the Qwen3.5 decoder ---------------------------------------------------------------------
def _packed_linear_attention(self, hidden_states, linear_chunks, seq_idx):
    batch, length, _ = hidden_states.shape
    qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    qkv = self.causal_conv1d_fn(x=qkv, weight=self.conv1d.weight.squeeze(1), bias=self.conv1d.bias,
                                activation=self.activation, seq_idx=seq_idx).transpose(1, 2)
    query, key, value = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(batch, length, self.num_k_heads, self.head_k_dim)
    key = key.reshape(batch, length, self.num_k_heads, self.head_k_dim)
    value = value.reshape(batch, length, self.num_v_heads, self.head_v_dim)
    repeats = self.num_v_heads // self.num_k_heads
    if repeats > 1:
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)
    beta = self.in_proj_b(hidden_states).sigmoid()
    gate = -self.A_log.float().exp() * nn.functional.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)
    parts = []
    for start, end, cu_seqlens in linear_chunks:
        out, _ = self.chunk_gated_delta_rule(
            query[:, start:end], key[:, start:end], value[:, start:end], g=gate[:, start:end],
            beta=beta[:, start:end], cu_seqlens=cu_seqlens, initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=True)
        parts.append(out)
    out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
    z = self.in_proj_z(hidden_states).reshape(-1, self.head_v_dim)
    out = self.norm(out.reshape(-1, self.head_v_dim), z)
    return self.out_proj(out.reshape(batch, length, self.value_dim))


def _packed_decoder_forward(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None,
                            past_key_values=None, linear_chunks=None, seq_idx=None, **kwargs):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    if linear_chunks is None:
        return Qwen3_5DecoderLayer.forward(self, hidden_states, position_embeddings, attention_mask=attention_mask,
                                           position_ids=position_ids, past_key_values=past_key_values, **kwargs)
    residual = hidden_states
    hidden_states = _packed_linear_attention(self.linear_attn, self.input_layernorm(hidden_states), linear_chunks,
                                             seq_idx)
    hidden_states = residual + hidden_states
    return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


def _packed_last_hidden(decoder, embeds, positions):
    """Final hidden states of a packed row (1, L, D) whose position ids restart at 0 for every prompt."""
    starts = (positions == 0).flatten()
    starts[0] = True
    cu = torch.cat([starts.nonzero().flatten(), starts.new_tensor([starts.numel()], dtype=torch.long)]).to(torch.int32)
    seq_idx = (starts.cumsum(0, dtype=torch.int32) - 1).unsqueeze(0)
    max_len = int(cu.diff().max())
    limit = 65535 // decoder.config.linear_num_value_heads  # FLA launches a grid of sequences x value heads
    if cu.numel() - 1 <= limit:
        chunks = [(0, starts.numel(), cu)]
    else:
        b = cu.tolist()
        chunks = [(b[i], b[min(i + limit, len(b) - 1)], cu[i:i + limit + 1] - b[i]) for i in range(0, len(b) - 1, limit)]
    out = decoder(inputs_embeds=embeds, position_ids=positions, use_cache=False, linear_chunks=chunks, seq_idx=seq_idx,
                  cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=max_len, max_length_k=max_len, return_dict=True)
    return out.last_hidden_state[0]


@torch.inference_mode()
def answer_hidden(model: DecisionModel, seqs, vectors=None, min_pack=MIN_PACK):
    """Final hidden state at the last token of every prompt. ``seqs``: token id lists; ``vectors``: (P, input_dim)
    model inputs of all profiles in order of their tokens (or None)."""
    lm = model.language_model
    dev = next(lm.parameters()).device
    n_marks = sum(s.count(model.image_token_index) for s in seqs)
    if n_marks != (0 if vectors is None else len(vectors)):
        raise ValueError(f"{n_marks} profile tokens in the prompts for {0 if vectors is None else len(vectors)} vectors")
    rows = None
    if n_marks:
        w = model.omics_projection.linear.weight
        v = torch.as_tensor(np.asarray(vectors, np.float32)).to(device=w.device, dtype=w.dtype)
        # one projection per profile: a GEMM over several profiles picks its kernel by the row count
        rows = torch.cat([model.omics_projection(v[i:i + 1]) for i in range(len(v))])
    if not model.packed:
        out, k = [], 0
        for s in seqs:
            ids = torch.tensor([s], device=dev)
            emb = lm.get_input_embeddings()(ids)
            mask = ids == model.image_token_index
            m = int(mask.sum())
            if m:
                emb = emb.clone()
                emb[mask] = rows[k:k + m].to(emb.dtype)
                k += m
            out.append(lm.model(inputs_embeds=emb, use_cache=False, return_dict=True).last_hidden_state[0, -1])
        return torch.stack(out)
    total = sum(map(len, seqs))
    fill = [model.pad_token_id] * (min_pack - total) if total < min_pack else []
    ids = torch.tensor([[t for s in seqs for t in s] + fill], device=dev)
    pos = torch.tensor([[p for s in seqs for p in range(len(s))] + list(range(len(fill)))], device=dev)
    emb = lm.get_input_embeddings()(ids)
    if n_marks:
        mask = ids == model.image_token_index
        emb = emb.clone()
        emb[mask] = rows.to(emb.dtype)
    h = _packed_last_hidden(lm.model, emb, pos)
    ends = torch.tensor([len(s) for s in seqs], device=dev).cumsum(0) - 1
    return h[ends]


def code_logits(model: DecisionModel, h, code_ids):
    """float32 logits of the given option codes at the answer positions."""
    head = model.language_model.lm_head
    idx = torch.as_tensor(code_ids, device=h.device)
    with torch.autocast(h.device.type, enabled=False):
        out = h.float() @ head.weight[idx].float().T
        if getattr(head, "bias", None) is not None:
            out = out + head.bias[idx].float()
    return out


def answer_scores(model: DecisionModel, seqs, vectors, code_ids, min_pack=MIN_PACK):
    """The readout: float32 option-code logits at each prompt's answer position."""
    return code_logits(model, answer_hidden(model, seqs, vectors, min_pack), code_ids)


def load_tokenizer(path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=False)
    tok.pad_token = tok.pad_token or tok.eos_token
    return tok
