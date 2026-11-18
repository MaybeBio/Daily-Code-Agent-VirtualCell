"""The GPU backend: option-code probabilities for a list of prompts (jobs).

A job is (token ids, profiles in placeholder order, number of shown options k); the backend returns, per job, the
softmax over the first k option codes at the answer position (presentation order). It is the readout the model was
calibrated with (``model.answer_scores``): bf16 weights, packed varlen forwards, float32 option-code logits, every
forward padded to at least ``MIN_PACK`` tokens, every profile projected on its own.

Shape buckets: the Triton kernels of the Gated DeltaNet layers are specialised on the packed length, so a forward of
a new length costs a compilation (about 12 s on an H100) the first time. Every packed forward is therefore padded to
one of a few token counts (``BUCKETS``) and every bucket is run once at start-up, so serving only meets warmed shapes.
"""
import time

import numpy as np
import torch

from . import prompt
from .model import MIN_PACK, OMICS, OMICS_SPAN, PROFILE_DIM, answer_scores, load_model, load_tokenizer

BUCKETS = [4096 * i for i in range(1, 9)] + [32768 + 8192 * i for i in range(1, 5)]


def bucket_of(n, buckets=BUCKETS):
    """The smallest bucket of at least n tokens (beyond the largest: n rounded up to a multiple of 8192)."""
    return next((b for b in buckets if b >= n), -(-n // 8192) * 8192)


class EngineOOM(RuntimeError):
    """The GPU ran out of memory for this batch (HTTP 503: retry later)."""


def _softmax(x):
    x = np.asarray(x, np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def plan(lengths, max_rows, max_tokens):
    """Micro-batches (lists of job indices) of length-sorted jobs within a row and token budget."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out, cur, tok = [], [], 0
    for i in order:
        if cur and (len(cur) >= max_rows or tok + lengths[i] > max_tokens):
            out.append(cur)
            cur, tok = [], 0
        cur.append(i)
        tok += lengths[i]
    if cur:
        out.append(cur)
    return out


class Backend:
    def __init__(self, model_dir, device="cuda", min_pack=MIN_PACK, max_rows=64, max_tokens=65536, packed=None):
        self.tok = load_tokenizer(model_dir)
        self.model = load_model(model_dir, device=device, dtype=torch.bfloat16, packed=packed)
        self.table = prompt.code_table(self.tok)
        self.image_token = self.model.image_token_index
        if self.tok.convert_tokens_to_ids("<|image_pad|>") != self.image_token:
            raise SystemExit("the tokenizer and the checkpoint disagree on the profile token")
        self.min_pack = min_pack if self.model.packed else 0
        self.max_rows = max_rows if self.model.packed else 8
        self.max_tokens = max_tokens
        self.device = device
        self.forwards = 0
        self.warming = False
        self.slow = []  # (tokens, bucket, seconds) of forwards slower than 3 s: a shape that was not warmed

    @torch.inference_mode()
    def score(self, jobs):
        out = [None] * len(jobs)
        try:
            for mb in plan([len(j[0]) for j in jobs], self.max_rows, self.max_tokens):
                vecs = [v for i in mb for v in jobs[i][1]]
                kmax = max(jobs[i][2] for i in mb)
                n_tok = sum(len(jobs[i][0]) for i in mb)
                pad = bucket_of(max(n_tok, self.min_pack)) if self.model.packed else 0
                t0 = time.perf_counter()
                inputs = self.model.model_inputs(np.stack(vecs)) if vecs else None
                logits = answer_scores(self.model, [jobs[i][0] for i in mb], inputs, self.table[:kmax], pad)
                logits = logits.cpu().numpy()
                for r, i in enumerate(mb):
                    out[i] = _softmax(logits[r, :jobs[i][2]])
                self.forwards += 1
                dt = time.perf_counter() - t0
                if dt > 3 and self.model.packed and not self.warming:
                    self.slow.append((n_tok, pad, round(dt, 1)))
                    print(f"slow forward: {n_tok} tokens in bucket {pad}: {dt:.1f}s", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise EngineOOM("the GPU ran out of memory for this batch") from None
        return out


def warmup(backend, log=print):
    """One small decision with and without a profile, then every shape bucket once (the DeltaNet kernels compile on
    first use), before the server accepts requests."""
    t0 = time.perf_counter()
    backend.warming = True
    tok = backend.tok
    jobs = []
    for n in (0, 1):
        state = (f"Expression profile: {OMICS}\n" if n else "") + "Warm-up state."
        text = prompt.render(tok, state, "Which option?", ["first", "second", "third"], "en").replace(OMICS, OMICS_SPAN)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        jobs.append((ids, [np.zeros(PROFILE_DIM, np.float32)] * n, 3))
    backend.score(jobs)
    if backend.model.packed:
        for b in BUCKETS:
            if b > backend.max_tokens:
                break
            t1 = time.perf_counter()
            n_rep = min(max(1, int(b * 0.9) // len(jobs[1][0])), backend.max_rows)
            ids = [list(jobs[1][0]) for _ in range(n_rep)]
            vecs = backend.model.model_inputs(np.zeros((n_rep, PROFILE_DIM), np.float32))
            answer_scores(backend.model, ids, vecs, backend.table[:3], b)
            log(f"  bucket {b:6d} tokens: {time.perf_counter() - t1:5.1f}s")
    backend.slow, backend.warming = [], False
    log(f"warm-up done in {time.perf_counter() - t0:.1f}s")
