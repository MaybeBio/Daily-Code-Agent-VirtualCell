"""Expression profiles, handled exactly as Thalamus-Reasoning handles them.

Nothing here re-implements profile handling: vectors, files, cell-line matching and placeholders all come from the
``thalamus-reasoning`` package (github.com/IngenixAI/Thalamus-Reasoning), so a profile means the same thing to both
models.

* A profile is one 20,029-value vector (data-type flag, 19,260-gene expression block, Geneformer-V2 block), built by
  ``tr-embed`` from your expression table or ``.h5ad`` (``thalamus_reasoning.preprocess``), with the gene panel and
  normalization files in the model repository's ``preprocessing/`` folder.
* Your samples come from ``.npz`` files written by ``tr-embed`` (``thalamus_reasoning.load_profiles``) and are referred
  to as ``@sample_id``. The 1,673 DepMap cell lines in ``depmap/`` are referred to by name (``MCF7``, ``mcf-7``,
  ``ACH-000019``, ``CVCL_0031``; ``@RD`` or "RD cells" for names that are also words or gene symbols).
* Text with references is resolved by ``thalamus_reasoning.chat.Conversation.prepare``, the code of ``tr-chat`` and
  ``tr-app``: the first mention of each sample or cell line gets an ``<omics>`` placeholder right after it, and the
  notes say what was matched. Text with explicit ``<omics>`` placeholders and vectors is used as given, as in
  ``ThalamusReasoning.ask(text, vectors)``.
"""
import io
import re
import tempfile
import threading
from pathlib import Path

import numpy as np

from thalamus_reasoning.chat import Conversation
from thalamus_reasoning.depmap import DepMap
from thalamus_reasoning.model import OMICS, escape_omics
from thalamus_reasoning.preprocess import VECTOR_DIM, InputError, Preprocessor, chat_safe_ids, load_profiles

SAMPLE_ID = re.compile(r"^[A-Za-z0-9][\w.:\-]*$")  # what an @reference can name (thalamus_reasoning.chat)


class ProfileError(ValueError):
    """An invalid or unknown profile: the request must change (HTTP 422)."""


def check_vectors(vectors) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float32)
    if v.ndim == 1:
        v = v[None]
    if v.ndim != 2 or v.shape[1] != VECTOR_DIM:
        raise ProfileError(f"profiles must have {VECTOR_DIM:,} values (as written by tr-embed), got shape {v.shape}")
    if not np.isfinite(v).all():
        raise ProfileError("profiles must be finite")
    return v


class Profiles:
    """Your samples (by ID), the DepMap cell lines (by name) and, optionally, tr-embed on the server.

    ``source``: the model folder or Hub repository with ``preprocessing/`` and ``depmap/`` (a Thalamus-Decision or
    Thalamus-Reasoning repository; the files are the same).
    """

    def __init__(self, source=None, depmap=True, store_dir=None, embed=False, device=None):
        self.source = str(source) if source else None
        self.depmap = DepMap(self.source) if depmap else None
        self.samples: dict[str, np.ndarray] = {}
        self.origin: dict[str, str] = {}
        self.store_dir = Path(store_dir) if store_dir else None
        self.embed_enabled = embed
        self.device = device
        self._pre = None
        self._lock = threading.Lock()
        if self.store_dir:
            self.store_dir.mkdir(parents=True, exist_ok=True)
            for f in sorted(self.store_dir.glob("*.npz")):
                self.load_npz(f, persist=False)

    # ---- adding samples ---------------------------------------------------------------------------------------------
    def add(self, vectors: dict, origin: str, persist=True) -> list[str]:
        """Add samples {id: vector}. IDs follow the chat's @reference rule; an existing ID is replaced."""
        ids = list(vectors)
        bad = [i for i in ids if not isinstance(i, str) or not SAMPLE_ID.match(i)]
        if bad:
            raise ProfileError(f"sample IDs must start with a letter or digit and use only letters, digits and _ . : - "
                               f"(as @references in the chat), got {bad[:5]}")
        mat = check_vectors(np.stack([np.asarray(vectors[i], np.float32).reshape(-1) for i in ids])) if ids else None
        with self._lock:
            for i, row in zip(ids, mat if mat is not None else []):
                self.samples[i] = row
                self.origin[i] = origin
        if persist and self.store_dir and ids:
            name = re.sub(r"[^\w.-]+", "_", origin)[:80] or "upload"
            np.savez(self.store_dir / f"{name}.npz", embeddings=mat, sample_ids=np.array(ids))
        return ids

    def load_npz(self, path, persist=False) -> list[str]:
        """Samples from a tr-embed ``.npz`` (``embeddings`` and ``sample_ids``)."""
        try:
            profiles = load_profiles(path)
        except (OSError, KeyError, ValueError) as e:
            raise ProfileError(f"could not read {Path(str(path)).name}: {e}") from None
        return self.add(profiles, Path(str(path)).stem, persist)

    def load_npz_bytes(self, data: bytes, origin="upload") -> list[str]:
        try:
            npz = np.load(io.BytesIO(data), allow_pickle=False)
            vectors, ids = npz["embeddings"], npz["sample_ids"].astype(str)
        except Exception as e:
            raise ProfileError(f"not a tr-embed .npz (needs 'embeddings' and 'sample_ids'): {e}") from None
        if vectors.ndim != 2 or len(ids) != len(vectors):
            raise ProfileError(f"'embeddings' must be (n, {VECTOR_DIM}) with one sample ID per row")
        return self.add(dict(zip(ids, vectors)), origin)

    def embed_table(self, data: bytes, filename: str, **options):
        """Run tr-embed on an uploaded expression table or .h5ad (as the Thalamus-Reasoning web app does)."""
        if not self.embed_enabled:
            raise ProfileError("this server does not embed tables (start it with --embed on); run tr-embed yourself and "
                               "upload the .npz")
        suffix = "".join(Path(filename).suffixes[-2:]) or ".tsv"
        with self._lock:
            if self._pre is None:
                self._pre = Preprocessor(self.source, self.device)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{suffix}"
            path.write_bytes(data)
            try:
                vectors, ids, report = self._pre.embed_file(path, max_profiles=options.pop("max_profiles", 200), **options)
            except InputError as e:
                raise ProfileError(str(e)) from None
        self.add(dict(zip(ids, vectors)), Path(filename).name)
        return ids, report.to_dict()

    # ---- lookup -------------------------------------------------------------------------------------------------------
    def list_samples(self, prefix="", limit=1000) -> list[dict]:
        names = sorted(k for k in self.samples if k.startswith(prefix))[:limit]
        return [dict(id=k, origin=self.origin.get(k)) for k in names]

    def info(self, sample_id: str) -> dict:
        v = self.samples.get(sample_id)
        if v is None:
            raise ProfileError(f"unknown sample @{sample_id}")
        return dict(id=sample_id, origin=self.origin.get(sample_id), dim=int(v.shape[0]),
                    kind="single_cell" if v[0] > -1 else "bulk", norm=float(np.linalg.norm(v)))

    def lines(self, query="", limit=50) -> list[dict]:
        if self.depmap is None:
            return []
        table = self.depmap.available(query)
        return table.head(limit).fillna("").to_dict(orient="records")  # some lines lack an ID or a disease (NaN)

    # ---- placing profiles in text -----------------------------------------------------------------------------------
    def resolve(self, text: str):
        """``tr-chat``'s resolution of one message: (text with ``<omics>`` placeholders, vectors, notes)."""
        conv = Conversation(lm=None, depmap=self.depmap, profiles=self.samples)
        turn = conv.prepare(text)
        return turn.text, [np.asarray(v, np.float32) for v in turn.vectors], turn.notes

    def reference(self, ref: str):
        """One profile named by ``@sample_id`` or a cell-line name: (vector, what it resolved to)."""
        if not isinstance(ref, str) or not ref.strip():
            raise ProfileError("a profile reference is '@sample_id' or a DepMap cell-line name")
        ref = ref.strip()
        sample = ref[1:] if ref.startswith("@") else ref
        if sample in self.samples:
            return self.samples[sample], f"@{sample}"
        if self.depmap is not None:
            res = self.depmap.resolver.resolve(ref if ref.startswith("@") else f"@{ref}")  # @ forces word-like names
            if len(res.rows) == 1:
                line = res.resolved[0][1]
                return np.asarray(self.depmap.embeddings[res.rows[0]], np.float32), f"{line.name} ({line.depmap_id})"
            if res.missing:
                raise ProfileError(f"{ref}: DepMap has no expression profile for {res.missing[0][1].name}")
        raise ProfileError(f"unknown profile {ref!r}: not a loaded sample (@id) or a DepMap cell line")


__all__ = ["OMICS", "Profiles", "ProfileError", "check_vectors", "escape_omics", "chat_safe_ids", "VECTOR_DIM"]
