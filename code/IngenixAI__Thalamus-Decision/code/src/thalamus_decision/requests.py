"""Requests -> prompts (branches) -> calibrated answers.

A request is a ``state`` (text or a JSON object) and named ``questions`` of three types: ``choice`` (one of 2-255
named options), ``noul`` (yes / no) and ``score`` (a level of an ordered scale). Every question is read in two option
orders (as given and reversed); its probabilities are the mean of the two, then calibrated with ``CALIBRATION.json``.

Profiles enter a request in one of three ways (one per request):

* **References in the state** (default): ``@sample_id`` for your samples and DepMap cell-line names, resolved exactly as
  ``tr-chat`` does (``thalamus_reasoning.chat.Conversation.prepare``): the first mention of each gets an ``<omics>``
  placeholder right after it.
* **Explicit vectors**: ``"vectors": [v1, v2, ...]`` (20,029 floats each, or base64 of little-endian float32) and one
  ``<omics>`` placeholder per vector in the state, in order, as in ``ThalamusReasoning.ask(text, vectors)``. Names in the
  state are not resolved.
* **A profile list**: ``"profiles": ["@GSM1001", "MCF7"]`` (or ``{"ref": ..., "label": ...}``). Each becomes a line
  ``Expression profile: <omics>`` before the state, the form of the model's single-profile training states. Names in
  the state are not resolved.
"""
import base64
import json
from dataclasses import dataclass, field

import numpy as np

from . import prompt
from .calibration import CATEGORY_BY_TYPE
from .model import OMICS, OMICS_SPAN
from .profiles import ProfileError, check_vectors, escape_omics

TYPES = ("choice", "noul", "score")
TYPE_ALIASES = {"yes_no": "noul", "yesno": "noul", "yes/no": "noul", "boolean": "noul", "binary": "noul",
                "multiple_choice": "choice", "scale": "score", "ordinal": "score"}
YES_NO = {"en": ("Yes", "No"), "pl": ("Tak", "Nie")}
ORDERS = ("orig", "rev")
DEFAULT_LABEL = "Expression profile"
RESERVED = ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>")


class RequestError(ValueError):
    """The request is invalid and must be changed (HTTP 422)."""


@dataclass
class Question:
    name: str
    prim: str
    instructions: str
    keys: list            # answer keys, canonical order
    shown: list           # option texts the model sees, canonical order
    max_error: float | None = None
    category: str | None = None
    rank: int = 0
    raw: bool = False


@dataclass
class Branch:
    q: int                # question index in the request
    order: str
    perm: list            # canonical indices in presentation order
    ids: list             # token ids
    probs: np.ndarray | None = None


@dataclass
class Prepared:
    state: str            # with <omics> placeholders
    lang: str
    vectors: list         # one 20,029-value profile per placeholder, in order
    notes: list
    questions: list
    branches: list = field(default_factory=list)


def permute(n: int, order: str) -> list[int]:
    """Canonical option indices in presentation order."""
    idx = list(range(n))
    return idx[::-1] if order == "rev" else idx


def _decode(v, i):
    if isinstance(v, str):
        try:
            raw = base64.b64decode(v, validate=True)
        except Exception:
            raise RequestError(f"vectors[{i}]: not base64") from None
        if len(raw) % 4:
            raise RequestError(f"vectors[{i}]: base64 length is not a whole number of float32 values")
        return np.frombuffer(raw, dtype="<f4").astype(np.float32)
    if isinstance(v, list):
        return np.asarray(v, dtype=np.float32)
    raise RequestError(f"vectors[{i}]: a list of numbers or base64 float32")


def _options(spec, prim, lang, name):
    crit = spec.get("criteria")
    hide = spec.get("option_keys", "show") == "hide"
    if spec.get("option_keys", "show") not in ("show", "hide"):
        raise RequestError(f"{name}: option_keys is 'show' or 'hide'")
    if prim == "noul":
        y, n = YES_NO[lang]
        if crit is None:
            return ["true", "false"], [y, n]
        if isinstance(crit, list) and len(crit) == 2 and all(isinstance(c, str) for c in crit):
            if [c.strip().lower() for c in crit] in (["yes", "no"], ["tak", "nie"], ["true", "false"]):
                return ["true", "false"], [y, n]
            crit = {"true": crit[0], "false": crit[1]}  # [what yes means, what no means]
        if not isinstance(crit, dict) or set(crit) - {"true", "false"}:
            raise RequestError(f"{name}: noul criteria are optional: {{\"true\": what yes means, \"false\": what no "
                               f"means}} or [what yes means, what no means]")
        return ["true", "false"], [f"{y}: {crit['true']}" if crit.get("true") else y,
                                   f"{n}: {crit['false']}" if crit.get("false") else n]
    if isinstance(crit, list):
        if not all(isinstance(c, str) and c.strip() for c in crit):
            raise RequestError(f"{name}: criteria list items must be non-empty strings")
        keys, shown = [str(i) for i in range(len(crit))] if prim == "score" else list(crit), list(crit)
    elif isinstance(crit, dict):
        keys, shown = [], []
        for k, d in crit.items():
            keys.append(str(k))
            shown.append(str(k) if d in (None, "") else (str(d) if hide else f"{k}: {d}"))
    else:
        raise RequestError(f"{name}: criteria must be a list of options or a {{key: description}} object")
    if len(set(keys)) != len(keys) or len(set(shown)) != len(shown):
        raise RequestError(f"{name}: options must be distinct")
    if not 2 <= len(keys) <= prompt.MAX_OPTIONS:
        raise RequestError(f"{name}: {len(keys)} options (2 to {prompt.MAX_OPTIONS} supported)")
    return keys, [escape_omics(s) for s in shown]


def place_profiles(req, state, profiles):
    """(state with <omics> placeholders, vectors, notes) by the request's way of naming profiles."""
    vectors, refs = req.get("vectors"), req.get("profiles")
    if vectors is not None and refs:
        raise RequestError("give profiles either as 'vectors' (with <omics> placeholders) or as 'profiles', not both")
    if vectors is not None:
        if not isinstance(vectors, list):
            raise RequestError("'vectors' is a list of profiles (20,029 floats or base64 float32 each)")
        vecs = [_decode(v, i) for i, v in enumerate(vectors)]
        try:
            vecs = list(check_vectors(np.stack(vecs))) if vecs else []
        except ProfileError as e:
            raise RequestError(str(e)) from None
        n = state.count(OMICS)
        if n != len(vecs):
            raise RequestError(f"the state has {n} {OMICS} placeholders for {len(vecs)} vectors")
        return state, vecs, []
    if refs:
        if not isinstance(refs, list):
            raise RequestError("'profiles' is a list of '@sample_id' / cell-line names (or {\"ref\", \"label\"})")
        lines, vecs, notes = [], [], []
        for r in refs:
            ref, label = (r, DEFAULT_LABEL) if isinstance(r, str) else (
                (r.get("ref"), str(r.get("label") or DEFAULT_LABEL)) if isinstance(r, dict) else (None, None))
            try:
                v, what = profiles.reference(ref)
            except ProfileError as e:
                raise RequestError(str(e)) from None
            lines.append(f"{escape_omics(label)}: {OMICS}")
            vecs.append(v)
            notes.append(f"{ref} -> {what}")
        text = escape_omics(state)
        return "\n".join(lines) + ("\n" + text if text else ""), vecs, ["profiles: " + ", ".join(notes)]
    if req.get("resolve_names", True) is False:
        return escape_omics(state), [], []
    text, vecs, notes = profiles.resolve(state)
    return text, vecs, notes


def parse(req, profiles, categories=()):
    """Validated request -> Prepared (without token ids)."""
    if not isinstance(req, dict):
        raise RequestError("the request body must be a JSON object")
    if "questions" not in req:
        raise RequestError("a request needs 'questions' (and usually a 'state')")
    state = req.get("state") or ""
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    qs = req["questions"]
    if not isinstance(qs, dict) or not qs:
        raise RequestError("'questions' is a non-empty object {name: question}")
    texts = [state] + [str(q.get("instructions", "")) for q in qs.values() if isinstance(q, dict)]
    if any(t in x for t in RESERVED for x in texts):
        raise RequestError(f"the request contains a reserved token ({', '.join(RESERVED)}); profiles go in as "
                           f"{OMICS} placeholders, references or a profile list")
    text, vectors, notes = place_profiles(req, state, profiles)
    lang = req.get("lang") or prompt.lang_of(text + " ".join(texts[1:]))
    if lang not in prompt.TEMPLATES:
        raise RequestError("'lang' is 'en' or 'pl'")
    questions = []
    for name, spec in qs.items():
        if not isinstance(spec, dict):
            raise RequestError(f"{name}: a question is an object with type, instructions and criteria")
        prim = TYPE_ALIASES.get(spec.get("type", "choice"), spec.get("type", "choice"))
        if prim not in TYPES:
            raise RequestError(f"{name}: type must be one of {TYPES} (got {prim!r})")
        ins = spec.get("instructions")
        if not isinstance(ins, str) or not ins.strip():
            raise RequestError(f"{name}: instructions (the question text) are required")
        keys, shown = _options(spec, prim, lang, name)
        me = spec.get("max_error")
        if me is not None and me not in (0.01, 0.05):
            raise RequestError(f"{name}: max_error is 0.01 or 0.05 (the certified targets)")
        cat = spec.get("category")
        if cat is not None and cat not in categories:
            raise RequestError(f"{name}: unknown category {cat!r}; one of {sorted(categories)}")
        rank = spec.get("rank", 0)
        if not isinstance(rank, int) or rank < 0:
            raise RequestError(f"{name}: rank is a non-negative integer")
        questions.append(Question(name, prim, escape_omics(ins), keys, shown, me, cat, rank,
                                  bool(spec.get("raw", False))))
    return Prepared(text, lang, vectors, notes, questions)


def build_branches(p, tok, image_token, max_tokens):
    """Token ids of every question x option order."""
    texts = []
    for qi, q in enumerate(p.questions):
        for o in ORDERS:
            perm = permute(len(q.shown), o)
            t = prompt.render(tok, p.state, q.instructions, [q.shown[c] for c in perm], p.lang)
            if t.count(OMICS) != len(p.vectors):
                raise RequestError(f"{t.count(OMICS)} {OMICS} placeholders for {len(p.vectors)} profiles")
            texts.append((qi, o, perm, t.replace(OMICS, OMICS_SPAN)))
    ids = tok([t[3] for t in texts], add_special_tokens=False)["input_ids"]
    for (qi, o, perm, _), x in zip(texts, ids):
        if len(x) > max_tokens:
            raise RequestError(f"{p.questions[qi].name}: the prompt has {len(x)} tokens (limit {max_tokens})")
        if x.count(image_token) != len(p.vectors):
            raise RequestError("the prompt's profile tokens do not match the profiles")
        p.branches.append(Branch(qi, o, perm, x))
    return p


def answers(p, cal, model_name):
    """The response: calibrated probabilities, the answer and confidence of every question."""
    out = {}
    for qi, q in enumerate(p.questions):
        k = len(q.keys)
        rows = []
        for b in p.branches:
            if b.q != qi:
                continue
            canon = np.zeros(k)
            canon[b.perm] = b.probs[:k]
            rows.append(canon / canon.sum())
        raw = np.mean(rows, 0)
        cp = cal.apply(q.prim, raw)
        top = int(np.argmax(cp))
        probs = {key: float(v) for key, v in zip(q.keys, cp)}
        a = {"type": q.prim}
        if q.prim == "choice":
            a.update(choice=q.keys[top], probabilities=probs, confidence=float(cp[top]))
        elif q.prim == "noul":
            a.update(noul=float(cp[0]), probabilities=probs, confidence=float(cp[top]))
        else:
            a.update(score=float(np.dot(np.arange(k), cp)), legend={str(i): s for i, s in enumerate(q.shown)},
                     probabilities=probs, confidence=float(cp[top]), level=top)
        if q.rank:
            order = np.argsort(-cp)[:q.rank]
            a["ranking"] = [{"key": q.keys[i], "probability": float(cp[i])} for i in order]
        if q.max_error is not None:
            cat = q.category or (CATEGORY_BY_TYPE.get(q.prim) if p.vectors else None)
            t = cal.threshold(q.prim, q.max_error, cat)
            if t is None:
                a["decision"] = dict(max_error=q.max_error, certified=False, automated=False,
                                     note="no threshold certified for this type and target: defer")
            else:
                a["decision"] = dict(max_error=q.max_error, certified=True, threshold=t["threshold"],
                                     source=t["source"], automated=bool(cp[top] >= t["threshold"]),
                                     test_error=t["test_error"], test_coverage=t["test_coverage"],
                                     holds_on_test=t["holds_on_test"])
        if q.raw:  # uncalibrated probabilities of each option order, in the canonical order of the keys
            a["raw"] = {o: [float(x) for x in r] for o, r in zip([b.order for b in p.branches if b.q == qi], rows)}
        out[q.name] = a
    return dict(model=model_name, answers=out, profiles=dict(count=len(p.vectors), notes=p.notes))
