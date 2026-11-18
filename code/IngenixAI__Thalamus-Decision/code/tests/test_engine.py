"""CPU tests: profiles, requests, calibration, the HTTP API and MCP, with a fake backend (no GPU, no weights).

They need a model folder without the weights: tokenizer files, config.json, CALIBRATION.json, decision.json and the
preprocessing/ and depmap/ folders (``hf download ingenix/Thalamus-Decision-9B --exclude "*.safetensors"``).

  TD_TEST_MODEL=/path/to/Thalamus-Decision-9B pytest -q
"""
import base64
import io
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

MODEL = Path(os.environ.get("TD_TEST_MODEL", "Thalamus-Decision-9B"))
pytestmark = pytest.mark.skipif(not (MODEL / "tokenizer.json").exists() or not (MODEL / "depmap").is_dir(),
                                reason="set TD_TEST_MODEL to a model folder (weights not needed)")
DIM = 20029


@pytest.fixture(scope="module")
def tok():
    from thalamus_decision.model import load_tokenizer
    return load_tokenizer(MODEL)


@pytest.fixture(scope="module")
def profiles():
    from thalamus_decision.profiles import Profiles
    p = Profiles(MODEL)
    rng = np.random.default_rng(0)
    p.add({"GSM1001": rng.normal(size=DIM), "GSM1004": rng.normal(size=DIM)}, "test", persist=False)
    return p


class FakeBackend:
    """The first presented option gets 0.6, the rest share 0.4; records the profiles it was given."""

    def __init__(self, tok, layout=None):
        from thalamus_decision import prompt
        self.tok, self.table = tok, prompt.code_table(tok, 8)
        self.image_token = tok.convert_tokens_to_ids("<|image_pad|>")
        self.model = SimpleNamespace(packed=False, config={"omics_input_layout": layout} if layout else {})
        self.calls, self.slow, self.seen = [], [], []

    def score(self, jobs):
        self.calls.append(len(jobs))
        out = []
        for ids, vecs, k in jobs:
            assert ids.count(self.image_token) == len(vecs) and all(v.shape == (DIM,) for v in vecs)
            self.seen.append(vecs)
            p = np.full(k, 0.4 / (k - 1))
            p[0] = 0.6
            out.append(p)
        return out


@pytest.fixture()
def engine(tok, profiles):
    from thalamus_decision.calibration import Calibration
    from thalamus_decision.engine import DecisionEngine
    return DecisionEngine(FakeBackend(tok), Calibration(MODEL / "CALIBRATION.json"), profiles, "test-model")


def test_depmap_and_samples_resolve_as_in_thalamus_reasoning(profiles):
    from thalamus_reasoning.chat import Conversation
    from thalamus_reasoning.depmap import DepMap
    text, vecs, notes = profiles.resolve("Compare MCF7 with @GSM1001 and again MCF7.")
    ref = Conversation(lm=None, depmap=DepMap(str(MODEL)), profiles=profiles.samples).prepare(
        "Compare MCF7 with @GSM1001 and again MCF7.")
    assert text == ref.text == "Compare MCF7<omics> with GSM1001<omics> and again MCF7."
    assert len(vecs) == 2 and all(np.array_equal(a, b) for a, b in zip(vecs, ref.vectors))
    row = profiles.depmap.resolver.resolve("MCF7").rows[0]
    assert np.array_equal(vecs[0], np.asarray(profiles.depmap.embeddings[row]))
    assert np.array_equal(vecs[1], profiles.samples["GSM1001"])
    assert any(n.startswith("omics: MCF7") for n in notes)
    # names that are also words need @ or "cells", as in the chat
    assert profiles.resolve("RD is short")[1] == [] and len(profiles.resolve("RD cells")[1]) == 1
    # a literal <omics> typed by a user does not become a profile slot
    assert profiles.resolve("a <omics> here")[1] == []


def test_npz_from_tr_embed(tmp_path):
    from thalamus_decision.profiles import ProfileError, Profiles
    p = Profiles(MODEL, depmap=False, store_dir=tmp_path / "store")
    buf = io.BytesIO()
    np.savez(buf, embeddings=np.ones((2, DIM), np.float32), sample_ids=np.array(["S1", "S2"]))
    assert p.load_npz_bytes(buf.getvalue(), "mine") == ["S1", "S2"]
    assert Profiles(MODEL, depmap=False, store_dir=tmp_path / "store").list_samples()[0]["id"] == "S1"  # kept on disk
    with pytest.raises(ProfileError):
        p.add({"S3": np.ones(DIM + 1)}, "x")  # a vector of another length is refused: profiles come from tr-embed
    with pytest.raises(ProfileError):
        p.add({"bad id": np.ones(DIM)}, "x")


def test_three_ways_to_attach_profiles(profiles, tok):
    from thalamus_decision.requests import RequestError, build_branches, parse
    q = {"t": {"type": "choice", "instructions": "Which tissue?", "criteria": ["breast", "lung"]}}
    p = parse({"state": "Cell line: MCF7", "questions": q}, profiles)
    assert p.state == "Cell line: MCF7<omics>" and len(p.vectors) == 1
    p = parse({"state": "Expression profile: <omics>", "vectors": [[0.0] * DIM], "questions": q}, profiles)
    assert len(p.vectors) == 1 and p.notes == []
    v = base64.b64encode(np.zeros(DIM, "<f4").tobytes()).decode()
    p = parse({"state": "A: <omics>\nB: <omics>", "vectors": [v, v], "questions": q}, profiles)
    assert len(p.vectors) == 2
    with pytest.raises(RequestError):
        parse({"state": "MCF7", "vectors": [[0.0] * DIM], "questions": q}, profiles)  # no placeholder
    p = parse({"profiles": ["@GSM1001", "MCF7"], "state": "Two samples.", "questions": q}, profiles)
    assert p.state == "Expression profile: <omics>\nExpression profile: <omics>\nTwo samples."
    assert np.array_equal(p.vectors[0], profiles.samples["GSM1001"])
    with pytest.raises(RequestError):
        parse({"profiles": ["NOT_A_LINE_X"], "questions": q}, profiles)
    with pytest.raises(RequestError):
        parse({"state": "<|image_pad|>", "questions": q}, profiles)
    p = build_branches(parse({"state": "Cell line: MCF7", "questions": q}, profiles), tok,
                       tok.convert_tokens_to_ids("<|image_pad|>"), 16384)
    assert [b.order for b in p.branches] == ["orig", "rev"] and all(b.ids.count(
        tok.convert_tokens_to_ids("<|image_pad|>")) == 1 for b in p.branches)


def test_only_tr_embed_profiles():
    from thalamus_decision.model import profile_filler
    tr = {"omics_input_dim": 20029, "omics_input_layout": {"scale_indicator": [0, 1], "expression": [1, 19261],
                                                           "geneformer_v2_104m": [19261, 20029]}}
    v = np.random.default_rng(1).normal(size=(3, DIM)).astype(np.float32)
    assert np.array_equal(profile_filler(tr)(v), v)
    with pytest.raises(ValueError):  # a checkpoint with any other input layout is refused
        profile_filler({"omics_input_dim": DIM, "omics_input_layout": {"scale_indicator": [0, 1],
                                                                        "geneformer_v2_104m": [1, 769],
                                                                        "expression": [769, DIM]}})


def test_decide_orders_and_calibration(engine):
    from thalamus_decision.calibration import scale
    r = engine.decide_sync({"state": "Cell line: MCF7", "questions": {
        "t": {"type": "choice", "instructions": "Which tissue?", "criteria": ["breast", "lung", "colon"],
              "max_error": 0.05},
        "er": {"type": "noul", "instructions": "Is it ER positive?"}}})
    # each option is first in one of the two orders: canonical mean (0.6+0.2)/2, 0.2, (0.2+0.6)/2
    raw = np.array([0.4, 0.2, 0.4])
    want = scale(raw, engine.cal.temperature("choice", 3))
    got = np.array(list(r["answers"]["t"]["probabilities"].values()))
    assert np.allclose(got, want)
    assert r["answers"]["t"]["decision"]["source"].startswith(("category", "type"))
    assert np.isclose(r["answers"]["er"]["noul"], 0.5)
    assert r["profiles"]["count"] == 1 and r["usage"]["branches"] == 4


def test_server_and_mcp(engine):
    import asyncio
    from starlette.testclient import TestClient
    from thalamus_decision.server import create_app

    args = SimpleNamespace(max_upload_mb=200, uploads="on", url_file=None, host="127.0.0.1", port=0)
    engine.ready = True
    engine._queue = asyncio.Queue()

    async def start(log=print):  # no warm-up for the fake backend
        engine._queue = asyncio.Queue()
        engine._worker = asyncio.create_task(engine._run())
        engine.ready = True
    engine.start = start
    with TestClient(create_app(engine, args)) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/v1/decide", json={"state": "Line: MCF7", "questions": {
            "t": {"type": "noul", "instructions": "Breast?"}}}).json()
        assert r["profiles"]["count"] == 1 and "t" in r["answers"]
        assert c.post("/v1/decide", json={"questions": {"t": {"type": "bogus", "instructions": "?"}}}).status_code == 422
        buf = io.BytesIO()
        np.savez(buf, embeddings=np.zeros((1, DIM), np.float32), sample_ids=np.array(["MY1"]))
        assert c.post("/v1/profiles", json={"npz": base64.b64encode(buf.getvalue()).decode()}).json() == {"ids": ["MY1"]}
        assert c.get("/v1/profiles/MY1").json()["dim"] == DIM
        assert c.get("/v1/profiles/NOPE").status_code == 404
        assert c.post("/v1/resolve", json={"text": "@MY1 vs HeLa cells"}).json()["profiles"] == 2
        assert any("MCF7" in str(x) for x in c.get("/v1/lines", params={"q": "MCF"}).json())
        r = c.get("/v1/lines", params={"q": "breast", "limit": 200})  # lines with a missing ID or disease
        assert r.status_code == 200 and len(r.json()) > 10
        assert c.get("/v1/models").json()["models"][0]["profiles"]["dim"] == DIM


def test_mcp_tools():
    pytest.importorskip("mcp")
    import asyncio
    from thalamus_decision.mcp_server import build

    async def call(op, payload):
        return {"op": op, **({"answers": {"answer": {"type": "noul"}}, "model": "m", "profiles": {}}
                             if op == "decide" else {})}
    mcp = build(call, local_files=True)
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert names == {"decide", "decide_many", "resolve", "list_profiles", "find_cell_lines", "upload_profiles"}
