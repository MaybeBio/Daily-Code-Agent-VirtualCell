"""Thalamus-Decision from Python, without a server.

    from thalamus_decision import ThalamusDecision
    td = ThalamusDecision.load()                      # ingenix/Thalamus-Decision-9B, or a local folder
    td.ask("Which tissue is this sample from?", ["Liver", "Lung", "Brain"], state="Expression profile: <omics>",
           vectors=[vector])                          # a 20,029-value profile from tr-embed
    td.ask("Is MCF7 oestrogen-receptor positive?", type="noul", state="Cell line: MCF7")   # DepMap profile by name
    td.load_profiles("mine.npz")                      # tr-embed output; then refer to samples as @sample_id

Answers are the engine's answers: calibrated probabilities, the answer, its confidence and, with ``max_error``, whether
to automate the decision or defer it to a person.
"""
import json
from pathlib import Path

from .backend import Backend, warmup
from .calibration import Calibration
from .engine import DecisionEngine
from .hub import model_path
from .profiles import Profiles


def model_name(folder: Path) -> str:
    meta = folder / "decision.json"
    return json.loads(meta.read_text()).get("name", folder.name) if meta.exists() else folder.name


class ThalamusDecision:
    def __init__(self, engine: DecisionEngine):
        self.engine = engine

    @classmethod
    def load(cls, source=None, device="cuda", profiles=(), depmap=True, warm=False) -> "ThalamusDecision":
        folder = model_path(source)
        backend = Backend(folder, device=device)
        engine = DecisionEngine(backend, Calibration(folder / "CALIBRATION.json"), Profiles(folder, depmap=depmap),
                                model_name(folder))
        for f in profiles:
            engine.profiles.load_npz(f)
        if warm:
            warmup(backend)
        return cls(engine)

    @property
    def profiles(self) -> Profiles:
        return self.engine.profiles

    def load_profiles(self, path) -> list[str]:
        """Samples from a tr-embed ``.npz``; refer to them as ``@sample_id``."""
        return self.engine.profiles.load_npz(path)

    def decide(self, state="", questions=None, vectors=None, profiles=None, lang=None, resolve_names=True) -> dict:
        """Several typed questions over one state (the HTTP API's request and response)."""
        req = {"state": state, "questions": questions or {}, "resolve_names": resolve_names}
        if vectors is not None:
            req["vectors"] = [list(map(float, v)) for v in vectors]
        if profiles:
            req["profiles"] = profiles
        if lang:
            req["lang"] = lang
        return self.engine.decide_sync(req)

    def decide_batch(self, requests: list[dict]) -> list[dict]:
        """Many HTTP-API requests at once (shared GPU forwards); answers in request order."""
        return self.engine.decide_batch_sync(requests)

    def ask(self, question, options=None, type="choice", state="", vectors=None, profiles=None, max_error=None,
            **kw) -> dict:
        """One question; returns its answer (choice / noul / score, probabilities, confidence, decision)."""
        q = {"type": type, "instructions": question, **kw}
        if type == "noul":
            if options:
                q["criteria"] = list(options)
        else:
            q["criteria"] = list(options or [])
        if max_error:
            q["max_error"] = max_error
        r = self.decide(state, {"answer": q}, vectors, profiles)
        return dict(r["answers"]["answer"], profiles=r["profiles"])
