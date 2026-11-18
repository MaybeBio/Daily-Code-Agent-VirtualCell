"""A small client for td-serve (standard library only: copy this file anywhere).

  from client import ThalamusDecisionClient           # thalamus_decision/client.py
  td = ThalamusDecisionClient("http://127.0.0.1:8000")
  td.ask("Is this cell line oestrogen-receptor positive?", type="noul", state="Cell line: MCF7")   # DepMap by name
  td.upload_npz("mine.npz")                            # samples from tr-embed, then refer to them as @sample_id
  td.decide(state="Sample A: @GSM1001\nSample B: @GSM1004",
            questions={"ifn": {"type": "choice", "instructions": "Which sample shows the stronger interferon response?",
                               "criteria": ["Sample A", "Sample B"]}})

  python client.py --url http://HOST:PORT demo          # example requests with DepMap cell lines
  python client.py --url http://HOST:PORT ask "Is MCF7 oestrogen-receptor positive?" --type noul --state "Cell line: MCF7"
"""
import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


class ThalamusDecisionError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class ThalamusDecisionClient:
    def __init__(self, url="http://127.0.0.1:8000", timeout=300):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _call(self, method, path, body=None, params=None):
        url = self.url + path
        if params:
            url += "?" + "&".join(f"{k}={urllib.request.quote(str(v))}" for k, v in params.items())
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error", "")
            except Exception:
                msg = e.reason
            raise ThalamusDecisionError(e.code, msg) from None

    # ---- decisions ----------------------------------------------------------------------------------------------------
    def decide(self, state="", questions=None, vectors=None, profiles=None, lang=None, resolve_names=True):
        """The full API: several typed questions over one state (text or a JSON object) and its profiles."""
        body = {"state": state, "questions": questions or {}}
        if vectors is not None:
            body["vectors"] = [_b64(v) for v in vectors]
        if profiles:
            body["profiles"] = profiles
        if lang:
            body["lang"] = lang
        if not resolve_names:
            body["resolve_names"] = False
        return self._call("POST", "/v1/decide", body)

    def ask(self, question, options=None, type="choice", state="", vectors=None, profiles=None, max_error=None,
            rank=0, **kw):
        """One question; returns its answer (choice / noul / score, probabilities, confidence, decision)."""
        q = {"type": type, "instructions": question, **kw}
        if type == "noul":
            if options:
                q["criteria"] = list(options)
        else:
            q["criteria"] = list(options or [])
        if max_error:
            q["max_error"] = max_error
        if rank:
            q["rank"] = rank
        r = self.decide(state, {"answer": q}, vectors, profiles)
        return dict(r["answers"]["answer"], profiles=r["profiles"])

    def resolve(self, text):
        """Which samples and cell lines a text names (the chat's notes), without a decision."""
        return self._call("POST", "/v1/resolve", {"text": text})

    # ---- profiles -----------------------------------------------------------------------------------------------------
    def upload_npz(self, path, name=None):
        """Upload a .npz written by tr-embed; returns {"ids": [...]} (refer to them as @sample_id)."""
        data = Path(path).read_bytes()
        return self._call("POST", "/v1/profiles", {"npz": base64.b64encode(data).decode(), "name": name or Path(path).stem})

    def upload_vectors(self, vectors, name=None):
        """Upload {sample_id: 20,029 floats} (e.g. rows of a tr-embed .npz)."""
        return self._call("POST", "/v1/profiles", {"vectors": {k: _b64(v) for k, v in vectors.items()}, "name": name})

    def embed_table(self, path, **options):
        """Send an expression table or .h5ad to a server started with --embed on; tr-embed runs there. Options:
        kind, values, pseudobulk_by, orientation, layer, gene_column (as tr-embed's flags)."""
        data = Path(path).read_bytes()
        return self._call("POST", "/v1/profiles", {"table": base64.b64encode(data).decode(),
                                                   "filename": Path(path).name, **options})

    def list_profiles(self, prefix="", limit=1000):
        return self._call("GET", "/v1/profiles", params={"prefix": prefix, "limit": limit})

    def profile_info(self, sample_id):
        return self._call("GET", f"/v1/profiles/{sample_id}")

    def lines(self, query="", limit=50):
        """DepMap cell lines with a profile (name, IDs, tissue, disease)."""
        return self._call("GET", "/v1/lines", params={"q": query, "limit": limit})

    def models(self):
        return self._call("GET", "/v1/models")

    def health(self):
        return self._call("GET", "/health")


def _b64(vector):
    """base64 of little-endian float32 values."""
    import array
    buf = array.array("f", [float(x) for x in vector])
    if sys.byteorder != "little":
        buf.byteswap()
    return base64.b64encode(buf.tobytes()).decode()


def _show(title, req, resp):
    print(f"\n=== {title}\n--- request\n{json.dumps(req, ensure_ascii=False, indent=1)[:1500]}")
    r = json.loads(json.dumps(resp))
    for a in r.get("answers", {}).values():
        p = a.get("probabilities", {})
        if len(p) > 6:  # long lists: show the five most probable
            a["probabilities"] = dict(sorted(p.items(), key=lambda kv: -kv[1])[:5]) | {"...": f"{len(p) - 5} more"}
    print(f"--- response\n{json.dumps(r, ensure_ascii=False, indent=1)[:3000]}")


def demo(td):
    """Example requests with DepMap cell lines (every server has them) and text only."""
    print(json.dumps(td.models(), indent=1)[:1500])

    # 1. a cell line named in the state gets its DepMap profile (as in tr-chat)
    req = {"state": "Cell line: MCF7", "questions": {"tissue": {
        "type": "choice", "instructions": "Which tissue does this cancer cell line come from?",
        "criteria": ["breast", "lung", "colon", "skin", "blood"], "max_error": 0.05}}}
    _show("1. a cell line by name (DepMap profile attached)", req, td._call("POST", "/v1/decide", req))

    # 2. two cell lines compared in one state; several typed questions in one request
    req = {"state": "Line A: MCF7\nLine B: MDA-MB-231",
           "questions": {
               "mesenchymal": {"type": "choice", "instructions": "Which line has the more mesenchymal expression program?",
                               "criteria": ["Line A", "Line B"]},
               "er": {"type": "noul", "instructions": "Is line A oestrogen-receptor positive?"}}}
    _show("2. two cell lines, two questions", req, td._call("POST", "/v1/decide", req))

    # 3. the profile list: each profile on a line 'Expression profile: <omics>' before the state
    req = {"profiles": ["HCT 116"], "state": "", "questions": {"cancer": {
        "type": "choice", "instructions": "What type of cancer is this cell line derived from?",
        "criteria": ["colorectal carcinoma", "lung adenocarcinoma", "melanoma", "breast carcinoma"]}}}
    _show("3. a profile list", req, td._call("POST", "/v1/decide", req))

    # 4. text only: a rule applied to a JSON record
    req = {"state": {"patient": "female, 67", "eGFR_ml_min_1_73m2": 38,
                     "rules": "G1 >= 90; G2 60-89; G3a 45-59; G3b 30-44; G4 15-29; G5 < 15"},
           "questions": {"ckd": {"type": "score", "instructions": "What is the CKD stage?",
                                 "criteria": ["G1", "G2", "G3a", "G3b", "G4", "G5"]}}}
    _show("4. text only: a rule in a JSON state", req, td._call("POST", "/v1/decide", req))

    # 5. what a text names, without a decision
    print("\n=== 5. resolve\n", json.dumps(td.resolve("Compare HeLa cells with RD cells and with @GSM1001."), indent=1))
    print("\nhealth:", td.health())


def main(argv=None):
    ap = argparse.ArgumentParser(description="Thalamus-Decision client")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo")
    sub.add_parser("models")
    sub.add_parser("health")
    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("--options", nargs="*", default=None)
    p.add_argument("--type", default="choice", choices=["choice", "noul", "score"])
    p.add_argument("--state", default="")
    p.add_argument("--profiles", nargs="*", default=None)
    p.add_argument("--max_error", type=float, default=None)
    p = sub.add_parser("upload")
    p.add_argument("npz")
    p = sub.add_parser("profiles")
    p.add_argument("--prefix", default="")
    p = sub.add_parser("lines")
    p.add_argument("query", nargs="?", default="")
    p = sub.add_parser("resolve")
    p.add_argument("text")
    a = ap.parse_args(argv)
    td = ThalamusDecisionClient(a.url)
    if a.cmd == "demo":
        demo(td)
    elif a.cmd == "models":
        print(json.dumps(td.models(), indent=1))
    elif a.cmd == "health":
        print(json.dumps(td.health()))
    elif a.cmd == "ask":
        print(json.dumps(td.ask(a.question, a.options, a.type, a.state, profiles=a.profiles, max_error=a.max_error),
                         indent=1))
    elif a.cmd == "upload":
        print(json.dumps(td.upload_npz(a.npz)))
    elif a.cmd == "profiles":
        print(json.dumps(td.list_profiles(a.prefix), indent=1))
    elif a.cmd == "lines":
        print(json.dumps(td.lines(a.query), indent=1))
    elif a.cmd == "resolve":
        print(json.dumps(td.resolve(a.text), indent=1))


if __name__ == "__main__":
    main()
