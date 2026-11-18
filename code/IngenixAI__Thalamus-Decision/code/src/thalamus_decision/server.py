"""td-serve: the HTTP API (and MCP) of the Thalamus-Decision engine.

  POST /v1/decide            typed decisions (state, questions; profiles by reference, as vectors or as a list)
  POST /v1/resolve           how a text's @sample and cell-line references resolve (the chat's notes), without a decision
  GET  /v1/models            the served model, calibration file and categories
  GET  /health               {"status", "running", "queued", ...} (status "loading" until the warm-up ends)
  GET  /v1/profiles          your samples (?prefix=&limit=)
  GET  /v1/profiles/{id}     one sample: origin, kind, size
  POST /v1/profiles          add samples: {"npz": base64 of a tr-embed .npz} | {"vectors": {id: floats or base64}} |
                             {"table": base64, "filename": "x.tsv.gz", ...tr-embed options} (with --embed on)
  GET  /v1/lines             DepMap cell lines with a profile (?q=breast&limit=)
  /mcp                       MCP (streamable HTTP, --mcp on)

Errors: {"error": "..."} with 422 (invalid request), 404 (unknown sample), 413 (body too large), 503 (GPU out of
memory, or still loading) and 500 (anything else).

  td-serve --model ingenix/Thalamus-Decision-9B --profiles mine.npz --port 8000
"""
import argparse
import asyncio
import base64
import contextlib
import logging
import socket
import sys
import time
import traceback
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from . import __version__
from .backend import EngineOOM
from .profiles import ProfileError
from .requests import RequestError

log = logging.getLogger("thalamus_decision")


def _err(status, msg):
    return JSONResponse({"error": msg}, status_code=status)


def add_profiles(profiles, body):
    """POST /v1/profiles: samples from a tr-embed .npz, raw vectors, or a table embedded with tr-embed here."""
    if not isinstance(body, dict):
        raise ProfileError("the body is a JSON object")
    if "npz" in body:
        ids = profiles.load_npz_bytes(base64.b64decode(body["npz"]), str(body.get("name") or "upload"))
        return {"ids": ids}
    if "vectors" in body:
        vecs = body["vectors"]
        if not isinstance(vecs, dict) or not vecs:
            raise ProfileError("'vectors' is {sample_id: 20,029 floats or base64 float32}")
        import numpy as np
        decoded = {k: (np.frombuffer(base64.b64decode(v), dtype="<f4") if isinstance(v, str) else np.asarray(v, float))
                   for k, v in vecs.items()}
        return {"ids": profiles.add(decoded, str(body.get("name") or "upload"))}
    if "table" in body:
        opts = {k: body[k] for k in ("kind", "values", "pseudobulk_by", "orientation", "layer", "gene_column")
                if body.get(k) is not None}
        ids, report = profiles.embed_table(base64.b64decode(body["table"]), str(body.get("filename") or "table.tsv"),
                                           **opts)
        return {"ids": ids, "report": report}
    raise ProfileError("give 'npz' (a tr-embed .npz), 'vectors' ({id: vector}) or 'table' (an expression table)")


def create_app(engine, args, mcp=None):
    max_body = int(args.max_upload_mb * 1024 * 1024)

    async def body_of(request):
        if int(request.headers.get("content-length") or 0) > max_body:
            raise ValueError("too large")
        return await request.json()

    async def decide(request: Request):
        if not engine.ready:
            return _err(503, "the model is loading")
        try:
            body = await body_of(request)
        except ValueError:
            return _err(413, f"the request is larger than {args.max_upload_mb} MB")
        except Exception:
            return _err(422, "the request body is not valid JSON")
        try:
            return JSONResponse(await engine.decide(body))
        except (RequestError, ProfileError) as e:
            return _err(422, str(e))
        except EngineOOM as e:
            return _err(503, str(e))
        except Exception as e:
            log.error("decide failed: %s\n%s", e, traceback.format_exc())
            return _err(500, f"{type(e).__name__}: {e}")

    async def resolve(request: Request):
        try:
            body = await request.json()
            text, vectors, notes = await asyncio.to_thread(engine.profiles.resolve, str(body["text"]))
            return JSONResponse({"text": text, "profiles": len(vectors), "notes": notes})
        except (KeyError, TypeError, ValueError):
            return _err(422, "the body is {\"text\": \"...\"}")

    async def models(request: Request):
        b = engine.backend
        return JSONResponse({"models": [dict(
            name=engine.model_name, engine=f"thalamus-decision {__version__}",
            description="Thalamus-Decision: calibrated typed decisions (choice / noul / score) about text and "
                        "expression profiles",
            orders=2, max_options=255, max_tokens=engine.max_tokens, packed=b.model.packed,
            calibration=engine.cal.summary(),
            profiles=dict(dim=20029, made_with="tr-embed (thalamus-reasoning)", samples=len(engine.profiles.samples),
                          depmap_lines=0 if engine.profiles.depmap is None else int(
                              (engine.profiles.depmap.table.embedding_row >= 0).sum()),
                          uploads=args.uploads == "on",
                          embed=engine.profiles.embed_enabled),
        )]})

    async def health(request: Request):
        return JSONResponse(engine.health(), status_code=200 if engine.ready else 503)

    async def profiles_list(request: Request):
        q = request.query_params
        return JSONResponse(engine.profiles.list_samples(q.get("prefix", ""), int(q.get("limit", 1000))))

    async def profiles_get(request: Request):
        try:
            return JSONResponse(engine.profiles.info(request.path_params["sid"]))
        except ProfileError as e:
            return _err(404, str(e))

    async def profiles_put(request: Request):
        if args.uploads != "on":
            return _err(422, "uploads are disabled on this server")
        try:
            body = await body_of(request)
        except ValueError:
            return _err(413, f"the upload is larger than {args.max_upload_mb} MB")
        except Exception:
            return _err(422, "the request body is not valid JSON")
        try:
            return JSONResponse(await asyncio.to_thread(add_profiles, engine.profiles, body))
        except (ProfileError, ValueError, TypeError) as e:
            return _err(422, str(e))

    async def lines(request: Request):
        q = request.query_params
        return JSONResponse(engine.profiles.lines(q.get("q", ""), int(q.get("limit", 50))))

    routes = [Route("/v1/decide", decide, methods=["POST"]), Route("/v1/resolve", resolve, methods=["POST"]),
              Route("/v1/models", models, methods=["GET"]), Route("/health", health, methods=["GET"]),
              Route("/v1/profiles", profiles_list, methods=["GET"]), Route("/v1/profiles", profiles_put, methods=["POST"]),
              Route("/v1/profiles/{sid:path}", profiles_get, methods=["GET"]), Route("/v1/lines", lines, methods=["GET"])]
    if mcp is not None:
        routes.append(Mount("/", app=mcp.streamable_http_app()))

    @contextlib.asynccontextmanager
    async def lifespan(app):
        await engine.start(log=lambda m: print(m, flush=True))
        if args.url_file:
            host = socket.gethostname() if args.host in ("0.0.0.0", "::") else args.host
            Path(args.url_file).write_text(f"http://{host}:{args.port}\n")
        print(f"READY http://{socket.gethostname()}:{args.port} ({engine.model_name}, calibration "
              f"{engine.cal.path.name}, {len(engine.profiles.samples)} samples)", flush=True)
        async with (mcp.session_manager.run() if mcp is not None else contextlib.nullcontext()):
            yield
        await engine.stop()

    return Starlette(routes=routes, lifespan=lifespan)


def build_engine(a):
    from .api import model_name
    from .backend import Backend
    from .calibration import Calibration
    from .engine import DecisionEngine
    from .hub import model_path
    from .profiles import Profiles
    folder = model_path(a.model)
    name = a.name or model_name(folder)
    t0 = time.perf_counter()
    backend = Backend(folder, device=a.device, max_rows=a.max_rows, max_tokens=a.max_forward_tokens)
    print(f"loaded {name} from {folder} in {time.perf_counter() - t0:.0f}s (packed readout: {backend.model.packed})",
          flush=True)
    profiles = Profiles(folder, depmap=a.depmap == "on", store_dir=a.profile_store, embed=a.embed == "on",
                        device=a.device)
    for f in a.profiles:
        print(f"loaded {len(profiles.load_npz(f))} samples from {f}", flush=True)
    return DecisionEngine(backend, Calibration(folder / "CALIBRATION.json"), profiles, name, max_tokens=a.max_tokens,
                          batch_window_ms=a.batch_window_ms, max_batch_tokens=a.max_batch_tokens,
                          log_path=a.log_decisions)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Thalamus-Decision server (HTTP API + MCP)")
    ap.add_argument("--model", default=None, help="local model folder or Hub repo id (default: $THALAMUS_DECISION_HOME "
                                                  "or ingenix/Thalamus-Decision-9B)")
    ap.add_argument("--name", default=None, help="model name in responses (default: decision.json)")
    ap.add_argument("--profiles", nargs="*", default=[], help=".npz files from tr-embed (samples as @sample_id)")
    ap.add_argument("--profile-store", default=None, help="folder that keeps uploaded samples across restarts")
    ap.add_argument("--uploads", default="on", choices=["on", "off"], help="POST /v1/profiles")
    ap.add_argument("--embed", default="off", choices=["on", "off"],
                    help="run tr-embed on uploaded expression tables (downloads Geneformer-V2-104M once)")
    ap.add_argument("--depmap", default="on", choices=["on", "off"], help="resolve DepMap cell-line names")
    ap.add_argument("--max-upload-mb", type=float, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-tokens", type=int, default=16384, help="longest prompt (tokens) per question")
    ap.add_argument("--max-rows", type=int, default=64, help="prompts per packed forward")
    ap.add_argument("--max-forward-tokens", type=int, default=65536, help="tokens per packed forward")
    ap.add_argument("--max-batch-tokens", type=int, default=131072, help="tokens of one engine call")
    ap.add_argument("--batch-window-ms", type=float, default=3.0)
    ap.add_argument("--log-decisions", default=None, help="append one JSON line per request")
    ap.add_argument("--mcp", default="on", choices=["on", "off"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--url-file", default=None, help="write the server URL here once it accepts requests")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    for noisy in ("httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    engine = build_engine(a)
    mcp = None
    if a.mcp == "on":
        from .mcp_server import build

        async def call(op, payload):
            try:
                if op == "decide":
                    return await engine.decide(payload)
                if op == "resolve":
                    text, vectors, notes = await asyncio.to_thread(engine.profiles.resolve, payload["text"])
                    return {"text": text, "profiles": len(vectors), "notes": notes}
                if op == "list_profiles":
                    return engine.profiles.list_samples(payload.get("prefix", ""), int(payload.get("limit", 100)))
                if op == "find_cell_lines":
                    return engine.profiles.lines(payload.get("query", ""), int(payload.get("limit", 30)))
            except (RequestError, ProfileError) as e:
                raise ValueError(str(e)) from None
            raise ValueError(op)
        mcp = build(call, host_is_local=a.host in ("127.0.0.1", "localhost", "::1"))
    import uvicorn
    uvicorn.run(create_app(engine, a, mcp), host=a.host, port=a.port, log_level="warning", timeout_keep_alive=30)


if __name__ == "__main__":
    main()
