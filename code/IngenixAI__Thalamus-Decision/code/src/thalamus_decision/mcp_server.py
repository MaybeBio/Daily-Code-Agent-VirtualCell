"""MCP: Thalamus-Decision decisions as tools for agents (streamable HTTP at /mcp, stateless; or a stdio bridge).

Tools (read-only except ``upload_profiles``):
  * decide           one choice / noul / score question about a state, with profiles named in it;
  * decide_many      several typed questions over one state in one pass (the HTTP API request);
  * resolve          which samples and cell lines a text names (the chat's notes), without a decision;
  * list_profiles    the samples loaded on the server;
  * find_cell_lines  DepMap cell lines with a profile, by name, tissue or disease;
  * upload_profiles  (stdio bridge only) upload a tr-embed .npz from your machine; its samples become @sample_id.

``td-serve --mcp on`` serves the tools in-process; ``td-mcp --url URL`` is a stdio bridge for hosts that start MCP
servers as local processes, forwarding to a running server's HTTP API.
"""
import argparse
import json

INSTRUCTIONS = """Thalamus-Decision answers typed decisions (choice among named options, yes/no, ordered score) about a
state: text, a JSON record and/or expression profiles. Rules:
- Name profiles in the state text: @sample_id for samples loaded on the server (list_profiles), DepMap cell lines by name
  (MCF7, HCT 116, ACH-000019; @RD or "RD cells" for names that are also words). Use resolve to check what a text names.
- One condition per yes/no question; offer "cannot be determined" when the state may not contain the answer.
- Put facts and rules in the state; the model decides from what it reads.
- Probabilities are calibrated. Pass max_error 0.05 (or 0.01) to get decision.automated: false means defer to a person.
- Ask several questions about the same state with decide_many (one pass)."""


def build(call, host_is_local=True, log_level="WARNING", local_files=False):
    """FastMCP server whose tools use ``call(op, payload)`` -> dict."""
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.exceptions import ToolError
    from mcp.server.transport_security import TransportSecuritySettings
    sec = TransportSecuritySettings(enable_dns_rebinding_protection=host_is_local,
                                    allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
                                    allowed_origins=["http://127.0.0.1:*", "http://localhost:*"])
    mcp = FastMCP("thalamus-decision", instructions=INSTRUCTIONS, stateless_http=True, json_response=True,
                  transport_security=sec, log_level=log_level)

    async def run(op, payload):
        try:
            return await call(op, payload)
        except ValueError as e:
            raise ToolError(str(e)) from None

    @mcp.tool(description="Answer one typed question about a state. type: choice (options = the candidate answers, "
                          "2-255), noul (yes/no; options optional: [what yes means, what no means]) or score (options = "
                          "levels, lowest first). Profiles: name them in the state (@sample_id, or a DepMap cell-line "
                          "name such as MCF7), or list them in profiles (each becomes a line 'Expression profile: ...' "
                          "before the state). max_error 0.05 / 0.01 adds a decision (automate or defer).")
    async def decide(question: str, state: str = "", options: list[str] | None = None, type: str = "choice",
                     profiles: list[str] | None = None, max_error: float | None = None, category: str | None = None,
                     rank: int = 0, lang: str | None = None) -> dict:
        q = {"type": type, "instructions": question}
        if type == "noul":
            if options:
                if len(options) != 2:
                    raise ToolError("noul options are [what yes means, what no means]")
                q["criteria"] = {"true": options[0], "false": options[1]}
        else:
            q["criteria"] = options or []
        for k, v in (("max_error", max_error), ("category", category), ("rank", rank)):
            if v:
                q[k] = v
        req = {"state": state, "questions": {"answer": q}}
        if profiles:
            req["profiles"] = profiles
        if lang:
            req["lang"] = lang
        r = await run("decide", req)
        return dict(r["answers"]["answer"], model=r["model"], profiles=r["profiles"])

    @mcp.tool(description="Several typed questions over one state in one pass. questions maps a name to a question "
                          "object: {\"type\": \"choice\", \"instructions\": question text, \"criteria\": [option, ...] or "
                          "{key: description}} | {\"type\": \"noul\" (yes/no), \"instructions\": ..., \"criteria\" "
                          "(optional): {\"true\": what yes means, \"false\": what no means}} | {\"type\": \"score\", "
                          "\"instructions\": ..., \"criteria\": [lowest level, ..., highest level]}; optional on each: "
                          "\"max_error\": 0.05 or 0.01, \"rank\": k. Profiles as in decide (named in the state, or the "
                          "profiles list). Returns {model, answers: {name: answer}, profiles, usage}.")
    async def decide_many(questions: dict, state: str | dict = "", profiles: list | None = None,
                          lang: str | None = None) -> dict:
        req = {"state": state, "questions": questions}
        if profiles:
            req["profiles"] = profiles
        if lang:
            req["lang"] = lang
        return await run("decide", req)

    @mcp.tool(description="Show which samples (@sample_id) and DepMap cell lines a text names and how many profiles "
                          "it attaches, without making a decision.")
    async def resolve(text: str) -> dict:
        return await run("resolve", {"text": text})

    @mcp.tool(description="List the samples loaded on the server (refer to them as @sample_id).")
    async def list_profiles(prefix: str = "", limit: int = 100) -> list:
        return await run("list_profiles", {"prefix": prefix, "limit": limit})

    @mcp.tool(description="Find DepMap cell lines with a profile; query matches name, IDs, tissue or disease.")
    async def find_cell_lines(query: str = "", limit: int = 30) -> list:
        return await run("find_cell_lines", {"query": query, "limit": limit})

    if local_files:
        @mcp.tool(description="Upload a .npz written by tr-embed (embeddings + sample_ids) from this machine; its "
                              "samples can then be named as @sample_id.")
        async def upload_profiles(path: str) -> dict:
            return await run("upload", {"path": path})

    return mcp


def main(argv=None):
    """Stdio bridge: an MCP server on stdin/stdout whose tools call a running td-serve over HTTP."""
    import base64
    import logging
    from pathlib import Path

    import httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--timeout", type=float, default=300)
    a = ap.parse_args(argv)
    base = a.url.rstrip("/")

    async def call(op, payload):
        async with httpx.AsyncClient(timeout=a.timeout) as c:
            if op == "decide":
                r = await c.post(f"{base}/v1/decide", json=payload)
            elif op == "resolve":
                r = await c.post(f"{base}/v1/resolve", json=payload)
            elif op == "list_profiles":
                r = await c.get(f"{base}/v1/profiles", params=payload)
            elif op == "find_cell_lines":
                r = await c.get(f"{base}/v1/lines", params={"q": payload["query"], "limit": payload["limit"]})
            elif op == "upload":
                try:
                    data = Path(payload["path"]).expanduser().read_bytes()
                except OSError as e:
                    raise ValueError(f"cannot read {payload['path']}: {e.strerror or e}") from None
                r = await c.post(f"{base}/v1/profiles",
                                 json={"npz": base64.b64encode(data).decode(), "name": Path(payload["path"]).stem})
            else:
                raise ValueError(op)
        body = r.json()
        if r.status_code != 200:
            raise ValueError(body.get("error", json.dumps(body)) if isinstance(body, dict) else str(body))
        return body

    build(call, local_files=True).run("stdio")


if __name__ == "__main__":
    main()
