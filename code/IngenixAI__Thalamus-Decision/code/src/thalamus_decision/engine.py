"""The decision engine: request -> branches -> backend -> calibrated answers, with batching across requests.

One GPU worker. Requests are parsed, their profiles resolved and their prompts tokenized off the worker (thread pool),
then queued. The worker takes the first queued request and, when the server is under load (the previous call served
more than one request, or more are waiting), waits up to ``batch_window_ms`` for more, within ``max_batch_tokens``;
their branches run in the same packed forwards and every request is answered as soon as its call ends. An idle server
answers at once. Because every forward is padded to the readout's minimum pack and profiles are projected one by one,
batching never changes an answer.
"""
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .backend import warmup
from .requests import answers, build_branches, parse


@dataclass
class _Item:
    jobs: list
    tokens: int
    fut: asyncio.Future
    t_queued: float = field(default_factory=time.perf_counter)
    t_start: float = 0.0


class DecisionEngine:
    def __init__(self, backend, calibration, profiles, model_name, max_tokens=16384, batch_window_ms=3.0,
                 max_batch_tokens=131072, log_path=None):
        self.backend, self.cal, self.profiles = backend, calibration, profiles
        self.model_name, self.max_tokens = model_name, max_tokens
        self.window = batch_window_ms / 1000
        self.max_batch_tokens = max_batch_tokens
        self.log_path = log_path
        self.inflight = self.queued = self.served = self.calls = 0
        self.ready = False
        self._queue = None
        self._worker = None
        self._gpu = ThreadPoolExecutor(1, thread_name_prefix="gpu")
        self._cpu = ThreadPoolExecutor(8, thread_name_prefix="prep")

    async def start(self, log=print):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._gpu, warmup, self.backend, log)
        self._queue = asyncio.Queue()
        self._worker = asyncio.create_task(self._run())
        self.ready = True

    async def stop(self):
        if self._worker:
            self._worker.cancel()

    # ---- one request ------------------------------------------------------------------------------------------------
    def prepare(self, req):
        p = parse(req, self.profiles, self.cal.category_names())
        return build_branches(p, self.backend.tok, self.backend.image_token, self.max_tokens)

    def finish(self, p, probs, t0, t1, queue_ms=0.0):
        for b, pr in zip(p.branches, probs):
            b.probs = pr
        resp = answers(p, self.cal, self.model_name)
        resp["usage"] = dict(input_tokens=sum(len(b.ids) for b in p.branches), output_tokens=0,
                             questions=len(p.questions), branches=len(p.branches), profiles=len(p.vectors),
                             prepare_ms=round((t1 - t0) * 1000, 2), queue_ms=round(queue_ms, 2),
                             latency_ms=round((time.perf_counter() - t0) * 1000, 2))
        return resp

    async def decide(self, req):
        t0 = time.perf_counter()
        self.inflight += 1
        try:
            loop = asyncio.get_running_loop()
            p = await loop.run_in_executor(self._cpu, self.prepare, req)
            jobs = [(b.ids, p.vectors, len(p.questions[b.q].keys)) for b in p.branches]
            t1 = time.perf_counter()
            item = _Item(jobs, sum(len(j[0]) for j in jobs), loop.create_future())
            self.queued += 1
            await self._queue.put(item)
            probs = await item.fut
            resp = self.finish(p, probs, t0, t1, (item.t_start - item.t_queued) * 1000)
            self.served += 1
            if self.log_path:
                self._log(req, resp)
            return resp
        finally:
            self.inflight -= 1

    def decide_sync(self, req):
        """The same answer without the server loop (the Python API)."""
        t0 = time.perf_counter()
        p = self.prepare(req)
        t1 = time.perf_counter()
        probs = self.backend.score([(b.ids, p.vectors, len(p.questions[b.q].keys)) for b in p.branches])
        return self.finish(p, probs, t0, t1)

    def decide_batch_sync(self, reqs, chunk=256):
        """Many requests in shared packed forwards (evaluation); the same answers as one by one, in request order."""
        out = []
        for s in range(0, len(reqs), chunk):
            t0 = time.perf_counter()
            ps = [self.prepare(r) for r in reqs[s:s + chunk]]
            t1 = time.perf_counter()
            probs = self.backend.score([(b.ids, p.vectors, len(p.questions[b.q].keys)) for p in ps for b in p.branches])
            i = 0
            for p in ps:
                out.append(self.finish(p, probs[i:i + len(p.branches)], t0, t1))
                i += len(p.branches)
        return out

    def _log(self, req, resp):
        rec = dict(time=time.time(), request={k: v for k, v in req.items() if k != "vectors"},
                   answers=resp["answers"], profiles=resp["profiles"], usage=resp["usage"])
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- the GPU worker ---------------------------------------------------------------------------------------------
    async def _run(self):
        loop = asyncio.get_running_loop()
        last = 1
        while True:
            first = await self._queue.get()
            batch, tokens = [first], first.tokens
            if last > 1 or not self._queue.empty():
                deadline = loop.time() + self.window
                while tokens < self.max_batch_tokens:
                    try:
                        item = self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        left = deadline - loop.time()
                        if left <= 0:
                            break
                        try:
                            item = await asyncio.wait_for(self._queue.get(), left)
                        except (asyncio.TimeoutError, TimeoutError):
                            break
                    batch.append(item)
                    tokens += item.tokens
            now = time.perf_counter()
            for it in batch:
                it.t_start = now
            self.queued -= len(batch)
            last = len(batch)
            jobs = [j for it in batch for j in it.jobs]
            try:
                probs = await loop.run_in_executor(self._gpu, self.backend.score, jobs)
            except Exception as e:  # every request of the call gets the error (OOM -> 503 in the server)
                for it in batch:
                    if not it.fut.done():
                        it.fut.set_exception(e)
                continue
            self.calls += 1
            i = 0
            for it in batch:
                if not it.fut.done():
                    it.fut.set_result(probs[i:i + len(it.jobs)])
                i += len(it.jobs)

    def health(self):
        return dict(status="ok" if self.ready else "loading", running=self.inflight - self.queued, queued=self.queued,
                    served=self.served, engine_calls=self.calls, slow_forwards=len(getattr(self.backend, "slow", [])))
