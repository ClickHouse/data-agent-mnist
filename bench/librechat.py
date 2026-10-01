"""LibreChat product-surface runner: drive a LibreChat agent and read the run off its trace."""
import hashlib
import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable
from urllib.parse import quote

import httpx

from registry import ENDPOINTS

from .usage import TokenUsage

# The LibreChat runner could not read the agent's trace (ingestion lag, a sessionId
# mismatch, or no reader configured). The run is an infrastructure unknown, not a
# model failure, so the eval loop routes it to an 'error' outcome rather than letting
# an empty result set score as a fail. See 06_eval.score_pair.
ERR_LIBRECHAT_NO_TRACE = "librechat trace unavailable"


# ── LibreChat product-surface runner ────────────────────────────────────────────
# The runners above reimplement the agent; this one drives the product. A LibreChat
# agent answers the SAME question through its own prompt, its own loop and its own
# run_select_query tool, and the trajectory is read back off its Langfuse trace so
# every scoring component downstream stays byte-identical. Only where the run comes
# from changes. The agent's SQL tool must point at the frozen chDB the direct
# runners use (see librechat/mcp_warehouse.py) or ground truth stops applying.

_LIBRECHAT_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                 "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36")
_LIBRECHAT_NO_PARENT = "00000000-0000-0000-0000-000000000000"
# LibreChat suffixes the MCP tool name (e.g. run_select_query_mcp_<server>), so
# match on the substring, as 01_extract does for the source traces.
_SQL_TOOL_SUBSTR = "run_select_query"
# The agent-run trace name, used to pick the right trace out of a conversation. A
# LibreChat conversation also produces a title-generation trace under the same
# sessionId, so matching on the session alone can select the wrong observation set.
# 01_extract keys on the same name; override for a benchmark agent named otherwise.
_LIBRECHAT_TRACE_NAME = os.environ.get("LIBRECHAT_TRACE_NAME", "AgentRun")
# The login JWT expires (~15 min), so a long run must re-login. Refresh well before
# that, and re-login once on any 401. Without this a full-board run 401s after ~15 min.
_LIBRECHAT_TOKEN_TTL = 600


def _parse_sse_event(raw: str):
    """One SSE record (its "event:" / "data:" lines) -> {event, data}, or None."""
    event, data = "message", ""
    for line in re.split(r"\r?\n", raw):
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data += line[len("data:"):].strip()
    if not data:
        return None
    return {"event": event, "data": json.loads(data)}


def _final_answer_text(rmsg: dict) -> str:
    """The assistant's concluding answer from an agents responseMessage.

    On the agents endpoint `text` is empty; the body is in `content`, an ordered list
    of {type: "text"|"tool_call"} blocks spanning the whole agent turn. The answer is
    the text after the last tool call (the concluding step); with no tool call, it is
    all the text. This is the product's own final message, taken off the stream so it
    does not depend on trace ingestion.
    """
    txt = (rmsg.get("text") or "").strip()
    if txt:
        return txt
    content = rmsg.get("content")
    if not isinstance(content, list):
        return ""
    last_tool = max((i for i, b in enumerate(content)
                     if isinstance(b, dict) and b.get("type") == "tool_call"), default=-1)
    parts = [b.get("text", "") for b in content[last_tool + 1:]
             if isinstance(b, dict) and b.get("type") == "text"]
    if not parts:
        parts = [b.get("text", "") for b in content
                 if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(p for p in parts if p).strip()


class _LibreChatSession:
    """One authenticated LibreChat tenant, driving agent chats over the HTTP API.

    Built at import but NOT logged in: importing bench must not require a running
    LibreChat, the same reason the Vertex credentials resolve on first use. login is
    cached and happens on the first run. The flow (login -> POST chat -> GET the SSE
    stream to its final event) mirrors the drive-librechat-agent-chat skill.
    """

    def __init__(self, base_url, tenant_id, email, password,
                 *, provider="gateway", mcp_tool="run_select_query_mcp_ClickHouse",
                 timeout=600.0):
        self.base_url  = (base_url or "").rstrip("/")
        self.tenant_id = tenant_id
        self.email     = email
        self.password  = password
        self.provider  = provider       # the LibreChat endpoint the benchmark agent runs on
        self.mcp_tool  = mcp_tool        # the ClickHouse MCP tool key to attach to the agent
        self._token    = None
        self._token_at = 0.0             # when the token was issued, for TTL refresh
        self._agents   = {}              # (model, instructions) -> created agent id, reused per run
        self._client   = httpx.Client(timeout=timeout)
        self._lock     = threading.Lock()

    @classmethod
    def from_env(cls):
        return cls(
            base_url=os.environ.get("LIBRECHAT_BASE_URL") or ENDPOINTS.get("librechat", ""),
            tenant_id=os.environ.get("LIBRECHAT_TENANT_ID", ""),
            email=os.environ.get("LIBRECHAT_EMAIL", ""),
            password=os.environ.get("LIBRECHAT_PASSWORD", "ValidationPassword123!"),
            provider=os.environ.get("LIBRECHAT_PROVIDER", "gateway"),
            mcp_tool=os.environ.get("LIBRECHAT_MCP_TOOL", "run_select_query_mcp_ClickHouse"))

    def _headers(self, *, json_body=True):
        # X-Tenant-Id only when a tenant is configured: a single-tenant instance
        # rejects an empty tenant header, and the header is meaningless there.
        h = {"User-Agent": _LIBRECHAT_UA}
        if self.tenant_id:
            h["X-Tenant-Id"] = self.tenant_id
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        if json_body:
            h["Content-Type"] = "application/json"
        return h

    def _ensure_login(self, force: bool = False):
        with self._lock:
            fresh = self._token and (time.time() - self._token_at) < _LIBRECHAT_TOKEN_TTL
            if fresh and not force:
                return
            if not (self.base_url and self.email):
                raise RuntimeError(
                    "the LibreChat runner needs LIBRECHAT_BASE_URL and LIBRECHAT_EMAIL set and a "
                    "seeded instance reachable (see librechat/README.md); LIBRECHAT_TENANT_ID is "
                    "only needed for a multi-tenant instance. A provider: librechat model cannot "
                    "run without a reachable instance.")
            r = self._client.post(f"{self.base_url}/api/auth/login",
                                  headers={"User-Agent": _LIBRECHAT_UA, "Content-Type": "application/json",
                                           **({"X-Tenant-Id": self.tenant_id} if self.tenant_id else {})},
                                  json={"email": self.email, "password": self.password})
            r.raise_for_status()
            self._token = r.json()["token"]
            self._token_at = time.time()

    def ensure_agent(self, model: str, instructions: str) -> str:
        """Create (once, then reuse) a benchmark agent and return its id.

        The agent runs `model` on the configured provider, carries the ClickHouse
        MCP tool, and takes `instructions` as its system prompt. The runner passes
        the board's own system prompt here, so the LibreChat candidate answers under
        the SAME prompt the direct runners used on the board: the only variable
        between the two numbers is the agent loop, not the prompt. Created as the
        logged-in user, so it is owned and drivable without a separate grant. Cached
        per (model, instructions) because the prompt is constant across a run.
        """
        self._ensure_login()
        key = (model, hashlib.sha256((instructions or "").encode()).hexdigest())
        with self._lock:
            hit = self._agents.get(key)
        if hit:
            return hit
        body = {"name": f"dam-bench {model}", "instructions": instructions or "",
                "provider": self.provider, "model": model, "model_parameters": {},
                "tools": [self.mcp_tool]}
        r = self._client.post(f"{self.base_url}/api/agents", headers=self._headers(), json=body)
        if r.status_code == 401:                       # token expired mid-run; re-login and retry
            self._ensure_login(force=True)
            r = self._client.post(f"{self.base_url}/api/agents", headers=self._headers(), json=body)
        r.raise_for_status()
        agent_id = r.json()["id"]
        with self._lock:
            self._agents[key] = agent_id
        return agent_id

    def chat(self, text: str, agent_id: str) -> dict:
        """Drive one isolated conversation with the agent to its final message.

        conversationId is null, so every question is its own thread — the
        per-question isolation the direct runners get for free by construction.
        """
        self._ensure_login()
        message_id = str(uuid.uuid4())
        now        = datetime.now(timezone.utc)
        payload = {
            "text": text, "sender": "User", "clientTimestamp": now.isoformat(),
            "isCreatedByUser": True, "parentMessageId": _LIBRECHAT_NO_PARENT,
            "conversationId": None, "messageId": message_id, "error": False,
            "endpoint": "agents", "model": agent_id, "agent_id": agent_id,
            "key": (now + timedelta(hours=1)).isoformat(), "timezone": "UTC",
        }
        start = self._client.post(f"{self.base_url}/api/agents/chat",
                                  headers=self._headers(), json=payload)
        if start.status_code == 401:                   # token expired mid-run; re-login and retry
            self._ensure_login(force=True)
            start = self._client.post(f"{self.base_url}/api/agents/chat",
                                      headers=self._headers(), json=payload)
        start.raise_for_status()
        started = start.json()
        final   = self._read_stream(started["streamId"])
        rmsg    = final.get("responseMessage") or {}
        conv    = (rmsg.get("conversationId")
                   or (final.get("conversation") or {}).get("conversationId")
                   or started.get("conversationId") or started.get("streamId"))
        return {"conversationId": conv,
                "assistantMessageId": rmsg.get("messageId"),
                "final_answer": _final_answer_text(rmsg),
                "served_model": rmsg.get("model") or agent_id}

    def _read_stream(self, stream_id: str) -> dict:
        url = f"{self.base_url}/api/agents/chat/stream/{quote(stream_id, safe='')}"
        with self._client.stream("GET", url, headers=self._headers(json_body=False)) as r:
            r.raise_for_status()
            buffer = ""
            for chunk in r.iter_text():
                buffer += chunk
                records = re.split(r"\r?\n\r?\n", buffer)
                buffer  = records.pop()           # trailing partial record, completed next read
                for raw in records:
                    msg = _parse_sse_event(raw)
                    if not msg:
                        continue
                    if msg["data"].get("final") or msg["event"] == "done":
                        return msg["data"]
                    if msg["data"].get("error"):
                        raise RuntimeError(f"LibreChat stream error: {msg['data']}")
        raise RuntimeError("LibreChat stream ended without a final event")


class _LangfuseReader:
    """Read-only Langfuse REST client for pulling a finished agent run's trace.

    The LibreChat run's trajectory and usage live in the trace LibreChat wrote, not
    in a provider response we hold, so the runner reads them back the way 01_extract
    does (raw /api/public REST rather than the SDK).
    """

    def __init__(self, host, public_key, secret_key, *, timeout=120.0):
        self._client = httpx.Client(base_url=f"{host.rstrip('/')}/api/public",
                                    auth=(public_key, secret_key), timeout=timeout)

    @classmethod
    def from_env(cls):
        host = os.environ.get("LANGFUSE_HOST")
        pk = os.environ.get("LANGFUSE_RESEARCH_PUBLIC_KEY") or os.environ.get("LANGFUSE_PUBLIC_KEY")
        sk = os.environ.get("LANGFUSE_RESEARCH_SECRET_KEY") or os.environ.get("LANGFUSE_SECRET_KEY")
        if not (host and pk and sk):
            return None
        return cls(host, pk, sk)

    def trace_for_conversation(self, conversation_id, *, trace_name=_LIBRECHAT_TRACE_NAME,
                               attempts=24, delay=5.0):
        """Poll for one conversation's agent-run trace, and WAIT FOR INGESTION TO SETTLE
        before returning it. Langfuse ingests a trace incrementally, so a read taken as
        soon as any observation appears catches a PARTIAL trace: turns, sqls and results
        come back undercounted (in the worst case zero generations), which then scores a
        completed, correct run as a fail. So return the trace only once its GENERATION
        count is >0 and unchanged across two consecutive polls and the trace has a
        terminal output, i.e. the last generation has landed. Fall back to the fullest
        trace seen if the budget is spent (better than None), or None if nothing ingested.

        The trace is matched on the sessionId LibreChat sets to the conversationId AND the
        agent-run trace name: a conversation also emits a title-generation trace under the
        same session, and without the name filter that trace, which has generations but no
        run_select_query, could be reconstructed instead.
        """
        params = {"sessionId": conversation_id, "limit": 10}
        if trace_name:
            params["name"] = trace_name
        prev_gen, best = -1, None
        for _ in range(attempts):
            full = None
            for stub in self._client.get("/traces", params=params).json().get("data", []):
                f = self._client.get(f"/traces/{stub['id']}").json()
                if f.get("observations"):
                    full = f
                    break
            if full is not None:
                best = full
                g = sum(1 for o in full.get("observations", []) if o.get("type") == "GENERATION")
                settled = g > 0 and g == prev_gen
                terminal = isinstance(full.get("output"), str) and full["output"].strip()
                if settled and terminal:
                    return full
                prev_gen = g
            time.sleep(delay)
        return best


# Trace-parsing helpers. These mirror 01_extract's, kept separate because that
# module reads Langfuse keys at import and so is not import-safe from bench.
def _lf_kwargs(msg):
    return msg.get("kwargs", {}) if isinstance(msg, dict) else {}


def _lf_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif isinstance(b, str):
                parts.append(b)
        return "\n".join(p for p in parts if p)
    return ""


def _lf_content(msg) -> str:
    if not isinstance(msg, dict):
        return ""
    c = msg.get("content")
    if c is None:
        c = _lf_kwargs(msg).get("content")
    return _lf_text(c)


def _lf_tool_calls(msg):
    """Normalized [{name, args, id}] across the LangChain (args is a dict) and
    OpenAI-style (function.arguments is a JSON string) serializations."""
    k = _lf_kwargs(msg)
    raw = (k.get("tool_calls") or msg.get("tool_calls")
           or k.get("additional_kwargs", {}).get("tool_calls")
           or msg.get("additional_kwargs", {}).get("tool_calls") or [])
    out = []
    for tc in raw:
        name = tc.get("name") or tc.get("function", {}).get("name", "")
        args = tc.get("args")
        if args is None:
            a = tc.get("function", {}).get("arguments")
            try:
                args = json.loads(a) if isinstance(a, str) else (a or {})
            except Exception:
                args = {}
        out.append({"name": name, "args": args or {}, "id": tc.get("id")})
    return out


def _lf_messages(obs):
    inp = obs.get("input")
    if isinstance(inp, dict) and isinstance(inp.get("messages"), list):
        return inp["messages"]
    if isinstance(inp, list):
        return inp
    return []


def _reconstruct_librechat_run(trace) -> dict:
    """Rebuild the standard runner fields from a LibreChat agent's Langfuse trace.

    One conversation is one question, so every run_select_query call in the trace
    belongs to this run. `turns` is the count of model generations, the nearest analog
    to the model-call count the other runners report. The final answer is the
    trace-level output (the streamed answer, when present, is the same text).

    Two tool serializations are handled. The LangGraph agent emits a `tool-dispatch`
    step whose input is the batch of {name, args} calls and whose output.messages hold
    the results in the same order (the role names the tool and there is no tool_call_id
    to key on, so pair positionally). The older tool_batch shape keys results by
    tool_call_id and is kept as a fallback so a differently wired instance still
    reconstructs.
    """
    obs = (trace or {}).get("observations", []) or []
    sqls, sql_results = [], []

    # LangGraph tool-dispatch: results aligned positionally with the call batch.
    for o in obs:
        if o.get("name") != "tool-dispatch":
            continue
        calls = o.get("input") if isinstance(o.get("input"), list) else []
        out   = o.get("output") if isinstance(o.get("output"), dict) else {}
        msgs  = out.get("messages") if isinstance(out.get("messages"), list) else []
        for i, c in enumerate(calls):
            if not isinstance(c, dict) or _SQL_TOOL_SUBSTR not in (c.get("name") or ""):
                continue
            q = (c.get("args") or {}).get("query", "")
            if not q:
                continue
            sqls.append(q)
            sql_results.append(_lf_content(msgs[i]) if i < len(msgs) else "")

    # Fallback: the tool_batch shape, results keyed by tool_call_id.
    if not sqls:
        tool_batches  = [o for o in obs if o.get("type") == "TOOL" and o.get("name") == "tool_batch"]
        results_by_id = {}
        for o in tool_batches:
            out = o.get("output")
            for tm in (out.get("messages", []) if isinstance(out, dict) else []):
                tcid = _lf_kwargs(tm).get("tool_call_id")
                if tcid and not results_by_id.get(tcid):
                    results_by_id[tcid] = _lf_content(tm)
            for m in _lf_messages(o):
                tcid = _lf_kwargs(m).get("tool_call_id")
                if tcid and not results_by_id.get(tcid):
                    results_by_id[tcid] = _lf_content(m)
        scan_msgs = list(max((_lf_messages(o) for o in tool_batches), key=len, default=[]))
        for o in obs:
            if o.get("type") == "GENERATION" and isinstance(o.get("output"), dict):
                scan_msgs.append(o["output"])
        seen = set()
        for m in scan_msgs:
            for tc in _lf_tool_calls(m):
                if _SQL_TOOL_SUBSTR not in (tc["name"] or ""):
                    continue
                q, tcid = tc["args"].get("query", ""), tc.get("id")
                if not q or (tcid and tcid in seen):
                    continue
                if tcid:
                    seen.add(tcid)
                sqls.append(q)
                sql_results.append(results_by_id.get(tcid, ""))

    gens = sorted((o for o in obs if o.get("type") == "GENERATION"),
                  key=lambda o: o.get("startTime") or "")
    usage = TokenUsage()
    for g in gens:
        usage.add_langfuse(g.get("usageDetails") or g.get("usage"))

    # Final answer: the trace-level output (the assistant's last message). Fall back
    # to the last generation's text for a shape that carries no trace output.
    fa = trace.get("output") if isinstance(trace, dict) else None
    if isinstance(fa, str) and fa.strip():
        final_answer = fa.strip()
    elif gens:
        out = gens[-1].get("output")
        final_answer = (_lf_content(out) if isinstance(out, dict) else _lf_text(out)).strip()
    else:
        final_answer = ""

    return {"sqls": sqls, "sql_results": sql_results, "final_answer": final_answer,
            "turns": len(gens), "usage": usage}


# Built at import, never logged in / never a network call here; see the classes above.
_librechat_session = _LibreChatSession.from_env()
_langfuse_reader   = _LangfuseReader.from_env()


def run_candidate_librechat(
    nl_question: str,
    model_id: str,
    ch_query: Callable[[str], str] | None = None,
    system_prompt: str | None = None,
    *,
    session=None,
    reader=None,
) -> dict:
    """Drive a LibreChat agent as the candidate and read the run off its trace.

    `model_id` is the model the LibreChat agent runs (served by the configured
    provider); `system_prompt` is the board's own prompt, applied as the agent's
    instructions so the LibreChat candidate answers under the SAME prompt the direct
    runners used on the board. The only variable between the two numbers is the agent
    loop, not the prompt. `ch_query` is unused: the agent reaches the warehouse through
    its own run_select_query MCP tool, which must point at the same frozen chDB (see
    librechat/mcp_warehouse.py).

    The returned dict is the standard runner shape, so scoring is unchanged. sqls,
    sql_results, turns and usage come from the Langfuse trace because they are the
    product's numbers, read back rather than measured by us. How `turns` (model
    generations) relates to pass@B is the open question the spike measures.
    """
    sess   = session or _librechat_session
    reader = reader if reader is not None else _langfuse_reader
    start  = time.time()

    agent_id = sess.ensure_agent(model_id, system_prompt or "")
    conv     = sess.chat(nl_question, agent_id)
    trace   = reader.trace_for_conversation(conv["conversationId"]) if reader else None
    recon   = _reconstruct_librechat_run(trace)
    # The streamed answer does not depend on trace ingestion having completed, and it
    # is the same text as the reconstructed one, so prefer it.
    final_answer = conv.get("final_answer") or recon["final_answer"]
    return {
        "sqls": recon["sqls"], "sql_results": recon["sql_results"],
        "final_answer": final_answer, "turns": recon["turns"],
        "served_model": conv.get("served_model") or model_id,
        "latency": round(time.time() - start, 2),
        # An absent trace means ingestion lag or a session-id mismatch, not a model
        # failure: flagged so the eval loop scores it 'error' (unknown), not a fail.
        "error": None if trace is not None else ERR_LIBRECHAT_NO_TRACE,
        "usage": recon["usage"].as_dict(),
        "conversation_id": conv.get("conversationId"),
    }
