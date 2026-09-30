"""Guards for the LibreChat product-surface runner.

The runner drives a real LibreChat agent and reads the run back off its Langfuse
trace, so scoring stays identical while the trajectory comes from the product
instead of our loop. None of these tests reaches a network: the HTTP session and
the Langfuse reader are stubbed, so what is pinned is the parts that decide whether
the invariant holds.

  RECONSTRUCTION IS THE INVARIANT. Scoring consumes `sql_results` (the result-set
  list) and `final_answer`; if either is rebuilt wrong from the trace, the numbers
  stop being comparable to the board. The trace shape here mirrors a real LibreChat
  AgentRun (LangChain tool_batch observations plus OpenAI-style tool_calls).

  A MISSING TRACE IS FLAGGED, NOT ZEROED. Langfuse ingestion is async, so a trace
  can be absent when the run is not: that must surface as an error the consumer
  reads as "unknown", never as an empty-but-successful run.

  THE STANDARD DICT SHAPE. run_candidate dispatches on registry membership and the
  eval loop reads fixed keys off the returned dict; a librechat entry has to route
  here and come back in the same shape as every other runner.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

DAM = Path(__file__).resolve().parent.parent
os.environ.setdefault("DAM_DATA_ROOT", "/tmp/dam-librechat-test")
os.environ.setdefault("DAM_MODELS_CONFIG", str(DAM / "config" / "models.example.yaml"))
for _var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "FIREWORKS_API_KEY"):
    os.environ.setdefault(_var, "placeholder")
sys.path.insert(0, str(DAM))

bench = pytest.importorskip("bench")


# A trace shaped like a real LibreChat AgentRun: one SELECT run once, echoed both in
# a generation's OpenAI-style tool_calls and in the accumulated tool_batch input, its
# result in the tool_batch output, and a final generation carrying the answer.
def _trace():
    return {
        "id": "t1", "sessionId": "conv-1",
        "observations": [
            {"type": "GENERATION", "startTime": "2026-09-19T00:00:00Z",
             "usageDetails": {"input": 100, "output": 20, "cache_read_input_tokens": 40},
             "output": {"role": "ai", "tool_calls": [
                 {"id": "c1", "name": "run_select_query_mcp_dam",
                  "function": {"name": "run_select_query_mcp_dam",
                               "arguments": '{"query": "SELECT 1"}'}}]}},
            {"type": "TOOL", "name": "tool_batch",
             "input": {"messages": [
                 {"id": ["AIMessage"], "kwargs": {"tool_calls": [
                     {"id": "c1", "name": "run_select_query_mcp_dam",
                      "args": {"query": "SELECT 1"}}]}}]},
             "output": {"messages": [
                 {"id": ["ToolMessage"],
                  "kwargs": {"tool_call_id": "c1", "content": '[{"1":1}]'}}]}},
            {"type": "GENERATION", "startTime": "2026-09-19T00:01:00Z",
             "usageDetails": {"input": 200, "output": 30},
             "output": {"role": "ai", "content": "The answer is 1."}},
        ],
    }


class _StubSession:
    def __init__(self, **conv):
        self.conv = {"conversationId": "conv-1", "assistantMessageId": "a1",
                     "final_answer": "streamed answer", "served_model": "claude-sonnet-4-6"}
        self.conv.update(conv)
        self.agent_args = None
        self.seen = None

    def ensure_agent(self, model, instructions):
        self.agent_args = (model, instructions)
        return "agent_stub"

    def chat(self, text, agent_id):
        self.seen = (text, agent_id)
        return self.conv


class _StubReader:
    def __init__(self, trace):
        self._trace = trace

    def trace_for_conversation(self, conversation_id, **kw):
        self._asked = conversation_id
        return self._trace


# ── reconstruction ────────────────────────────────────────────────────────────

def test_reconstruction_pairs_each_query_with_its_result():
    r = bench._reconstruct_librechat_run(_trace())
    assert r["sqls"] == ["SELECT 1"], "the query, deduped across both serializations"
    assert r["sql_results"] == ['[{"1":1}]'], "paired to the query by tool_call_id"
    assert r["final_answer"] == "The answer is 1.", "last generation's text"


def test_reconstruction_counts_generations_as_turns():
    assert bench._reconstruct_librechat_run(_trace())["turns"] == 2


def test_reconstruction_sums_usage_across_generations():
    usage = bench._reconstruct_librechat_run(_trace())["usage"].as_dict()
    assert usage["prompt_tokens"] == 300 and usage["completion_tokens"] == 50
    assert usage["cache_read_tokens"] == 40
    assert usage["total_tokens"] == 300 + 50 + 40


def _trace_tool_dispatch():
    """The LangGraph shape a live instance emits: a tool-dispatch step whose input is
    the call batch and whose output.messages are results in the same order, and the
    answer at the trace level rather than in the last generation."""
    return {
        "id": "t2", "sessionId": "conv-2",
        "output": "Holmes-Schwartz is a Financial Services customer.",
        "observations": [
            {"type": "GENERATION", "startTime": "2026-09-19T00:00:00Z",
             "usageDetails": {},
             "output": {"role": "assistant", "content": "Let me look.",
                        "tool_calls": [{"id": "c1", "type": "function",
                                        "function": {"name": "run_select_query_mcp_ClickHouse",
                                                     "arguments": '{"query": "SHOW TABLES"}'}}]}},
            {"type": "CHAIN", "name": "tool-dispatch",
             "input": [{"name": "run_select_query_mcp_ClickHouse", "args": {"query": "SHOW TABLES"}},
                       {"name": "run_select_query_mcp_ClickHouse", "args": {"query": "SELECT 2"}}],
             "output": {"messages": [{"role": "run_select_query_mcp_ClickHouse", "content": "(empty result)"},
                                     {"role": "run_select_query_mcp_ClickHouse", "content": '{"2":2}'}]}},
            {"type": "GENERATION", "startTime": "2026-09-19T00:01:00Z", "usageDetails": {},
             "output": {"role": "assistant", "content": "answer step"}},
        ],
    }


def test_reconstruction_handles_the_tool_dispatch_shape():
    r = bench._reconstruct_librechat_run(_trace_tool_dispatch())
    assert r["sqls"] == ["SHOW TABLES", "SELECT 2"], "both calls, in dispatch order"
    assert r["sql_results"] == ["(empty result)", '{"2":2}'], "results paired positionally"
    assert r["final_answer"] == "Holmes-Schwartz is a Financial Services customer.", \
        "answer from trace.output, not the last generation"
    assert r["turns"] == 2, "two model generations"


def test_reconstruction_of_an_empty_trace_is_empty_not_an_error():
    r = bench._reconstruct_librechat_run(None)
    assert r == {"sqls": [], "sql_results": [], "final_answer": "", "turns": 0,
                 "usage": r["usage"]}
    assert r["usage"].as_dict()["api_calls"] == 0


# ── the runner returns the standard dict ────────────────────────────────────────

REQUIRED = {"sqls", "sql_results", "final_answer", "turns", "served_model",
            "latency", "error", "usage"}


def test_runner_returns_the_standard_shape_from_the_trace():
    sess = _StubSession()
    res = bench.run_candidate_librechat("how many?", "claude-sonnet-4-6",
                                        system_prompt="BOARD SYSTEM PROMPT",
                                        session=sess, reader=_StubReader(_trace()))
    assert REQUIRED <= set(res), f"missing keys: {REQUIRED - set(res)}"
    assert sess.agent_args == ("claude-sonnet-4-6", "BOARD SYSTEM PROMPT"), \
        "the board system prompt is applied as the agent's instructions"
    assert sess.seen == ("how many?", "agent_stub"), "driven against the provisioned agent"
    assert res["sqls"] == ["SELECT 1"] and res["sql_results"] == ['[{"1":1}]']
    assert res["turns"] == 2 and res["error"] is None
    assert res["final_answer"] == "streamed answer", "streamed answer preferred over reconstructed"


def test_runner_flags_a_missing_trace_rather_than_scoring_a_zero_run():
    res = bench.run_candidate_librechat("q", "claude-sonnet-4-6", system_prompt="p",
                                        session=_StubSession(), reader=_StubReader(None))
    assert res["error"] == bench.ERR_LIBRECHAT_NO_TRACE
    assert res["sqls"] == [] and res["usage"]["calls_missing_usage"] == 0
    assert res["final_answer"] == "streamed answer", "the stream still gives the answer"


def test_ensure_agent_posts_the_board_prompt_and_caches(monkeypatch):
    sess = bench._LibreChatSession("http://lc", "", "e@x", "pw", provider="gateway",
                                   mcp_tool="run_select_query_mcp_ClickHouse")
    import time as _t
    sess._token, sess._token_at = "t", _t.time()   # a fresh token, skip the login round-trip

    posts = []

    class _R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"id": "agent_1"}

    monkeypatch.setattr(sess._client, "post",
                        lambda url, headers=None, json=None: posts.append((url, json)) or _R())
    a1 = sess.ensure_agent("claude-sonnet-4-6", "BOARD PROMPT")
    a2 = sess.ensure_agent("claude-sonnet-4-6", "BOARD PROMPT")
    assert a1 == a2 == "agent_1"
    assert len(posts) == 1, "created once, then reused from cache"
    url, body = posts[0]
    assert url.endswith("/api/agents")
    assert body["instructions"] == "BOARD PROMPT", "the board prompt is the agent's instructions"
    assert body["provider"] == "gateway" and body["model"] == "claude-sonnet-4-6"
    assert body["tools"] == ["run_select_query_mcp_ClickHouse"]


# ── dispatch routes a librechat entry here ──────────────────────────────────────

def test_run_candidate_routes_librechat_membership_to_this_runner(monkeypatch):
    calls = {}

    def stub(*a, **k):
        calls["hit"] = (a, k)
        return {"ok": True}

    monkeypatch.setitem(bench.LIBRECHAT_CANDIDATES, "lc-x", "DAM Bench Agent")
    monkeypatch.setattr(bench, "run_candidate_librechat", stub)
    out = bench.run_candidate("q", "lc-x", "DAM Bench Agent",
                              ch_query=lambda s: "[]", system_prompt="s")
    assert out == {"ok": True} and "hit" in calls, "librechat entry did not route to the runner"
    assert calls["hit"][0][:2] == ("q", "DAM Bench Agent"), "question and spec forwarded"


# ── SSE parsing ─────────────────────────────────────────────────────────────────

def test_sse_event_parses_event_and_json_data():
    msg = bench._parse_sse_event('event: message\ndata: {"final": true, "x": 1}')
    assert msg["event"] == "message" and msg["data"]["final"] is True


def test_sse_record_without_data_is_ignored():
    assert bench._parse_sse_event("event: ping") is None


# ── the warehouse MCP shim forwards to the bundle /query ─────────────────────────

def _mcp_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mcp_warehouse", DAM / "librechat" / "mcp_warehouse.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mcp_shim_forwards_the_query_and_returns_the_result(monkeypatch):
    mod = _mcp_module()
    seen = {}

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"result": '[{"n":3}]'}

    def fake_post(url, json, timeout):
        seen.update(url=url, body=json)
        return _Resp()

    monkeypatch.setattr(mod.httpx, "post", fake_post)
    out = mod.run_select_query_text("SELECT 3", warehouse_url="http://wh:8123")
    assert out == '[{"n":3}]'
    assert seen["url"] == "http://wh:8123/query" and seen["body"] == {"query": "SELECT 3"}


def test_mcp_shim_returns_errors_as_a_tool_string_not_a_raise(monkeypatch):
    mod = _mcp_module()

    def boom(*a, **k):
        raise RuntimeError("warehouse down")

    monkeypatch.setattr(mod.httpx, "post", boom)
    out = mod.run_select_query_text("SELECT 1")
    assert out.startswith("Error:") and "warehouse down" in out
