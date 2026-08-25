"""
Candidate runners and blind judge for the synthetic text2sql benchmark.

ch_query and system_prompt are passed as arguments to keep DB/schema
concerns in the notebook and infrastructure concerns here.
"""
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

import boto3
import botocore.auth
import botocore.awsrequest
import google.auth
import google.auth.transport.requests
import httpx
import anthropic
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ── Constants ─────────────────────────────────────────────────────────────────

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")

# Model registry, provider endpoints and judge seats live in configuration, not
# here: publishing this module would otherwise publish the catalog and our internal
# hosts, and reading the registry would keep requiring provider credentials because
# importing this module constructs six clients. See registry.py.
from registry import (  # noqa: E402
    ADAPTIVE_THINKING_ONLY, ALL_CANDIDATES, ANNOTATORS, ANTHROPIC_CANDIDATES,
    BEDROCK_REASONING, CANDIDATES, EFFORT_CAPABLE, ENDPOINTS, FIREWORKS_CANDIDATES,
    GATEWAY_CANDIDATES, GEMINI_CANDIDATES, GEMINI_GLOBAL, JUDGE_MODEL, LINKER,
    MODELS,
    JUDGE_MODEL_IDS, JUDGE_PROVIDER, JUDGE_SEATS, MANTLE_CANDIDATES,
    MANTLE_RESPONSES_CANDIDATES, OPENAI_CANDIDATES, OPENAI_RESPONSES_ONLY,
    RETIRED_CANDIDATES,
)

# Overridable for sensitivity sweeps: the budget is never announced to
# the model, so runs at different budgets share a distribution over early turns.
MAX_TURNS   = int(os.environ.get("DAM_MAX_TURNS", "10"))
EVAL_SEED   = 42

# Error marker for a turn truncated by the output-token cap (Messages API
# stop_reason "max_tokens" / OpenAI finish_reason "length"). The answer is
# partial or empty, so the run is flagged rather than scored as complete; the
# sweep treats an empty-answer truncation like a max-turns exhaustion.
ERR_MAX_OUTPUT_TOKENS = "max output tokens"





# Result-set agreement tolerance: catches 59K vs 70K, ignores rounding.
AGREEMENT_TOL = 0.05

# ── Tool schemas ──────────────────────────────────────────────────────────────

TOOLS_BEDROCK = [{
    "toolSpec": {
        "name": "run_select_query",
        "description": "Run a read-only SELECT query against the ClickHouse data warehouse.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        },
    }
}]

TOOLS_OPENAI = [{
    "type": "function",
    "function": {
        "name": "run_select_query",
        "description": "Run a read-only SELECT query against the ClickHouse data warehouse.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}]

# OpenAI Responses API tool shape — flat (no nested "function" wrapper), unlike
# TOOLS_OPENAI for chat/completions.
TOOLS_RESPONSES = [{
    "type": "function",
    "name": "run_select_query",
    "description": "Run a read-only SELECT query against the ClickHouse data warehouse.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}]

# Native Anthropic Messages API tool shape (name/description/input_schema).
TOOLS_MESSAGES = [{
    "name": "run_select_query",
    "description": "Run a read-only SELECT query against the ClickHouse data warehouse.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}]

# ── Auth helpers ──────────────────────────────────────────────────────────────

class _AWSv4Auth(httpx.Auth):
    def __init__(self, service: str, region: str):
        self._creds  = boto3.Session().get_credentials()
        self._signer = botocore.auth.SigV4Auth(self._creds, service, region)

    def auth_flow(self, request):
        aws_req = botocore.awsrequest.AWSRequest(
            method=request.method, url=str(request.url),
            data=request.content or b"", headers=dict(request.headers))
        self._signer.add_auth(aws_req)
        for k, v in aws_req.headers.items():
            request.headers[k] = v
        yield request


class _GCPAuth(httpx.Auth):
    """Vertex application-default credentials, resolved on first request.

    Not at construction. `google.auth.default()` raises when there are no ADC,
    and this is instantiated at module scope, so an eager lookup made `import
    bench` fail outright for anyone without a GCP project — including the
    three-API-key path the runnable example documents, where nothing calls Gemini
    at all. Discovered by CI, which has no ADC; it passed locally only because a
    developer machine does.

    The failure now lands on the Gemini call that actually needs credentials,
    where the message is about the request being made rather than about an import.
    """

    def __init__(self):
        self._creds = None

    def auth_flow(self, request):
        if self._creds is None:
            self._creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        if not self._creds.valid:
            self._creds.refresh(google.auth.transport.requests.Request())
        request.headers["Authorization"] = f"Bearer {self._creds.token}"
        yield request

# ── Clients ───────────────────────────────────────────────────────────────────

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

mantle_client = OpenAI(
    base_url=ENDPOINTS["mantle_chat"].format(region=AWS_REGION),
    api_key="aws",
    http_client=httpx.Client(auth=_AWSv4Auth("bedrock", AWS_REGION)),
)
# Gemma 4 is served only on bedrock-mantle's /openai/v1 base (model card: "served
# at /openai/v1/responses, not the default /v1/responses"), and its chat/completions
# route rejects function tools alongside reasoning — same constraint class as
# gpt-5.6, so it takes the Responses-API runner.
mantle_openai_client = OpenAI(
    base_url=ENDPOINTS["mantle_responses"].format(region=AWS_REGION),
    api_key="aws",
    http_client=httpx.Client(auth=_AWSv4Auth("bedrock", AWS_REGION)),
)

openai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

_gcp_project  = os.environ.get("GCP_PROJECT", "clickhouse-aiml")
_gcp_location = os.environ.get("GCP_LOCATION", "us-central1")
gemini_client = OpenAI(
    base_url=ENDPOINTS["vertex_regional"].format(project=_gcp_project,
                                                 location=_gcp_location),
    api_key="adc",
    http_client=httpx.Client(auth=_GCPAuth()),
)

# Gemini 3.x is only on the `global` location, which uses a different host form.
gemini_global_client = OpenAI(
    base_url=ENDPOINTS["vertex"].format(project=_gcp_project),
    api_key="adc",
    http_client=httpx.Client(auth=_GCPAuth()),
)

fireworks_client = OpenAI(
    base_url=ENDPOINTS["fireworks"],
    api_key=os.environ.get("FIREWORKS_API_KEY"),
)

# ClickHouse inference gateway (OpenAI-compatible). URL/key from env; dev default.
# Normalise the base so an override that already carries a /v1 suffix (the usual
# OpenAI-compatible convention) or a trailing slash does not become /v1/v1.
_gateway_base = os.environ.get(
    "INFERENCE_GATEWAY_URL",
    ENDPOINTS["gateway"]).rstrip("/")
if not _gateway_base.endswith("/v1"):
    _gateway_base += "/v1"
gateway_client = OpenAI(
    base_url=_gateway_base,
    api_key=os.environ.get("INFERENCE_GATEWAY_KEY"),
)

# Direct Anthropic API via its OpenAI-compatible endpoint. Retained for reference;
# unreleased Claudes now run on the native Messages API below (the OpenAI-compat
# endpoint rejects adaptive thinking, so thinking/effort sweeps need the native API).
anthropic_client = OpenAI(
    base_url=ENDPOINTS["anthropic"],
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

# Native Anthropic Messages API client — supports adaptive thinking + output_config
# effort (low/medium/high/xhigh), which the OpenAI-compat endpoint does not.
anthropic_native = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ── Concurrency + retry ─────────────────────────────────────────────────────────
# The annotate/eval loops fan work out across a thread pool (one (question,
# candidate) pair or annotator run per task). All the SDK clients above are
# thread-safe for concurrent calls; the only shared non-thread-safe resource is the
# chDB session, which serializes on its own lock (see warehouse.Warehouse). Running
# many model calls at once makes provider-side throttling (429 / ThrottlingException)
# far more likely, so every model call is wrapped in `retry` with exponential
# backoff — otherwise a single throttle would sink a candidate mid-run.

# Retryable failures, matched by exception class name so we don't have to import
# every SDK's error hierarchy. botocore ClientError (Bedrock) carries the real code
# in .response["Error"]["Code"]; OpenAI/Anthropic errors expose .status_code.
_RETRYABLE_NAMES = {
    "ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException",
    "ModelNotReadyException", "RateLimitError", "APITimeoutError", "APIConnectionError",
    "InternalServerError", "APIStatusError", "OverloadedError",
}
_RETRYABLE_CODES  = {"ThrottlingException", "TooManyRequestsException",
                     "ServiceUnavailableException", "Throttling", "RequestLimitExceeded"}
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _is_retryable(e: Exception) -> bool:
    if type(e).__name__ in _RETRYABLE_NAMES:
        return True
    resp = getattr(e, "response", None)
    if isinstance(resp, dict) and resp.get("Error", {}).get("Code") in _RETRYABLE_CODES:
        return True
    return getattr(e, "status_code", None) in _RETRYABLE_STATUS


def retry(fn: Callable, *, what: str = "call", max_retries: int = 5, base: float = 1.0):
    """Call fn(), retrying retryable (throttle/timeout/5xx) failures with capped
    exponential backoff. Non-retryable errors propagate immediately."""
    delay = base
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries or not _is_retryable(e):
                raise
            print(f"    [{what}: {type(e).__name__}, retry {attempt+1}/{max_retries} in {delay:.0f}s]")
            time.sleep(delay)
            delay = min(delay * 2, 30.0)


def map_concurrent(fn: Callable, items: Iterable, workers: int):
    """Run fn(item) across a thread pool, yielding (item, result, exc) in completion
    order. Exceptions are captured and returned as the third element (result None)
    rather than raised, so one failing item never sinks the batch. workers<=1 runs
    inline (no pool) for easy debugging."""
    items = list(items)
    if workers <= 1:
        for it in items:
            try:
                yield it, fn(it), None
            except Exception as e:  # noqa: BLE001 - surfaced to caller as third tuple element
                yield it, None, e
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for fut in as_completed(futs):
            it = futs[fut]
            try:
                yield it, fut.result(), None
            except Exception as e:  # noqa: BLE001 - surfaced to caller as third tuple element
                yield it, None, e

# ── Candidate runners ─────────────────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def _bedrock_converse_with_retry(max_retries: int = 5, **kwargs):
    # Retries throttling + service-unavailable + 5xx (see bench.retry); matters most
    # under the concurrent eval loop, where many converse calls are in flight at once.
    return retry(lambda: bedrock.converse(**kwargs), what="bedrock.converse",
                 max_retries=max_retries)


# Bedrock-served models that emit reasoning before the answer, so they need the larger
# output budget (see run_candidate_bedrock). Matched as substrings of the model id.
#
# Verified by probing each Claude on the board with a converse call: only Opus 5 returns a
# reasoningContent block by default. opus48, opus47, sonnet5 and sonnet46 return text only,
# so they are NOT listed here — their published numbers were produced without reasoning and
# raising their budget would change what the board measured without re-running it.


class TokenUsage:
    """Per-run token accumulator, summed across the agentic turns.

    Usage can only be captured at call time, so a run without it can never be
    costed retroactively: re-tokenizing the transcript structurally UNDER-counts
    reasoning models, because hidden thinking is billed as output and never
    appears in the transcript. That is exactly the set of models whose
    cost we most need.

    Providers name these differently AND disagree on what nests inside what, so
    the accumulator normalises to one schema in which prompt / cache-read /
    cache-write are DISJOINT. Every `add_*` is defensive: a provider that omits
    `usage`, or a gateway that drops the details sub-objects, must degrade to
    zeros rather than raise. Telemetry is never allowed to kill a run that has
    already spent real money on tool calls.
    """

    __slots__ = ("prompt", "completion", "reasoning", "cache_read", "cache_write",
                 "calls", "missing")

    def __init__(self):
        self.prompt = self.completion = self.reasoning = 0
        self.cache_read = self.cache_write = 0
        self.calls = self.missing = 0

    @staticmethod
    def _int(obj, *names):
        """First present, integer-valued attribute/key among `names`, else 0."""
        for n in names:
            v = getattr(obj, n, None)
            if v is None and isinstance(obj, dict):
                v = obj.get(n)
            if isinstance(v, (int, float)):
                return int(v)
        return 0

    @staticmethod
    def _sub(obj, name):
        v = getattr(obj, name, None)
        if v is None and isinstance(obj, dict):
            v = obj.get(name)
        return v

    def _record(self, u, prompt, completion, reasoning=0, c_read=0, c_write=0):
        self.calls += 1
        # `u is None` is not the only miss: a provider can return an empty or
        # unrecognised usage object, and adding zeros from it would look like an
        # exact zero rather than a gap. Any real call bills input, so extracting
        # nothing on both axes means we failed to read it.
        if u is None or (prompt == 0 and completion == 0):
            self.missing += 1
            return
        self.prompt += prompt
        self.completion += completion
        self.reasoning += reasoning
        self.cache_read += c_read
        self.cache_write += c_write

    def add_bedrock(self, resp):
        """Converse: cache figures sit beside inputTokens, not inside it."""
        try:
            u = (resp or {}).get("usage")
            self._record(u,
                         self._int(u, "inputTokens"), self._int(u, "outputTokens"),
                         0,
                         self._int(u, "cacheReadInputTokens"),
                         self._int(u, "cacheWriteInputTokens"))
        except Exception:
            self.calls += 1
            self.missing += 1

    def add_openai(self, resp):
        """chat.completions: prompt_tokens is INCLUSIVE of cached_tokens.

        Unlike the Anthropic/Bedrock convention, where cache reads are reported
        alongside a cache-free input count. Subtract so `self.prompt` means the
        same thing on every path and the total does not double-count.
        """
        try:
            u = getattr(resp, "usage", None)
            cd = self._sub(u, "completion_tokens_details")
            pd = self._sub(u, "prompt_tokens_details")
            cached = self._int(pd, "cached_tokens")
            self._record(u,
                         max(self._int(u, "prompt_tokens", "input_tokens") - cached, 0),
                         self._int(u, "completion_tokens", "output_tokens"),
                         self._int(cd, "reasoning_tokens"),
                         cached)
        except Exception:
            self.calls += 1
            self.missing += 1

    def add_responses(self, resp):
        """Responses API: same inclusive-input convention as chat.completions.

        Both `cached_tokens` and `cache_write_tokens` live under
        `input_tokens_details`, i.e. they are components OF `input_tokens`, so
        BOTH come out of the residual prompt figure or the buckets stop being
        disjoint and the total double-counts writes.
        """
        try:
            u = getattr(resp, "usage", None)
            od = self._sub(u, "output_tokens_details")
            idt = self._sub(u, "input_tokens_details")
            cached = self._int(idt, "cached_tokens")
            written = self._int(idt, "cache_write_tokens")
            self._record(u,
                         max(self._int(u, "input_tokens") - cached - written, 0),
                         self._int(u, "output_tokens"),
                         self._int(od, "reasoning_tokens"),
                         cached, written)
        except Exception:
            self.calls += 1
            self.missing += 1

    def add_anthropic(self, resp):
        """Messages API: thinking is billed inside output_tokens, not broken out."""
        try:
            u = getattr(resp, "usage", None)
            self._record(u,
                         self._int(u, "input_tokens"), self._int(u, "output_tokens"),
                         0,
                         self._int(u, "cache_read_input_tokens"),
                         self._int(u, "cache_creation_input_tokens"))
        except Exception:
            self.calls += 1
            self.missing += 1

    def as_dict(self) -> dict:
        # The three input components are disjoint by construction (see
        # add_openai / add_responses), so they are safe to sum. Cache reads and
        # writes are both billed, reads at a discount and writes at a premium, so
        # a "leanness" total that drops them flatters cache-heavy providers.
        #
        # `reasoning_tokens` is deliberately NOT in the total: every provider that
        # reports it counts it inside its completion figure, so adding it again
        # would double-count exactly the models it is meant to illuminate. It is a
        # breakdown of completion_tokens, not a fifth bucket.
        return {
            "prompt_tokens": self.prompt, "completion_tokens": self.completion,
            "reasoning_tokens": self.reasoning,
            "cache_read_tokens": self.cache_read, "cache_write_tokens": self.cache_write,
            "total_tokens": (self.prompt + self.completion
                             + self.cache_read + self.cache_write),
            "api_calls": self.calls,
            # >0 means some turns reported no usage, so the totals are a LOWER
            # bound and must not be presented as exact.
            "calls_missing_usage": self.missing,
        }


def run_candidate_bedrock(
    nl_question: str,
    model_id: str,
    ch_query: Callable[[str], str],
    system_prompt: str,
) -> dict:
    start    = time.time()
    system   = [{"text": system_prompt}]
    messages = [{"role": "user", "content": [{"text": nl_question}]}]
    sqls, sql_results = [], []
    # Reasoning models bill thinking as output and emit it before the answer, so a 2048
    # budget truncates them mid-reasoning: the turn ends with stopReason "max_tokens"
    # having produced no tool call and no answer, which scores as a turn-limited failure
    # rather than an error. Measured on Opus 5 with a step-by-step SQL question: 2048 ->
    # max_tokens at 2048 output tokens, 8192 -> end_turn at 4089. This is the same
    # starvation already fixed for the OpenAI-compatible path, which the Bedrock path
    # never got.
    token_limit = 8192 if model_id in BEDROCK_REASONING else 2048

    usage = TokenUsage()
    for turn in range(MAX_TURNS):
        resp = _bedrock_converse_with_retry(
            modelId=model_id, system=system, messages=messages,
            toolConfig={"tools": TOOLS_BEDROCK},
            inferenceConfig={"maxTokens": token_limit},
        )
        usage.add_bedrock(resp)
        out  = resp["output"]["message"]
        messages.append(out)
        stop = resp["stopReason"]

        if stop == "end_turn":
            final = "".join(b.get("text", "") for b in out["content"])
            # served_model: Converse has no alias resolution — the request id is exact
            return {"sqls": sqls, "sql_results": sql_results,
                    "final_answer": final.strip(), "turns": turn + 1,
                    "served_model": model_id,
                    "latency": round(time.time() - start, 2), "error": None,
                    "usage": usage.as_dict()}

        if stop == "tool_use":
            tool_results = []
            for block in out["content"]:
                if "toolUse" not in block:
                    continue
                tc     = block["toolUse"]
                q      = tc["input"].get("query", "")
                sqls.append(q)
                result = ch_query(q)
                sql_results.append(result)
                tool_results.append({
                    "toolResult": {"toolUseId": tc["toolUseId"],
                                   "content":   [{"text": result}]}
                })
            messages.append({"role": "user", "content": tool_results})
        else:
            break

    return {"sqls": sqls, "sql_results": sql_results,
            "final_answer": "", "turns": MAX_TURNS, "served_model": model_id,
            "latency": round(time.time() - start, 2), "error": f"max turns ({MAX_TURNS})",
            "usage": usage.as_dict()}


def run_candidate_openai_compat(
    nl_question: str,
    model_id: str,
    client: OpenAI,
    ch_query: Callable[[str], str],
    system_prompt: str,
) -> dict:
    start        = time.time()
    # o-series and gpt-5.x are reasoning models -> max_completion_tokens; gemini 2.5-pro
    # and gemini 3.x, deepseek-v4, and Kimi (K2 Thinking / K2.6) are thinking models that
    # emit reasoning inline -> need a larger budget so reasoning tokens don't starve the
    # answer (at 2048 Kimi K2.6 truncated mid-reasoning on 18% of questions before it could
    # even issue a query).
    _is_reasoning = model_id.startswith("o") or model_id.startswith("gpt-5")
    _token_kwarg  = "max_completion_tokens" if _is_reasoning else "max_tokens"
    # qwen3p8-max emits no inline thinking but is verbose enough to hit a 2048 cap
    # on plain prose (observed finish_reason=length in the pre-trust probe).
    # The substring list is history: it dates from when "does this model emit
    # inline reasoning" had to be guessed from an id. The registry states it now,
    # so a model flagged `reasoning: true` gets the larger budget whatever it is
    # called. Additive rather than a replacement, deliberately: every id below
    # keeps its budget, so no board candidate changes. Without this, a flagged
    # model whose id matches nothing in the list (glm-5p2 in the example config)
    # was given the reasoning floor as a judge and the 2048 cap as a candidate,
    # from the same flag.
    _big          = (_is_reasoning or model_id in BEDROCK_REASONING
                     or model_id.startswith("google/gemini-2.5-pro")
                     or model_id.startswith("google/gemini-3") or "deepseek-v4" in model_id
                     or "kimi" in model_id or "qwen3p8-max" in model_id)
    _token_limit  = 8192 if _big else 2048
    messages     = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": nl_question},
    ]
    sqls, sql_results = [], []
    served = model_id  # resolved id as reported by the endpoint

    usage = TokenUsage()
    for turn in range(MAX_TURNS):
        resp   = retry(lambda: client.chat.completions.create(
            model=model_id, messages=messages, tools=TOOLS_OPENAI,
            **{_token_kwarg: _token_limit}), what=f"chat.completions[{model_id}]")
        usage.add_openai(resp)
        served = getattr(resp, "model", None) or served
        choice = resp.choices[0]
        msg    = choice.message

        asst = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            asst["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(asst)

        if choice.finish_reason == "stop" or not msg.tool_calls:
            # finish_reason == "length" means the budget ran out mid-generation, so
            # the answer is partial/empty — flag it rather than scoring as complete.
            err = ERR_MAX_OUTPUT_TOKENS if choice.finish_reason == "length" else None
            return {"sqls": sqls, "sql_results": sql_results,
                    "final_answer": _strip_thinking(msg.content),
                    "turns": turn + 1, "served_model": served,
                    "latency": round(time.time() - start, 2), "error": err,
                    "usage": usage.as_dict()}

        for tc in msg.tool_calls:
            if tc.function.name == "run_select_query":
                try:
                    q = json.loads(tc.function.arguments).get("query", "")
                except Exception:
                    q = ""
                sqls.append(q)
                result = ch_query(q)
                sql_results.append(result)
            else:
                result = "Error: unknown tool."
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return {"sqls": sqls, "sql_results": sql_results,
            "final_answer": "", "turns": MAX_TURNS, "served_model": served,
            "latency": round(time.time() - start, 2), "error": f"max turns ({MAX_TURNS})",
            "usage": usage.as_dict()}


def run_candidate_responses_api(
    nl_question: str,
    model_id: str,
    client: OpenAI,
    ch_query: Callable[[str], str],
    system_prompt: str,
) -> dict:
    """Agentic loop over the OpenAI Responses API (/v1/responses).

    The supported path for gpt-5.6 with function tools + reasoning on (chat/completions
    rejects that combination). Conversation state is kept server-side via
    `previous_response_id` (store=True): each turn sends only the new turn's items (the
    question, then the function_call_output for each tool call), and the server carries
    the model's reasoning + function_call context forward — so reasoning continuity holds
    across tool calls without re-sending output items (which aren't valid as input).
    """
    start = time.time()
    sqls, sql_results = [], []
    served = model_id  # resolved id as reported by the endpoint
    prev_id = None
    pending: list = [{"role": "user", "content": nl_question}]

    usage = TokenUsage()
    for turn in range(MAX_TURNS):
        resp = retry(lambda: client.responses.create(
            model=model_id, instructions=system_prompt, input=pending,
            previous_response_id=prev_id, tools=TOOLS_RESPONSES,
            max_output_tokens=16000, store=True,
        ), what=f"responses.create[{model_id}]")
        usage.add_responses(resp)
        served = getattr(resp, "model", None) or served
        prev_id = resp.id

        calls = [it for it in resp.output if getattr(it, "type", None) == "function_call"]
        if not calls:
            # No tool call -> concluded, or truncated mid-generation by the token cap.
            err = (ERR_MAX_OUTPUT_TOKENS
                   if resp.status == "incomplete"
                   and getattr(resp.incomplete_details, "reason", None) == "max_output_tokens"
                   else None)
            return {"sqls": sqls, "sql_results": sql_results,
                    "final_answer": _strip_thinking(resp.output_text or ""),
                    "turns": turn + 1, "served_model": served,
                    "latency": round(time.time() - start, 2), "error": err,
                    "usage": usage.as_dict()}

        pending = []
        for fc in calls:
            if fc.name == "run_select_query":
                try:
                    q = json.loads(fc.arguments).get("query", "")
                except Exception:
                    q = ""
                sqls.append(q)
                result = ch_query(q)
                sql_results.append(result)
            else:
                result = "Error: unknown tool."
            pending.append({"type": "function_call_output",
                            "call_id": fc.call_id, "output": result})

    return {"sqls": sqls, "sql_results": sql_results,
            "final_answer": "", "turns": MAX_TURNS, "served_model": served,
            "latency": round(time.time() - start, 2), "error": f"max turns ({MAX_TURNS})",
            "usage": usage.as_dict()}


def run_candidate_messages_api(
    nl_question: str,
    model_id: str,
    ch_query: Callable[[str], str],
    system_prompt: str,
    *,
    thinking: str = "off",
    effort: str = "high",
) -> dict:
    """Native Anthropic Messages API agentic loop.

    `thinking` is "off" (disabled) or "on" (adaptive); `effort` is one of
    low/medium/high/xhigh. The OpenAI-compat endpoint rejects adaptive thinking,
    so the native API is required for thinking/effort sweeps. Assistant content
    (including thinking blocks) is echoed back unchanged on each turn, as the API
    requires. Reports cumulative `output_tokens` — thinking is billed as output,
    so this is the cost axis for the effort/thinking tradeoff. It is derived from
    the TokenUsage accumulator rather than read off each response separately: the
    two are the same quantity, and the accumulator's read is the defended one.
    """
    start        = time.time()
    # Mythos-class models (Fable) reject thinking.type.disabled — thinking defaults to
    # adaptive and can only be raised via enabled+budget. So omit the thinking config
    # for them (adaptive default), which is also the fair mode to benchmark a reasoning
    # model in (cf. GPT-5.6 keeping reasoning on). Effort still applies.
    if model_id not in EFFORT_CAPABLE:
        extra = {}                       # public Claude models reject these outright
    elif model_id in ADAPTIVE_THINKING_ONLY:
        extra = {"output_config": {"effort": effort}}
    else:
        thinking_cfg = {"type": "adaptive"} if thinking == "on" else {"type": "disabled"}
        extra        = {"thinking": thinking_cfg, "output_config": {"effort": effort}}
    messages     = [{"role": "user", "content": nl_question}]
    sqls, sql_results = [], []
    served       = model_id  # resolved id as reported by the endpoint

    usage = TokenUsage()
    for turn in range(MAX_TURNS):
        resp = retry(lambda: anthropic_native.messages.create(
            model=model_id, max_tokens=16000, system=system_prompt,
            messages=messages, tools=TOOLS_MESSAGES, extra_body=extra,
        ), what=f"messages.create[{model_id}]")
        usage.add_anthropic(resp)
        served      = getattr(resp, "model", None) or served
        # Echo assistant content back verbatim (incl. thinking blocks) for the next turn.
        messages.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})

        if resp.stop_reason == "tool_use":
            tool_results = []
            # The Messages API requires a tool_result for EVERY tool_use block in
            # the echoed assistant turn; an unanswered tool_use fails the next call.
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                if block.name == "run_select_query":
                    q = (block.input or {}).get("query", "")
                    sqls.append(q)
                    result = ch_query(q)
                    sql_results.append(result)
                else:
                    result = f"Error: unknown tool {block.name!r}"
                tool_results.append({"type": "tool_result",
                                     "tool_use_id": block.id, "content": result})
            messages.append({"role": "user", "content": tool_results})
        else:
            final = "".join(b.text for b in resp.content if b.type == "text")
            # max_tokens means the turn was truncated mid-generation (often mid-
            # thinking, since thinking bills as output) — the answer is partial or
            # empty, so flag it as an error rather than scoring it as complete.
            err = ERR_MAX_OUTPUT_TOKENS if resp.stop_reason == "max_tokens" else None
            return {"sqls": sqls, "sql_results": sql_results,
                    "final_answer": final.strip(), "turns": turn + 1,
                    "served_model": served,
                    "latency": round(time.time() - start, 2), "error": err,
                    "output_tokens": usage.completion,
                    "usage": usage.as_dict()}

    return {"sqls": sqls, "sql_results": sql_results, "final_answer": "",
            "turns": MAX_TURNS, "served_model": served,
            "latency": round(time.time() - start, 2),
            "error": f"max turns ({MAX_TURNS})", "output_tokens": usage.completion,
            "usage": usage.as_dict()}


def run_candidate(
    nl_question: str,
    model_name: str,
    model_id: str,
    ch_query: Callable[[str], str],
    system_prompt: str,
) -> dict:
    if model_name in GEMINI_CANDIDATES:
        client = gemini_global_client if model_name in GEMINI_GLOBAL else gemini_client
        return run_candidate_openai_compat(nl_question, model_id, client, ch_query, system_prompt)
    if model_name in OPENAI_CANDIDATES:
        if model_name in OPENAI_RESPONSES_ONLY:
            return run_candidate_responses_api(nl_question, model_id, openai_client, ch_query, system_prompt)
        return run_candidate_openai_compat(nl_question, model_id, openai_client, ch_query, system_prompt)
    if model_name in MANTLE_RESPONSES_CANDIDATES:
        return run_candidate_responses_api(nl_question, model_id, mantle_openai_client, ch_query, system_prompt)
    if model_name in MANTLE_CANDIDATES:
        return run_candidate_openai_compat(nl_question, model_id, mantle_client, ch_query, system_prompt)
    if model_name in FIREWORKS_CANDIDATES:
        return run_candidate_openai_compat(nl_question, model_id, fireworks_client, ch_query, system_prompt)
    if model_name in GATEWAY_CANDIDATES:
        return run_candidate_openai_compat(nl_question, model_id, gateway_client, ch_query, system_prompt)
    if model_name in ANTHROPIC_CANDIDATES:
        return run_candidate_messages_api(nl_question, model_id, ch_query, system_prompt)
    return run_candidate_bedrock(nl_question, model_id, ch_query, system_prompt)

# ── Cross-model judge ─────────────────────────────────────────────────────────

def judge_token_budget(model_id: str, max_tokens: int) -> tuple[str, int]:
    """-> (token kwarg, limit) for a judge call.

    "Emits reasoning, so a small verdict budget starves it" is a property of the
    model, not of its name. This was a prefix test that only recognised OpenAI's,
    so a thinking model on any other provider spent its whole budget reasoning and
    returned an empty verdict. On a three-seat panel that is one lost vote; on a
    two-seat panel it takes the question down with it, because judge_panel needs
    two live votes to score at all.

    Extracted so the rule is testable without a provider call.
    """
    reasoning = (model_id.startswith("o") or model_id.startswith("gpt-5")
                 or model_id in BEDROCK_REASONING)
    if reasoning:
        return "max_completion_tokens", max(max_tokens, 2048)
    return "max_tokens", max_tokens


def _judge_complete(judge_name: str, prompt: str, max_tokens: int = 512,
                    temperature: float | None = None) -> str:
    """Single-shot judge completion (no tools), routed to the judge's provider.
    Judges may be non-candidate models (e.g. the gpt-5.x flagships), so IDs and
    clients resolve from the judge registry, not ALL_CANDIDATES.

    `temperature` is opt-in and left unset by default, which keeps every judge
    call on its provider's default as before. The column linker asks for 0: it is
    scoring rather than a vote, so the same pair of column sets has to map the
    same way across runs. Silently dropped when the model reports reasoning,
    since those reject any temperature but their own default.
    """
    # Route on the judge's own provider, not on the name of the seat it sits in.
    # The seat name is a label for provider diversity ("anthropic", "google"); it
    # is not a client. Treating it as one hardcoded seat "anthropic" -> Bedrock,
    # which is right for our config (that seat holds Bedrock-served Claudes) and
    # silently wrong for anyone whose Claude judge is a native Anthropic API key.
    # A judge that is also a declared model resolves from the registry; a
    # judge-only model (the gpt-5.x flagships) falls back to its seat.
    provider = (MODELS[judge_name]["provider"] if judge_name in MODELS
                else JUDGE_PROVIDER.get(judge_name, "anthropic"))
    model_id = JUDGE_MODEL_IDS.get(judge_name) or ALL_CANDIDATES.get(judge_name, JUDGE_MODEL)
    # Reasoning models reject a temperature other than their default, so an
    # explicit 0 would turn the call into a 400 instead of making it reproducible.
    _temp = None if judge_token_budget(model_id, max_tokens)[0] == "max_completion_tokens" \
        else temperature

    if provider == "bedrock":
        _cfg = {"maxTokens": max_tokens}
        if _temp is not None:
            _cfg["temperature"] = _temp
        resp = retry(lambda: bedrock.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig=_cfg,
        ), what=f"judge.converse[{judge_name}]")
        return "".join(b.get("text", "") for b in resp["output"]["message"]["content"]).strip()

    if provider == "anthropic":
        resp = retry(lambda: anthropic_native.messages.create(
            model=model_id, max_tokens=max(max_tokens, 1024),
            messages=[{"role": "user", "content": prompt}],
            **({} if _temp is None else {"temperature": _temp}),
        ), what=f"judge.messages[{judge_name}]")
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    clients = {"openai": openai_client, "google": gemini_client,
               "gemini": gemini_client, "mantle": mantle_client,
               "fireworks": fireworks_client, "gateway": gateway_client}
    if provider not in clients:
        raise ValueError(
            f"judge {judge_name!r} has provider {provider!r}, which has no judge "
            f"client. Supported: bedrock, anthropic, {', '.join(sorted(clients))}.")
    client = clients[provider]
    # gpt-5.x / o-series are reasoning models: use max_completion_tokens and a larger
    # budget so reasoning tokens don't starve the short JSON verdict.
    _kwarg, _limit = judge_token_budget(model_id, max_tokens)
    if model_id.startswith("google/gemini-2.5-pro"):
        _limit = max(_limit, 4096)
    resp = retry(lambda: client.chat.completions.create(
        model=model_id, messages=[{"role": "user", "content": prompt}],
        **{_kwarg: _limit},
        **({} if _temp is None else {"temperature": _temp}),
    ), what=f"judge.completions[{judge_name}]")
    return _strip_thinking(resp.choices[0].message.content)


_JUDGE_PROMPT = """\
You are evaluating whether two SQL agents answered a data warehouse question equivalently.

Question: {question}

A deterministic, schema-aware comparison of the two agents' returned result sets reports them as: {data_equiv}. This is the authoritative verdict on whether they retrieved the same DATA — trust it over your own reading of the rows below (the rows may be a truncated sample).

=== Answer {label_a} ===
SQL queries run: {n_a}
Result set (sample):
{results_a}
Conclusion: {answer_a}

=== Answer {label_b} ===
SQL queries run: {n_b}
Result set (sample):
{results_b}
Conclusion: {answer_b}

Decide whether the two answers are equivalent:
- For data-retrieval questions (lists, lookups, breakdowns) the result set IS the answer — defer to the result-set comparison above.
- For interpretive questions (yes/no, comparisons, "are they over X") the conclusion is the answer — two agents can reach the same correct conclusion from differently-shaped result sets, so weigh the conclusions even when the result sets differ.
Ignore style, formatting, and column-name differences.

Respond ONLY with JSON:
{{"verdict": "equivalent"|"not_equivalent"|"cannot_determine", "reason": "<one sentence>"}}

Use "cannot_determine" only if neither the result sets nor the conclusions give enough to judge."""


def _has_results(results: list) -> bool:
    return any(
        r and not r.startswith("Error:") and r != "(empty result)"
        for r in results
    )


def _fmt_results(results: list, max_chars: int = 2000, max_rows: int = 60) -> str:
    non_empty = [r for r in results if r and not r.startswith("Error:") and r != "(empty result)"]
    if not non_empty:
        return "(no results)"
    best     = max(non_empty, key=len)
    all_rows = [line for line in best.split("\n... (truncated)")[0].splitlines() if line.strip()]
    rows     = all_rows[:max_rows]
    text     = "\n".join(rows)
    if len(text) > max_chars:
        text = text[:max_chars] + " […]"
    # Tell the judge when WE truncated, so it never mistakes our sampling for the
    # agent answering incompletely.
    if len(rows) < len(all_rows):
        text += f"\n[showing {len(rows)} of {len(all_rows)} rows]"
    return text


def judge_score(
    question: str,
    gt_results: list, gt_answer: str,
    cand_results: list, cand_answer: str,
    judge_name: str = "opus47",
    data_equiv: bool | None = None,
) -> dict:
    """Single judge's verdict on whether a candidate matched ground truth.
    Blind (A/B randomized). `judge_name` selects the judge model (any provider).

    The data comparison is done deterministically (entity-linked `annotators_agree`,
    untruncated) and handed to the judge as a signal; the judge weighs it against the
    conclusions (authoritative for retrieval questions, advisory for interpretive
    ones). `data_equiv` is computed here if not supplied (the panel passes it once)."""
    gt_has   = _has_results(gt_results)
    cand_has = _has_results(cand_results)

    if not gt_has and not cand_has:
        return {"outcome": "tie", "verdict": "cannot_determine",
                "reasoning": "both sides returned no data", "judge": judge_name}
    if not gt_has:
        return {"outcome": "tie", "verdict": "cannot_determine",
                "reasoning": "ground truth returned no data", "judge": judge_name}
    if not cand_has:
        return {"outcome": "fail", "verdict": "not_equivalent",
                "reasoning": "candidate returned no data", "judge": judge_name}

    if data_equiv is None:
        data_equiv = annotators_agree(cand_results, gt_results)
    de_str = "EQUIVALENT" if data_equiv else "NOT EQUIVALENT"

    _swap    = random.random() < 0.5
    _a_res   = cand_results if _swap else gt_results
    _b_res   = gt_results   if _swap else cand_results
    _a_ans   = cand_answer  if _swap else gt_answer
    _b_ans   = gt_answer    if _swap else cand_answer
    _label_a, _label_b = ("B", "A") if _swap else ("A", "B")

    _prompt = _JUDGE_PROMPT.format(
        question=question, data_equiv=de_str,
        label_a=_label_a, n_a=len(_a_res), results_a=_fmt_results(_a_res),
        answer_a=(_a_ans or "(none)")[:4000],
        label_b=_label_b, n_b=len(_b_res), results_b=_fmt_results(_b_res),
        answer_b=(_b_ans or "(none)")[:4000],
    )
    _text  = _judge_complete(judge_name, _prompt, max_tokens=512)
    _match = re.search(r"\{.*\}", _text, re.DOTALL)
    try:
        _raw = json.loads(_match.group()) if _match else {}
    except Exception:
        _raw = {}
    _verdict = _raw.get("verdict", "cannot_determine")

    outcome = {"equivalent": "pass", "not_equivalent": "fail"}.get(_verdict, "tie")
    return {"outcome": outcome, "verdict": _verdict, "reasoning": _raw.get("reason", ""),
            "judge": judge_name, "data_equiv": de_str}


def select_panel(cand_model_name: str) -> list[str]:
    """One judge per provider seat, best-first, never the candidate itself
    (drops to a provider's #2 when its #1 is the candidate). Always 3 judges ->
    odd-sized -> a majority always resolves and no model judges itself."""
    panel: list[str] = []
    for models in JUDGE_SEATS.values():
        pick = next((m for m in models if m != cand_model_name), None)
        if pick is not None:
            panel.append(pick)
    return panel


def judge_panel(
    question: str,
    gt_results: list, gt_answer: str,
    cand_results: list, cand_answer: str,
    cand_model_name: str,
) -> dict:
    """Score a candidate with a provider-diverse panel. Majority vote over
    the judges' outcomes; a non-majority split (e.g. pass/tie/fail) scores 'tie'.
    Returns the aggregate plus each judge's individual vote."""
    panel = select_panel(cand_model_name)
    # Deterministic data-equivalence computed once (entity-linked, untruncated) and
    # shared across the panel, so all judges see the same authoritative data signal.
    data_equiv = (_has_results(cand_results) and _has_results(gt_results)
                  and annotators_agree(cand_results, gt_results))
    votes = []
    for jn in panel:
        try:
            votes.append(judge_score(question, gt_results, gt_answer,
                                     cand_results, cand_answer, judge_name=jn,
                                     data_equiv=data_equiv))
        except Exception as e:
            votes.append({"outcome": "error", "verdict": "error",
                          "reasoning": str(e), "judge": jn})

    ok_votes = [v for v in votes if v["outcome"] != "error"]
    tally  = Counter(v["outcome"] for v in ok_votes)
    ranked = tally.most_common()
    if len(ok_votes) < 2:
        outcome = "error"                    # need a real panel; <2 live judges isn't scorable
    elif len(ranked) >= 2 and ranked[0][1] == ranked[1][1]:
        outcome = "tie"                      # no majority -> ambiguous
    else:
        outcome = ranked[0][0]

    return {"outcome": outcome, "panel": panel, "tally": dict(tally), "votes": votes}

# ── Trajectory utilities ──────────────────────────────────────────────────────

def is_exploratory(sql: str) -> bool:
    s = (sql or "").strip().upper()
    if re.match(r"(DESCRIBE|DESC)\b", s):
        return True
    if re.match(r"SHOW\b", s):
        return True
    if any(k in s for k in ("SYSTEM.TABLES", "SYSTEM.COLUMNS", "SYSTEM.DATABASES",
                             "INFORMATION_SCHEMA")):
        return True
    m = re.search(r"\bLIMIT\s+(\d+)\b", s)
    if m and int(m.group(1)) <= 10:
        return True
    return False

# ── Majority-vote ground truth ───────────────────────────────────────

def _parse_result(result_str):
    """Parse a result-set string into normalized, sorted rows (numbers rounded)."""
    if not result_str or result_str.startswith("Error:") or result_str == "(empty result)":
        return []
    rows = []
    for _ln in result_str.strip().splitlines():
        try:
            _row, _norm = json.loads(_ln), {}
            for k, v in _row.items():
                try:    _norm[k.lower().strip()] = round(float(v), 2)
                except: _norm[k.lower().strip()] = str(v)
            rows.append(_norm)
        except Exception:
            continue
    return sorted(rows, key=lambda r: json.dumps(r, sort_keys=True, default=str))


_COL_LINK_CACHE: dict = {}   # (tuple(keys_a), tuple(keys_b)) -> {a_col: b_col}
_COL_LINK_LOCK  = threading.Lock()
_COL_LINK_WARNED = False


def _link_columns(keys_a: list, keys_b: list) -> dict:
    """Map each column in A to the column in B that means the same quantity, so
    aliasing (e.g. total_dollar_usage <-> monthly_spend) doesn't hide a real value
    comparison. Identity when the column sets match (no model call); otherwise a
    cached Haiku call (temp 0 -> reproducible). Falls back to shared names on error.

    The eval loop calls this from many worker threads. The Haiku call runs OUTSIDE
    the lock so column-linking stays concurrent, but the cache read and write are
    locked and the write is a first-writer-wins `setdefault` -> every thread that
    misses on the same key returns the one stored mapping (temp-0 output isn't
    byte-guaranteed across calls, so last-writer-wins could otherwise hand different
    threads different mappings for the same key)."""
    ka, kb = tuple(keys_a), tuple(keys_b)
    if set(ka) == set(kb):
        return {k: k for k in ka}
    with _COL_LINK_LOCK:
        if (ka, kb) in _COL_LINK_CACHE:
            return _COL_LINK_CACHE[(ka, kb)]
    prompt = (
        "Two SQL result sets answer the same question but may use different column "
        "names. Map each column in A to the column in B that represents the SAME "
        "quantity (same meaning and unit). Use null when there is no match.\n"
        f"A columns: {list(ka)}\nB columns: {list(kb)}\n"
        'Respond ONLY with JSON: {"mapping": {"<a_col>": "<b_col or null>"}}'
    )
    try:
        text = _judge_complete(LINKER, prompt, max_tokens=256, temperature=0)
        m   = re.search(r"\{.*\}", text, re.DOTALL)
        raw = json.loads(m.group()).get("mapping", {}) if m else {}
        mapping = {a: b for a, b in raw.items() if a in set(ka) and b in set(kb)}
    except Exception as e:
        # Loudly. The fallback matches identical names only, so aliased columns
        # stop linking and equivalent answers score WRONG. That is a silent change
        # to the scoring rule, and it used to happen to anyone without an AWS
        # account, because this call was hardcoded to Bedrock. Once per
        # process: it fires per column pair and the run should stay readable.
        global _COL_LINK_WARNED
        if not _COL_LINK_WARNED:
            _COL_LINK_WARNED = True
            print(f"WARNING: column linker ({LINKER}) failed: {e!r}\n"
                  f"  Falling back to exact column-name matching for the whole run. "
                  f"Answers that are correct but name their columns differently "
                  f"will score as failures. Set judges.linker in the model config "
                  f"to a model you can reach.", file=sys.stderr)
        mapping = {k: k for k in ka if k in set(kb)}   # fallback: shared names only
    with _COL_LINK_LOCK:
        return _COL_LINK_CACHE.setdefault((ka, kb), mapping)


def _results_match(res_a: str, res_b: str, tol: float = AGREEMENT_TOL) -> bool:
    """Whether two result-set strings are equivalent within `tol` (rounding-safe).
    Columns are entity-linked first so differently-aliased value columns are still
    compared by value, not silently skipped."""
    a, b = _parse_result(res_a), _parse_result(res_b)
    if not a and not b:
        return True
    if not a or not b or len(a) != len(b):
        return False
    mapping = _link_columns(sorted({k for r in a for k in r}),
                            sorted({k for r in b for k in r}))
    if not mapping:
        return False
    # Project BOTH sides onto the linked columns (in a's namespace) so they share a
    # key set -> identical sort order -> rows align. Extra/unmapped columns (e.g. a
    # candidate that also returned id/mrr/tier) are ignored, not compared.
    a = [{ac: r[ac] for ac in mapping if ac in r} for r in a]
    b = [{ac: r[bc] for ac, bc in mapping.items() if bc in r} for r in b]
    _key = lambda r: json.dumps(r, sort_keys=True, default=str)
    for ra, rb in zip(sorted(a, key=_key), sorted(b, key=_key)):
        common = set(ra.keys()) & set(rb.keys())
        if not common:
            return False
        for k in common:
            x, y = ra[k], rb[k]
            try:
                fx, fy = float(x), float(y)
                if abs(fx - fy) / max(abs(fx), abs(fy), 1e-9) > tol:
                    return False
            except (ValueError, TypeError):
                if str(x) != str(y):
                    return False
    return True


def annotators_agree(results_a: list, results_b: list) -> bool:
    """Two annotators agree if any of their non-empty result-set strings match.
    Both-empty does NOT count: a question no annotator can answer has no usable
    ground truth and is excluded, not asserted as 'no rows'."""
    a = [r for r in results_a if r and not r.startswith("Error:") and r != "(empty result)"]
    b = [r for r in results_b if r and not r.startswith("Error:") and r != "(empty result)"]
    if not a or not b:
        return False
    return any(_results_match(x, y) for x in a for y in b)


def majority_vote_gt(annotator_runs: dict) -> dict:
    """Build ground truth from independent annotator runs.

    `annotator_runs` is {annotator_name: run_candidate(...) result dict}. Annotators
    are clustered by result-set agreement; if a cluster of >=2 exists, its first
    member's run becomes the ground truth. If every annotator disagrees, the
    question is excluded. Replaces the v1 single-Opus stability filter and the
    manual exclusion list."""
    names = list(annotator_runs)
    res   = {n: annotator_runs[n].get("sql_results", []) for n in names}

    clusters: list[list[str]] = []
    for n in names:
        for cl in clusters:
            if annotators_agree(res[n], res[cl[0]]):
                cl.append(n)
                break
        else:
            clusters.append([n])
    clusters.sort(key=len, reverse=True)

    top = clusters[0] if clusters else []
    if len(top) >= 2:
        # Pick the representative agreer at random so the stored GT bytes don't skew
        # to one annotator's formatting. Deterministic in the agreed content (seeded
        # by agreer names + their result sets) -> reproducible across re-runs.
        _key   = "|".join(sorted(top)) + "::" + "|".join("".join(res[n]) for n in sorted(top))
        winner = random.Random(_key).choice(sorted(top))
        run    = annotator_runs[winner]
        return {"excluded": False, "agreers": top, "gt_from": winner,
                "gt_results": run.get("sql_results", []),
                "gt_answer":  run.get("final_answer", ""),
                "gt_sql":     run.get("sqls", []),
                "clusters":   clusters}
    return {"excluded": True, "agreers": [], "gt_from": None,
            "gt_results": [], "gt_answer": "", "gt_sql": [],
            "clusters": clusters, "reason": "annotators fully disagree"}


def bt_sigma_aggregate(*args, **kwargs):
    """Bradley-Terry + per-judge reliability (sigma) jury aggregation
    (arXiv 2602.16610). Deferred: only worthwhile once the benchmark
    scales past ~71 questions. Use majority_vote_gt / judge_panel until then."""
    raise NotImplementedError(
        "BT-sigma jury deferred until the benchmark scales past ~71 questions"
    )
