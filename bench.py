"""
Candidate runners and blind judge for the synthetic text2sql benchmark.

ch_query and system_prompt are passed as arguments to keep DB/schema
concerns in the notebook and infrastructure concerns here.
"""
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, NamedTuple
from urllib.parse import quote

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
    GATEWAY_CANDIDATES, GEMINI_CANDIDATES, GEMINI_GLOBAL, JUDGE_MODEL,
    LIBRECHAT_CANDIDATES, LINKER,
    MODELS,
    JUDGE_MODEL_IDS, JUDGE_PROVIDER, JUDGE_SEATS, MANTLE_CANDIDATES,
    MANTLE_RESPONSES_CANDIDATES, OPENAI_CANDIDATES, OPENAI_RESPONSES_ONLY,
    RETIRED_CANDIDATES,
    SCORE_ABS_TOL, SCORE_REL_TOL, SCORE_ROUND_DECIMALS,
)

# Overridable for sensitivity sweeps: the budget is never announced to
# the model, so runs at different budgets share a distribution over early turns.
MAX_TURNS   = int(os.environ.get("DAM_MAX_TURNS", "60"))
EVAL_SEED   = 42

# Error marker for a turn truncated by the output-token cap (Messages API
# stop_reason "max_tokens" / OpenAI finish_reason "length"). The answer is
# partial or empty, so the run is flagged rather than scored as complete; the
# sweep treats an empty-answer truncation like a max-turns exhaustion.
ERR_MAX_OUTPUT_TOKENS = "max output tokens"

# The LibreChat runner could not read the agent's trace (ingestion lag, a sessionId
# mismatch, or no reader configured). The run is an infrastructure unknown, not a
# model failure, so the eval loop routes it to an 'error' outcome rather than letting
# an empty result set score as a fail. See 06_eval.score_pair.
ERR_LIBRECHAT_NO_TRACE = "librechat trace unavailable"





# Result-set numeric agreement policy. Two numbers agree when they are within
# the absolute OR the relative tolerance (math.isclose). Values are first rounded
# to `round_decimals` places, or kept at full precision when it is None.
#
# The board is a currency warehouse, so the default rounds to cents and allows 5%
# relative drift: it catches 59K vs 70K and ignores rounding. That default is
# wrong for a warehouse of concentrations, p-values or dose-response, where a
# wrong answer by a factor of two sits inside 5% and two small values both round
# to 0.00. A non-currency operator overrides these in the `scoring` section of
# the model config (see config/models.example.yaml); the values come from
# registry.py so annotate-time and eval-time share one policy.
class ComparisonPolicy(NamedTuple):
    round_decimals: int | None
    rel_tol: float
    abs_tol: float


SCORING = ComparisonPolicy(SCORE_ROUND_DECIMALS, SCORE_REL_TOL, SCORE_ABS_TOL)

# Back-compat alias: the sensitivity scripts read bench.AGREEMENT_TOL.
AGREEMENT_TOL = SCORING.rel_tol

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
        # Serialises resolve-and-refresh. `refresh()` mutates the credential in
        # place, so without this a worker can read `.token` while another is
        # midway through replacing it and send a torn value.
        self._lock = threading.Lock()

    def _token(self, force: bool = False) -> str:
        with self._lock:
            if self._creds is None:
                self._creds, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            if force or not self._creds.valid:
                self._creds.refresh(google.auth.transport.requests.Request())
            return self._creds.token

    def auth_flow(self, request):
        # Retry once on 401 with a forced refresh. `.valid` is not sufficient:
        # a token can be accepted locally and rejected by Vertex, which answers
        # 401 ACCESS_TOKEN_TYPE_UNSUPPORTED rather than anything expiry-shaped.
        # The retry layer cannot cover this — 401 is deliberately not in
        # _RETRYABLE_STATUS, since for every other provider it means a bad key —
        # so a lost token costs the whole question. Measured: 6 of 201 questions
        # on a run that outlived one token lifetime.
        request.headers["Authorization"] = f"Bearer {self._token()}"
        response = yield request
        if response.status_code == 401:
            request.headers["Authorization"] = f"Bearer {self._token(force=True)}"
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
                 "calls", "missing", "per_turn")

    def __init__(self):
        self.prompt = self.completion = self.reasoning = 0
        self.cache_read = self.cache_write = 0
        self.calls = self.missing = 0
        # One entry per turn, which the sums cannot reconstruct in either direction.
        #
        # A sum answers "what did the run cost". It cannot answer how close a turn
        # came to the per-call output cap, and that cap is what truncates a run
        # (ERR_MAX_OUTPUT_TOKENS) and therefore scores it as a failure. From
        # completion_tokens / api_calls you get the mean turn and never the largest,
        # and the gap between them is unbounded: measured on the board, that
        # inference leaves the peak somewhere in an interval 23 to 91 points wide.
        #
        # Storing the vector rather than a single max because the statistic worth
        # having is not decided yet. Floor, median, p90 and peak all fall out of it,
        # and so does input growth per turn, which is what the prompt-caching work
        # needed and could not see: it could measure that 42.9% of input was served
        # from cache overall, but not which turn the cache started hitting.
        #
        # Cheap: about 10 to 60 entries per run, a few hundred KB across a full
        # board against a results file already in the tens of MB.
        self.per_turn = []

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
        # Recorded raw, in the provider's own accounting, with nothing derived.
        # `reasoning` is NOT added to `completion`: every provider that reports it
        # counts it inside its completion figure, so adding them would double-count
        # exactly the models the field exists to illuminate, and would overstate
        # how close a turn came to the output cap. Kept as its own element so a
        # reader can still see the split per turn.
        self.per_turn.append([prompt, completion, reasoning, c_read, c_write])

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

    def add_langfuse(self, usage_details, *, prompt_is_inclusive: bool = False):
        """One Langfuse observation's usageDetails -> one turn.

        The LibreChat runner does not hold a provider response; it reads usage back
        off the agent's trace, in whatever keys LibreChat's integration wrote. So
        the disjointness the other add_* methods inherit from a provider's own
        accounting has to be re-established here from key names, and anything not
        recognised is left at zero rather than guessed.

        `input`'s convention is the one thing a dict cannot settle: OpenAI-style
        counts cache reads inside it, Anthropic-style reports them beside it. We do
        not guess. Pass prompt_is_inclusive=True once a real trace shows the inclusive
        form, and the cache figures come out of the prompt so the buckets stay
        disjoint; left False, prompt is recorded as given and a nonzero cache_read is
        the signal that the assumption still needs checking against a trace.
        """
        ud = usage_details or {}
        prompt     = self._int(ud, "input", "prompt_tokens", "input_tokens")
        completion = self._int(ud, "output", "completion_tokens", "output_tokens")
        c_read = self._int(ud, "cache_read_input_tokens", "cacheReadInputTokens", "cached_tokens")
        c_write = self._int(ud, "cache_creation_input_tokens", "cacheWriteInputTokens",
                            "cache_write_tokens")
        if prompt_is_inclusive:
            prompt = max(prompt - c_read - c_write, 0)
        self._record(ud or None, prompt, completion, 0, c_read, c_write)

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
            # One [prompt, completion, reasoning, cache_read, cache_write] per turn
            # that reported usage, in call order. Every per-turn statistic derives
            # from this and none of them from the sums: peak output against the cap,
            # median turn, and input growth across the transcript. Turns that
            # reported no usage are absent, so len() can be below api_calls; that
            # gap is `calls_missing_usage`.
            "per_turn": self.per_turn,
            # >0 means some turns reported no usage, so the totals are a LOWER
            # bound and must not be presented as exact.
            "calls_missing_usage": self.missing,
        }


def run_candidate_bedrock(
    nl_question: str,
    model_id: str,
    ch_query: Callable[[str], str],
    system_prompt: str,
    gate: Callable[[str, str], str | None] | None = None,
) -> dict:
    start    = time.time()
    # PROMPT CACHING, the same two static breakpoints as the native
    # Anthropic path. Converse caches nothing without an explicit cachePoint,
    # while the OpenAI, Fireworks and Vertex paths cache unasked, so writing no
    # cache code produced DIFFERENT behaviour per provider rather than uniform
    # behaviour. Mantle proves this is opt-in rather than a Bedrock limit: it is
    # Bedrock through an OpenAI-compatible surface and caches 22.5% by itself,
    # while Converse cached 0%.
    #
    # System prompt and tool list only, both byte-identical across every turn and
    # every question. The growing transcript is left uncached on purpose: see the
    # note in run_candidate_messages_api for why a rolling breakpoint is out of
    # scope for this methodology.
    system   = [{"text": system_prompt}, {"cachePoint": {"type": "default"}}]
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
            toolConfig={"tools": TOOLS_BEDROCK
                        + [{"cachePoint": {"type": "default"}}]},
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
                sql_results.append(result)                 # store the clean result the judge grades
                note = gate(q, result) if gate is not None else None
                shown = f"{note}\n\n{result}" if note is not None else result
                tool_results.append({
                    "toolResult": {"toolUseId": tc["toolUseId"],
                                   "content":   [{"text": shown}]}
                })
            messages.append({"role": "user", "content": tool_results})
        else:
            break

    return {"sqls": sqls, "sql_results": sql_results,
            "final_answer": "", "turns": MAX_TURNS, "served_model": model_id,
            "latency": round(time.time() - start, 2), "error": f"max turns ({MAX_TURNS})",
            "usage": usage.as_dict()}


def emits_inline_reasoning(model_id: str) -> bool:
    """Does this model spend output tokens thinking before it says anything?

    Extracted because it was written twice and the copies drifted.
    `run_candidate_openai_compat` had the full list; `12_contamination_probe`
    had a three-term version missing every Gemini, so the probe gave
    gemini-3.1-pro 100 output tokens where the eval gives it 8192, and the model
    either truncated mid-thought or emitted a stub. That looked like a model
    declining to recall an id and was an unfunded budget.

    The registry's `reasoning: true` flag is the forward-looking half; the
    substring list is history, kept so no board candidate's budget changes.
    """
    return (model_id.startswith(("o", "gpt-5"))
            or model_id in BEDROCK_REASONING
            or model_id.startswith("google/gemini-2.5-pro")
            or model_id.startswith("google/gemini-3")
            or "deepseek-v4" in model_id
            or "kimi" in model_id
            or "qwen3p8-max" in model_id)


def run_candidate_openai_compat(
    nl_question: str,
    model_id: str,
    client: OpenAI,
    ch_query: Callable[[str], str],
    system_prompt: str,
    gate: Callable[[str, str], str | None] | None = None,
) -> dict:
    start        = time.time()
    # o-series, gpt-5.x, and anything the registry flags `reasoning` take
    # max_completion_tokens rather than max_tokens. gemini 2.5-pro and gemini 3.x,
    # deepseek-v4, and Kimi (K2 Thinking / K2.6) are thinking models that emit
    # reasoning inline -> need a larger budget so reasoning tokens don't starve the
    # answer (at 2048 Kimi K2.6 truncated mid-reasoning on 18% of questions before it
    # could even issue a query).
    _is_reasoning = (model_id.startswith("o") or model_id.startswith("gpt-5")
                     or model_id in BEDROCK_REASONING)
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
    _token_limit  = 8192 if emits_inline_reasoning(model_id) else 2048
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
                sql_results.append(result)                 # store the clean result the judge grades
                # An optional gate may inspect the query and its result and prepend a
                # short note to the tool message the model reads next, steering the next
                # turn without withholding the data or mutating the stored result;
                # None leaves the message unchanged.
                note = gate(q, result) if gate is not None else None
                shown = f"{note}\n\n{result}" if note is not None else result
            else:
                shown = "Error: unknown tool."
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": shown})

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
    gate: Callable[[str, str], str | None] | None = None,
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
                sql_results.append(result)                 # store the clean result the judge grades
                note = gate(q, result) if gate is not None else None
                shown = f"{note}\n\n{result}" if note is not None else result
            else:
                shown = "Error: unknown tool."
            pending.append({"type": "function_call_output",
                            "call_id": fc.call_id, "output": shown})

    return {"sqls": sqls, "sql_results": sql_results,
            "final_answer": "", "turns": MAX_TURNS, "served_model": served,
            "latency": round(time.time() - start, 2), "error": f"max turns ({MAX_TURNS})",
            "usage": usage.as_dict()}


def run_candidate_messages_api(
    nl_question: str,
    model_id: str,
    ch_query: Callable[[str], str],
    system_prompt: str,
    gate: Callable[[str, str], str | None] | None = None,
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

    # PROMPT CACHING. Anthropic caches nothing without an explicit
    # cache_control breakpoint, while the OpenAI, Fireworks and Vertex paths cache
    # automatically. Without this the Claude models paid full input price on every
    # turn of a 60-turn loop and their competitors did not, which made the board's
    # cost column a measurement of our client code rather than of the models.
    #
    # Two static breakpoints, on the system prompt and on the tool list. Both are
    # byte-identical across every turn AND every question, so the prefix is shared
    # by concurrent workers and re-hit for the whole run, not just within one
    # question.
    #
    # STATIC ONLY: NO PROVIDER-SPECIFIC OPTIMIZATION. The benchmark's methodology
    # is that every provider path gets the same treatment, so the board measures
    # models rather than how much tuning each integration received.
    #
    # Zero caching was not that treatment, it was a defect. OpenAI, Fireworks and
    # Vertex cache a stable prefix without being asked; Anthropic and Bedrock
    # Converse require an explicit opt-in for the same behaviour. So "write no
    # cache code" produced DIFFERENT behaviour per provider, not uniform behaviour.
    # These two breakpoints buy parity of intent, nothing more: cache the stable
    # prefix, which is what the automatic providers already do.
    #
    # A third, rolling breakpoint on the growing transcript was tried and rejected.
    # Measured share of input served from cache on this workload:
    #
    #     no breakpoints (what shipped)        0.0%
    #     system + tools (this)               42.9%
    #     + rolling point on the transcript   99.6%
    #
    # It works, and it is exactly the kind of change this methodology excludes: a
    # technique available on one provider's API, with no counterpart being applied
    # to the other five paths, which reach their endpoints through a shared
    # OpenAI-compatible call that exposes no cache controls at all. Adopting it
    # would make the cost column partly a record of which integrations we hand-
    # tuned. The 99.6% is recorded here so the ceiling is known rather than hidden.
    #
    # Bedrock Converse needs the same static treatment for the same reason: it
    # accepts cachePoint at system, messages and tools, and currently sets none,
    # which is why the six Claude models on that path also sat at zero.
    cached_system = [{"type": "text", "text": system_prompt,
                      "cache_control": {"type": "ephemeral"}}]
    cached_tools = [dict(t) for t in TOOLS_MESSAGES]
    cached_tools[-1]["cache_control"] = {"type": "ephemeral"}

    usage = TokenUsage()
    for turn in range(MAX_TURNS):
        resp = retry(lambda: anthropic_native.messages.create(
            model=model_id, max_tokens=16000, system=cached_system,
            messages=messages, tools=cached_tools, extra_body=extra,
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
                    sql_results.append(result)             # store the clean result the judge grades
                    note = gate(q, result) if gate is not None else None
                    shown = f"{note}\n\n{result}" if note is not None else result
                else:
                    shown = f"Error: unknown tool {block.name!r}"
                tool_results.append({"type": "tool_result",
                                     "tool_use_id": block.id, "content": shown})
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


def run_candidate(
    nl_question: str,
    model_name: str,
    model_id: str,
    ch_query: Callable[[str], str],
    system_prompt: str,
    gate: Callable[[str, str], str | None] | None = None,
) -> dict:
    if model_name in GEMINI_CANDIDATES:
        client = gemini_global_client if model_name in GEMINI_GLOBAL else gemini_client
        return run_candidate_openai_compat(nl_question, model_id, client, ch_query, system_prompt, gate)
    if model_name in OPENAI_CANDIDATES:
        if model_name in OPENAI_RESPONSES_ONLY:
            return run_candidate_responses_api(nl_question, model_id, openai_client, ch_query, system_prompt, gate)
        return run_candidate_openai_compat(nl_question, model_id, openai_client, ch_query, system_prompt, gate)
    if model_name in MANTLE_RESPONSES_CANDIDATES:
        return run_candidate_responses_api(nl_question, model_id, mantle_openai_client, ch_query, system_prompt, gate)
    if model_name in MANTLE_CANDIDATES:
        return run_candidate_openai_compat(nl_question, model_id, mantle_client, ch_query, system_prompt, gate)
    if model_name in FIREWORKS_CANDIDATES:
        return run_candidate_openai_compat(nl_question, model_id, fireworks_client, ch_query, system_prompt, gate)
    if model_name in GATEWAY_CANDIDATES:
        return run_candidate_openai_compat(nl_question, model_id, gateway_client, ch_query, system_prompt, gate)
    if model_name in ANTHROPIC_CANDIDATES:
        return run_candidate_messages_api(nl_question, model_id, ch_query, system_prompt, gate)
    if model_name in LIBRECHAT_CANDIDATES:
        # The product drives its own loop over HTTP, so there is no hook to apply the
        # in-loop gate; LibreChat candidates run ungated.
        return run_candidate_librechat(nl_question, model_id, ch_query, system_prompt)
    return run_candidate_bedrock(nl_question, model_id, ch_query, system_prompt, gate)

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

def _parse_result(result_str, decimals: int | None = SCORING.round_decimals):
    """Parse a result-set string into normalized, sorted rows.

    Numbers are rounded to `decimals` places when it is not None (the board rounds
    to cents); None keeps full precision so small-magnitude values are not
    flattened before comparison. Non-numeric cells are kept as strings.
    """
    if not result_str or result_str.startswith("Error:") or result_str == "(empty result)":
        return []
    rows = []
    for _ln in result_str.strip().splitlines():
        try:
            _row, _norm = json.loads(_ln), {}
            for k, v in _row.items():
                try:    _norm[k.lower().strip()] = round(float(v), decimals) if decimals is not None else float(v)
                except: _norm[k.lower().strip()] = str(v)
            rows.append(_norm)
        except Exception:
            continue
    return sorted(rows, key=lambda r: json.dumps(r, sort_keys=True, default=str))


_COL_LINK_CACHE: dict = {}   # (tuple(keys_a), tuple(keys_b)) -> {a_col: b_col}
_COL_LINK_LOCK  = threading.Lock()
_COL_LINK_WARNED = False
_COL_LINK_EMPTY_WARNED = False   # parsed-but-empty mapping seen (not a linker error)


def _link_columns(keys_a: list, keys_b: list) -> dict:
    """Map each column in A to the column in B that means the same quantity, so
    aliasing (e.g. total_dollar_usage <-> monthly_spend) doesn't hide a real value
    comparison. Identity when one column set contains the other (no model call);
    otherwise a cached Haiku call (temp 0 -> reproducible). Falls back to shared
    names on error; logs a parsed-but-empty mapping instead of scoring it blind.

    The eval loop calls this from many worker threads. The Haiku call runs OUTSIDE
    the lock so column-linking stays concurrent, but the cache read and write are
    locked and the write is a first-writer-wins `setdefault` -> every thread that
    misses on the same key returns the one stored mapping (temp-0 output isn't
    byte-guaranteed across calls, so last-writer-wins could otherwise hand different
    threads different mappings for the same key)."""
    ka, kb = tuple(keys_a), tuple(keys_b)
    sa, sb = set(ka), set(kb)
    # Identity when one column set contains the other (no model call): the smaller
    # set's columns all exist on the other side, so mapping each to itself scores the
    # SELECT-subset case (candidate returns the ground truth's columns plus extras)
    # on the ground truth's columns alone, instead of sending a wide list to the
    # model and reading its truncated, valid-but-empty reply as "no columns match".
    if sa <= sb or sb <= sa:
        return {k: k for k in (ka if sa <= sb else kb)}
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
        # Generous budget: a truncated reply parses as valid-but-empty JSON.
        text = _judge_complete(LINKER, prompt, max_tokens=1024, temperature=0)
        m   = re.search(r"\{.*\}", text, re.DOTALL)
        raw = json.loads(m.group()).get("mapping", {}) if m else {}
        mapping = {a: b for a, b in raw.items() if a in sa and b in sb}
        if not mapping:
            # Still a mismatch (no shared meaning), but say so once: on very wide
            # non-subset sets an empty mapping can be a truncation artifact.
            global _COL_LINK_EMPTY_WARNED
            if not _COL_LINK_EMPTY_WARNED:
                _COL_LINK_EMPTY_WARNED = True
                print(f"WARNING: column linker ({LINKER}) returned an empty mapping "
                      f"for {len(ka)}x{len(kb)} columns; scoring as a mismatch.",
                      file=sys.stderr)
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


def _results_match(res_a: str, res_b: str, policy: ComparisonPolicy = SCORING) -> bool:
    """Whether two result-set strings are equivalent under `policy` (rounding-safe).
    Columns are entity-linked first so differently-aliased value columns are still
    compared by value, not silently skipped. Two numbers agree when they are within
    the policy's absolute OR relative tolerance; the currency default is 5% relative."""
    a, b = _parse_result(res_a, policy.round_decimals), _parse_result(res_b, policy.round_decimals)
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
                if not math.isclose(fx, fy, rel_tol=policy.rel_tol, abs_tol=policy.abs_tol):
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
