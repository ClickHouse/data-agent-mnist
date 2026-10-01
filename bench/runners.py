"""Agentic tool loops, one per provider API, and the dispatcher that picks one."""
import json
import os
import re
import time
from typing import Callable

from openai import OpenAI

from registry import (
    ADAPTIVE_THINKING_ONLY, ANTHROPIC_CANDIDATES, BEDROCK_REASONING, EFFORT_CAPABLE,
    FIREWORKS_CANDIDATES, GATEWAY_CANDIDATES, GEMINI_CANDIDATES, GEMINI_GLOBAL,
    LIBRECHAT_CANDIDATES, MANTLE_CANDIDATES, MANTLE_RESPONSES_CANDIDATES,
    OPENAI_CANDIDATES, OPENAI_RESPONSES_ONLY,
)

from . import clients, librechat
from .concurrency import retry
from .usage import TokenUsage

# Overridable for sensitivity sweeps: the budget is never announced to
# the model, so runs at different budgets share a distribution over early turns.
MAX_TURNS   = int(os.environ.get("DAM_MAX_TURNS", "60"))

# Error marker for a turn truncated by the output-token cap (Messages API
# stop_reason "max_tokens" / OpenAI finish_reason "length"). The answer is
# partial or empty, so the run is flagged rather than scored as complete; the
# sweep treats an empty-answer truncation like a max-turns exhaustion.
ERR_MAX_OUTPUT_TOKENS = "max output tokens"

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

# ── Candidate runners ─────────────────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def _bedrock_converse_with_retry(max_retries: int = 5, **kwargs):
    # Retries throttling + service-unavailable + 5xx (see concurrency.retry); matters most
    # under the concurrent eval loop, where many converse calls are in flight at once.
    return retry(lambda: clients.bedrock.converse(**kwargs), what="bedrock.converse",
                 max_retries=max_retries)


# Bedrock-served models that emit reasoning before the answer, so they need the larger
# output budget (see run_candidate_bedrock). Matched as substrings of the model id.
#
# Verified by probing each Claude on the board with a converse call: only Opus 5 returns a
# reasoningContent block by default. opus48, opus47, sonnet5 and sonnet46 return text only,
# so they are NOT listed here — their published numbers were produced without reasoning and
# raising their budget would change what the board measured without re-running it.


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
        resp = retry(lambda: clients.anthropic_native.messages.create(
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



def run_candidate(
    nl_question: str,
    model_name: str,
    model_id: str,
    ch_query: Callable[[str], str],
    system_prompt: str,
    gate: Callable[[str, str], str | None] | None = None,
) -> dict:
    if model_name in GEMINI_CANDIDATES:
        client = clients.gemini_global_client if model_name in GEMINI_GLOBAL else clients.gemini_client
        return run_candidate_openai_compat(nl_question, model_id, client, ch_query, system_prompt, gate)
    if model_name in OPENAI_CANDIDATES:
        if model_name in OPENAI_RESPONSES_ONLY:
            return run_candidate_responses_api(nl_question, model_id, clients.openai_client, ch_query, system_prompt, gate)
        return run_candidate_openai_compat(nl_question, model_id, clients.openai_client, ch_query, system_prompt, gate)
    if model_name in MANTLE_RESPONSES_CANDIDATES:
        return run_candidate_responses_api(nl_question, model_id, clients.mantle_openai_client, ch_query, system_prompt, gate)
    if model_name in MANTLE_CANDIDATES:
        return run_candidate_openai_compat(nl_question, model_id, clients.mantle_client, ch_query, system_prompt, gate)
    if model_name in FIREWORKS_CANDIDATES:
        return run_candidate_openai_compat(nl_question, model_id, clients.fireworks_client, ch_query, system_prompt, gate)
    if model_name in GATEWAY_CANDIDATES:
        return run_candidate_openai_compat(nl_question, model_id, clients.gateway_client, ch_query, system_prompt, gate)
    if model_name in ANTHROPIC_CANDIDATES:
        return run_candidate_messages_api(nl_question, model_id, ch_query, system_prompt, gate)
    if model_name in LIBRECHAT_CANDIDATES:
        # The product drives its own loop over HTTP, so there is no hook to apply the
        # in-loop gate; LibreChat candidates run ungated.
        return librechat.run_candidate_librechat(nl_question, model_id, ch_query, system_prompt)
    return run_candidate_bedrock(nl_question, model_id, ch_query, system_prompt, gate)
