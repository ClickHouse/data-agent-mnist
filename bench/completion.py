"""One-shot completion routed to a judge's provider, and its token budget rule.

Used by the judge panel for verdicts and by the column linker for scoring. Kept apart
from judge.py because scoring.py needs `_judge_complete` and judge.py needs
`scoring.annotators_agree`.
"""
from registry import ALL_CANDIDATES, BEDROCK_REASONING, JUDGE_MODEL, JUDGE_MODEL_IDS, JUDGE_PROVIDER, MODELS

from . import clients
from .concurrency import retry
from .runners import _strip_thinking

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
        resp = retry(lambda: clients.bedrock.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig=_cfg,
        ), what=f"judge.converse[{judge_name}]")
        return "".join(b.get("text", "") for b in resp["output"]["message"]["content"]).strip()

    if provider == "anthropic":
        resp = retry(lambda: clients.anthropic_native.messages.create(
            model=model_id, max_tokens=max(max_tokens, 1024),
            messages=[{"role": "user", "content": prompt}],
            **({} if _temp is None else {"temperature": _temp}),
        ), what=f"judge.messages[{judge_name}]")
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    chat_clients = {"openai": clients.openai_client, "google": clients.gemini_client,
                    "gemini": clients.gemini_client, "mantle": clients.mantle_client,
                    "fireworks": clients.fireworks_client, "gateway": clients.gateway_client}
    if provider not in chat_clients:
        raise ValueError(
            f"judge {judge_name!r} has provider {provider!r}, which has no judge "
            f"client. Supported: bedrock, anthropic, {', '.join(sorted(chat_clients))}.")
    client = chat_clients[provider]
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
