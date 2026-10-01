"""Retry with backoff and a thread-pool map, shared by every stage that calls models."""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable

# ── Concurrency + retry ─────────────────────────────────────────────────────────
# The annotate/eval loops fan work out across a thread pool (one (question,
# candidate) pair or annotator run per task). All the SDK clients in clients.py are
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
