"""Guards for the Vertex credential flow.

Vertex answers `401 ACCESS_TOKEN_TYPE_UNSUPPORTED` for a token it will not
accept, which reads like a misconfigured credential and is not one. Two things
made that cost whole questions on a paid run, and both are pinned here.

A TORN TOKEN. `refresh()` mutates the credential in place, so a worker could read
`.token` while another was midway through replacing it. Under eight workers this
appeared as scattered 401s, 6 of 201 questions on a run that outlived one token
lifetime, and looked like a provider fault rather than a race.

A 401 THAT CANNOT BE RETRIED. 401 is deliberately absent from
`_RETRYABLE_STATUS`, because for every other provider it means a bad key and
retrying five times would only delay a clear failure. So the generic retry layer
cannot rescue a rejected Vertex token; the auth flow has to do it itself, once,
with a forced refresh.

Hermetic: `google.auth.default` is stubbed, so this runs with no ADC and no
network, which is also the state CI is in.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import pytest

os.environ.setdefault("DAM_DATA_ROOT", "/tmp/dam-gcp-auth-test")
os.environ.setdefault("DAM_MODELS_CONFIG",
                      str(Path(__file__).resolve().parents[1] / "config" / "models.example.yaml"))
os.environ.setdefault("OPENAI_API_KEY", "placeholder")
os.environ.setdefault("ANTHROPIC_API_KEY", "placeholder")
os.environ.setdefault("FIREWORKS_API_KEY", "placeholder")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

bench = pytest.importorskip("bench")
httpx = pytest.importorskip("httpx")


class _FakeCreds:
    """Mimics the one behaviour that matters: refresh() replaces .token in place."""

    def __init__(self):
        self.valid = False
        self.token = None
        self.refreshes = 0

    def refresh(self, _request):
        self.refreshes += 1
        self.valid = True
        self.token = f"tok{self.refreshes}"


@pytest.fixture
def auth(monkeypatch):
    creds = _FakeCreds()
    monkeypatch.setattr(bench.google.auth, "default", lambda **_: (creds, "proj"))
    a = bench._GCPAuth()
    return a, creds


def test_401_retries_once_with_a_fresh_token(auth):
    a, creds = auth
    flow = a.auth_flow(httpx.Request("POST", "https://vertex.invalid/chat"))
    first = next(flow)
    assert first.headers["Authorization"] == "Bearer tok1"

    second = flow.send(httpx.Response(401, request=first))
    assert second.headers["Authorization"] == "Bearer tok2", "must not resend the rejected token"
    assert creds.refreshes == 2

    with pytest.raises(StopIteration):
        flow.send(httpx.Response(401, request=second))  # once, not forever


def test_a_successful_response_does_not_retry(auth):
    a, creds = auth
    flow = a.auth_flow(httpx.Request("POST", "https://vertex.invalid/chat"))
    req = next(flow)
    with pytest.raises(StopIteration):
        flow.send(httpx.Response(200, request=req))
    assert creds.refreshes == 1, "a 200 must not trigger a refresh"


def test_concurrent_requests_never_see_a_torn_token(auth):
    a, creds = auth
    seen, errors = [], []

    def hit(_):
        try:
            req = httpx.Request("GET", "https://vertex.invalid/x")
            next(a.auth_flow(req))
            seen.append(req.headers["Authorization"])
        except Exception as e:            # noqa: BLE001 - reported, not raised, per worker
            errors.append(e)

    threads = [threading.Thread(target=hit, args=(i,)) for i in range(64)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert set(seen) == {"Bearer tok1"}, "every worker must see the same resolved token"
    assert creds.refreshes == 1, "64 workers must not each refresh"
    assert all("None" not in s for s in seen)
