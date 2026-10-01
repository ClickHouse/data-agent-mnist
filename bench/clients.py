"""Provider clients for the benchmark: auth helpers and one client per endpoint.

Every client is built at import so the runners and the judge can share them across
worker threads. Nothing here makes a network call at import: the Vertex credentials
resolve on first use, and the OpenAI-compatible clients only need a key string.
"""
import os
import threading
from pathlib import Path

import anthropic
import boto3
import botocore.auth
import botocore.awsrequest
import google.auth
import google.auth.transport.requests
import httpx
from dotenv import load_dotenv
from openai import OpenAI

# The experiment's .env, one directory above this package.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")

# Model registry, provider endpoints and judge seats live in configuration, not
# here: publishing this package would otherwise publish the catalog and our internal
# hosts, and reading the registry would keep requiring provider credentials because
# importing this module constructs the clients. See registry.py.
from registry import ENDPOINTS  # noqa: E402

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
