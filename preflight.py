"""Credential preflight, so a paid run cannot start or degrade silently.

A board run spends real money for hours. It used to start without checking it
could reach the providers it needs, and a dead credential surfaced only later and
quietly: a failed judge became a per-question error, and a failed column linker
printed one warning and then scored the whole run with exact-name matching. The
judge seats and the linker are separate credentials on separate providers from
the candidate, and they are the ones that fail invisibly because their failure
does not stop the expensive half.

This resolves every provider the selected candidates, judge seats and linker will
touch, and fails before the first paid call when a credential is missing or not
usable now:

  - AWS (bedrock and mantle): `sts get-caller-identity`, so an unusable or absent
    profile aborts up front. The SSO access-token expiry is compared against a
    rough projected run length and surfaced as a warning (a hard stop would need
    the SSO session expiry, not the access-token expiry, which refreshes).
  - Vertex (gemini): resolve application-default credentials and mint one token.
  - OpenAI, Fireworks, gateway, Anthropic: the API key must be present.

The in-flight Vertex 401 retry (bench `_GCPAuth.auth_flow`) already rescues a
token rejected mid-run; this closes the different gap of a credential that is
missing or unusable at launch. No paid completion is issued here: credential
resolution plus key presence covers the failures that were actually observed,
without coupling to a per-provider model id.

    imported and called by 06_eval.py after the board guard, before the pool.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import JUDGE_PROVIDER, MODELS

# Providers whose credential is an AWS profile / SSO token (SigV4 or boto3).
AWS_PROVIDERS = {"bedrock", "mantle"}
# Providers whose credential is Vertex application-default credentials. "gemini"
# is the MODELS provider id; "google" is the judge seat label bench serves on the
# same Vertex client, so a judge named only in extra_ids resolves to it.
GCP_PROVIDERS = {"gemini", "google"}
# Providers whose credential is a bearer API key, and the env var that holds it.
KEY_ENV = {"openai": "OPENAI_API_KEY", "fireworks": "FIREWORKS_API_KEY",
           "anthropic": "ANTHROPIC_API_KEY", "gateway": "INFERENCE_GATEWAY_KEY"}

SSO_CACHE_DIR = Path.home() / ".aws" / "sso" / "cache"
# Conservative per-(question, candidate) wall time when the results file carries
# no latency prior, used only for the SSO-expiry warning projection.
DEFAULT_RUN_SECONDS = 90.0


class PreflightError(RuntimeError):
    """One or more providers the run needs are missing or not usable at launch."""


def provider_of(key: str) -> str | None:
    e = MODELS.get(key)
    if e:
        return e.get("provider")
    # Judge seats and the linker may name a model that lives only in
    # judges.extra_ids, not MODELS. Its provider is the seat it sits in.
    return JUDGE_PROVIDER.get(key)


def providers_needed(candidate_keys, judge_keys, linker_key) -> set[str]:
    keys = set(candidate_keys) | set(judge_keys) | {linker_key}
    return {p for p in (provider_of(k) for k in keys) if p}


def _parse_iso(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def sso_expiry(cache_dir: Path = SSO_CACHE_DIR) -> datetime | None:
    """Latest access-token `expiresAt` in the AWS SSO cache, or None if there is
    no cache to read. boto3 refreshes the access token from the SSO session, so
    this is a soft signal, not a session deadline."""
    latest = None
    if not cache_dir.exists():
        return None
    for f in sorted(cache_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except (ValueError, OSError):
            continue
        # Only access-token files carry the token expiry. Client-registration
        # files sit in the same dir and their `expiresAt` is the ~90-day
        # registration expiry, which would mask an expiring token.
        if "accessToken" not in data:
            continue
        exp = _parse_iso(data.get("expiresAt", ""))
        if exp and (latest is None or exp > latest):
            latest = exp
    return latest


def project_duration_seconds(candidate_keys, n_questions: int, workers: int,
                             results_path: Path | None) -> float:
    """Rough wall-clock estimate: per-model median run latency (from the results
    file when present) times the question count, over the worker count. An
    estimate for the expiry warning, not a schedule."""
    per_model: dict[str, float] = {}
    if results_path and Path(results_path).exists():
        lat: dict[str, list[float]] = defaultdict(list)
        for line in Path(results_path).read_text().splitlines():
            if not line.strip():
                continue
            for m, c in (json.loads(line).get("candidates") or {}).items():
                if c.get("latency"):
                    lat[m].append(float(c["latency"]))
        per_model = {m: statistics.median(v) for m, v in lat.items() if v}
    serial = n_questions * sum(per_model.get(m, DEFAULT_RUN_SECONDS) for m in candidate_keys)
    return serial / max(workers, 1)


def check_aws(projected_end: datetime, *, boto3_module=None) -> list[str]:
    b = boto3_module
    if b is None:
        import boto3 as b
    try:
        ident = b.client("sts").get_caller_identity()
    except Exception as e:                          # noqa: BLE001 - report any failure
        raise PreflightError(
            f"AWS credentials are not usable (sts get-caller-identity failed): {e}. "
            f"Log in first, for example `aws sso login`.")
    lines = [f"AWS ok (account {ident.get('Account', '?')})"]
    exp = sso_expiry()
    if exp:
        if exp < projected_end:
            lines.append(
                f"WARNING: AWS SSO access token expires {exp.isoformat(timespec='minutes')}, "
                f"before the run is projected to finish {projected_end.isoformat(timespec='minutes')}. "
                f"boto3 refreshes the token from the SSO session, but if that session "
                f"has expired the run will fail on the linker or a judge. Re-run "
                f"`aws sso login` if unsure.")
        else:
            lines.append(f"AWS SSO token valid until {exp.isoformat(timespec='minutes')}")
    return lines


def check_gcp(*, default_fn=None) -> list[str]:
    try:
        import google.auth
        import google.auth.transport.requests as greq
        default = default_fn or google.auth.default
        creds, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(greq.Request())
        if not getattr(creds, "token", None):
            raise RuntimeError("no token minted from application-default credentials")
    except Exception as e:                          # noqa: BLE001 - report any failure
        raise PreflightError(
            f"Vertex application-default credentials are not usable: {e}. "
            f"Run `gcloud auth application-default login`.")
    return ["Vertex ADC ok"]


def check_key(provider: str) -> list[str]:
    env = KEY_ENV[provider]
    if not os.environ.get(env):
        raise PreflightError(f"{provider} is needed but {env} is unset")
    return [f"{provider} key present ({env})"]


def preflight(candidate_keys, judge_keys, linker_key, n_questions: int, workers: int,
              results_path: Path | None = None, now: datetime | None = None,
              *, boto3_module=None, gcp_default_fn=None) -> list[str]:
    """Check every provider the run will touch. Returns summary lines on success;
    raises PreflightError naming every provider that failed. The injected
    boto3_module and gcp_default_fn keep it testable without network or spend."""
    now = now or datetime.now(timezone.utc)
    provs = providers_needed(candidate_keys, judge_keys, linker_key)
    # Every provider must map to a check. A provider that matches none of the sets
    # would be silently skipped, which is the failure this preflight exists to stop.
    unchecked = provs - AWS_PROVIDERS - GCP_PROVIDERS - set(KEY_ENV)
    if unchecked:
        raise PreflightError(
            f"no preflight check for provider(s): {', '.join(sorted(unchecked))}. "
            f"Their credential would go unchecked; add each to the right provider set.")
    dur = project_duration_seconds(candidate_keys, n_questions, workers, results_path)
    end = now + timedelta(seconds=dur)

    lines = [f"preflight: {len(provs)} provider(s) to reach "
             f"({', '.join(sorted(provs))}); projected run <= {dur / 3600:.1f}h"]
    errors: list[str] = []

    def run(check):
        try:
            lines.extend(check())
        except PreflightError as e:
            errors.append(str(e))

    if provs & AWS_PROVIDERS:
        run(lambda: check_aws(end, boto3_module=boto3_module))
    if provs & GCP_PROVIDERS:
        run(lambda: check_gcp(default_fn=gcp_default_fn))
    for p in sorted(provs & set(KEY_ENV)):
        run(lambda p=p: check_key(p))

    if errors:
        raise PreflightError(
            "credential preflight failed, no paid call was made:\n  - "
            + "\n  - ".join(errors))
    return lines
