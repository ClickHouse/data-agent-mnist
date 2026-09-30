"""Guards for the credential preflight.

The preflight exists to turn a silent mid-run credential failure into an upfront
abort, so its own failure and success paths have to be exact. These drive every
check with injected AWS and GCP stubs and canned env, so nothing reaches a
provider and nothing is spent.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

os.environ.setdefault("DAM_MODELS_CONFIG",
                      str(Path(__file__).resolve().parents[1] / "config" / "models.example.yaml"))
os.environ.setdefault("DAM_DATA_ROOT", "/tmp/dam-preflight-test")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

preflight = pytest.importorskip("preflight")


class _STS:
    def __init__(self, fail=False):
        self._fail = fail

    def get_caller_identity(self):
        if self._fail:
            raise RuntimeError("ExpiredToken: the security token has expired")
        return {"Account": "123456789012"}


class _Boto:
    """Stands in for the boto3 module: only .client('sts') is used."""

    def __init__(self, fail=False):
        self._fail = fail

    def client(self, name):
        assert name == "sts"
        return _STS(self._fail)


class _Creds:
    def __init__(self, token="tok"):
        self.token = None
        self._t = token

    def refresh(self, _req):
        self.token = self._t


def _gcp_default_ok(scopes=None):
    return _Creds(), "project"


def _gcp_default_broken(scopes=None):
    raise RuntimeError("could not find application-default credentials")


@pytest.fixture
def patched_models(monkeypatch):
    """A controlled model->provider registry. The real registry is a process-wide
    singleton loaded from whichever test imported it first, so provider-resolution
    tests inject their own rather than depend on that ambient config."""
    m = {"oai": {"id": "oai", "provider": "openai"},
         "ant": {"id": "ant", "provider": "anthropic"},
         "fw": {"id": "fw", "provider": "fireworks"}}
    monkeypatch.setattr(preflight, "MODELS", m)
    return m


# ── provider resolution ───────────────────────────────────────────────────────

def test_providers_needed_maps_every_role_through_the_registry(patched_models):
    provs = preflight.providers_needed(["oai"], {"ant"}, "fw")
    assert provs == {"openai", "anthropic", "fireworks"}


def test_providers_needed_resolves_extra_id_judge_seats(patched_models, monkeypatch):
    # A judge seat (or the linker) can name a model that lives only in
    # judges.extra_ids, not MODELS; its provider is the seat it sits in. The
    # preflight must still reach that provider, or its credential goes unchecked.
    monkeypatch.setattr(preflight, "JUDGE_PROVIDER", {"gpt-5.4": "openai"})
    provs = preflight.providers_needed(["ant"], {"gpt-5.4"}, "ant")
    assert "openai" in provs


def test_google_seat_label_routes_to_the_gcp_check(patched_models, monkeypatch):
    # bench serves the "google" judge seat on the Vertex client, so a judge named
    # only in extra_ids resolves to the label "google" and must reach the GCP check.
    monkeypatch.setattr(preflight, "JUDGE_PROVIDER", {"gseat": "google"})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    with pytest.raises(preflight.PreflightError) as e:
        preflight.preflight(["ant"], {"gseat"}, "ant", n_questions=3, workers=2,
                            gcp_default_fn=_gcp_default_broken)
    assert "Vertex" in str(e.value)


def test_unrecognized_provider_aborts_rather_than_skipping(patched_models, monkeypatch):
    # A provider that maps to no check set must fail loudly before any paid call,
    # not be silently dropped from the run.
    monkeypatch.setattr(preflight, "JUDGE_PROVIDER", {"weird": "nowhere"})
    with pytest.raises(preflight.PreflightError) as e:
        preflight.preflight(["ant"], {"weird"}, "ant", n_questions=1, workers=1)
    assert "nowhere" in str(e.value)


# ── key providers, end to end ───────────────────────────────────────────────────

def test_all_present_key_providers_pass(patched_models, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("FIREWORKS_API_KEY", "x")
    lines = preflight.preflight(
        ["oai", "ant", "fw"], {"oai", "ant", "fw"}, "ant",
        n_questions=10, workers=4)
    assert any("key present" in ln for ln in lines)


def test_missing_key_aborts_and_names_the_var(patched_models, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("FIREWORKS_API_KEY", "x")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(preflight.PreflightError) as e:
        preflight.preflight(["ant"], set(), "ant", n_questions=5, workers=2)
    assert "ANTHROPIC_API_KEY" in str(e.value)


# ── AWS ─────────────────────────────────────────────────────────────────────────

def test_aws_unusable_aborts():
    end = datetime.now(timezone.utc) + timedelta(hours=1)
    with pytest.raises(preflight.PreflightError) as e:
        preflight.check_aws(end, boto3_module=_Boto(fail=True))
    assert "get-caller-identity" in str(e.value)


def test_aws_ok_warns_when_sso_expires_before_the_run(monkeypatch):
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(preflight, "sso_expiry", lambda *a, **k: now + timedelta(minutes=5))
    lines = preflight.check_aws(now + timedelta(hours=6), boto3_module=_Boto())
    assert any("AWS ok" in ln for ln in lines)
    assert any("WARNING" in ln and "SSO" in ln for ln in lines)


def test_aws_ok_no_warning_when_token_outlives_the_run(monkeypatch):
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(preflight, "sso_expiry", lambda *a, **k: now + timedelta(days=1))
    lines = preflight.check_aws(now + timedelta(hours=6), boto3_module=_Boto())
    assert not any("WARNING" in ln for ln in lines)


def test_preflight_runs_aws_branch_once(monkeypatch):
    monkeypatch.setattr(preflight, "provider_of", lambda k: "bedrock")
    monkeypatch.setattr(preflight, "sso_expiry", lambda *a, **k: None)
    lines = preflight.preflight(["m"], set(), "m", n_questions=5, workers=2,
                                boto3_module=_Boto())
    assert sum("AWS ok" in ln for ln in lines) == 1


# ── GCP ─────────────────────────────────────────────────────────────────────────

def test_gcp_ok_mints_a_token():
    assert preflight.check_gcp(default_fn=_gcp_default_ok) == ["Vertex ADC ok"]


def test_gcp_unusable_aborts():
    with pytest.raises(preflight.PreflightError) as e:
        preflight.check_gcp(default_fn=_gcp_default_broken)
    assert "application-default" in str(e.value)


# ── SSO cache parsing ───────────────────────────────────────────────────────────

def test_sso_expiry_reads_latest_future_token(tmp_path):
    (tmp_path / "a.json").write_text('{"accessToken": "x", "expiresAt": "2026-01-01T00:00:00Z"}')
    (tmp_path / "b.json").write_text('{"accessToken": "x", "expiresAt": "2027-06-01T12:00:00Z"}')
    (tmp_path / "c.json").write_text('{"accessToken": "x", "no_expiry": true}')
    exp = preflight.sso_expiry(tmp_path)
    assert exp == datetime(2027, 6, 1, 12, 0, tzinfo=timezone.utc)


def test_sso_expiry_ignores_client_registration_files(tmp_path):
    # A client-registration file has no accessToken and a far-future expiry; the
    # short-lived access token is the one the run depends on.
    (tmp_path / "reg.json").write_text(
        '{"clientId": "c", "clientSecret": "s", "expiresAt": "2099-01-01T00:00:00Z"}')
    (tmp_path / "token.json").write_text(
        '{"accessToken": "x", "expiresAt": "2026-01-01T00:00:00Z"}')
    exp = preflight.sso_expiry(tmp_path)
    assert exp == datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def test_sso_expiry_none_without_cache(tmp_path):
    assert preflight.sso_expiry(tmp_path / "missing") is None
