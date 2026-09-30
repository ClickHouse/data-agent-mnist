"""Guards for the many-facet judge model.

The fit is the audit: if it cannot recover a known ability, severity and
own-family gap from data it generated itself, its numbers on the board mean
nothing. These tests plant those gaps and check recovery, with no board data and
no model calls, so they run in CI under placeholder credentials.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("DAM_DATA_ROOT", "/tmp/dam-judge-bt-test")
os.environ.setdefault("DAM_MODELS_CONFIG",
                      str(Path(__file__).resolve().parents[1] / "config" / "models.example.yaml"))
os.environ.setdefault("OPENAI_API_KEY", "placeholder")
os.environ.setdefault("ANTHROPIC_API_KEY", "placeholder")
os.environ.setdefault("FIREWORKS_API_KEY", "placeholder")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("bench")
pytest.importorskip("scipy")
jbt = importlib.import_module("13_judge_bt")


def test_selftest_recovers_planted_gaps():
    assert jbt._selftest() is True


def test_family_indicator_uses_the_vendor_namespace():
    # opus/gpt/gemini map to the seat providers; other vendors never match.
    assert jbt.cand_vendor("opus48") == "anthropic"
    assert jbt.cand_vendor("fable51") == "anthropic"   # Anthropic model, own-family for the Anthropic seat
    assert jbt.cand_vendor("gpt-5.5") == "openai"
    assert jbt.cand_vendor("gemini-3.7-flash") == "google"
    assert jbt.cand_vendor("kimi-k3") not in {"anthropic", "openai", "google"}


def test_fit_centres_each_group():
    import numpy as np
    rng = np.random.default_rng(1)
    judges = ["ja", "jb", "jc"]
    jidx = {j: k for k, j in enumerate(judges)}
    cidx = {f"c{i}": i for i in range(5)}
    ci, ji, fam, y, trace = [], [], [], [], []
    for q in range(60):
        for i in range(5):
            for jk in judges:
                ci.append(i); ji.append(jidx[jk]); fam.append(0.0)
                y.append(float(rng.random() < 0.5)); trace.append(f"q{q}")
    v = jbt.Votes(cidx, jidx, ci, ji, fam, y, trace)
    theta, beta, gamma = jbt.fit(v)
    assert abs(float(theta.mean())) < 1e-6
    assert abs(float(beta.mean())) < 1e-6
