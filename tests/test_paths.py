"""Guards for the data-root resolution.

The point of paths.py is a property, not a feature: no module reaches the
monorepo root on its own to find data. That property is invisible at review time
and easy to undo, because the old idiom
(`Path(__file__).resolve().parents[2] / "data/benchmarks/..."`) is one line and
looks unremarkable in a diff. test_no_module_derives_its_own_data_root is
therefore the load-bearing test here; the rest check that the resolver behaves.
"""
from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

DAM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DAM))

import paths  # noqa: E402

# No exceptions. The two tools that genuinely need the repository itself (the DVC
# object cache, and recording the source commit in a bundle manifest) import
# paths.REPO_ROOT, so the monorepo assumption lives in one labelled place and this
# guard has nothing to allow.
ALLOWED: set[str] = set()


def _modules():
    for p in sorted(DAM.rglob("*.py")):
        rel = p.relative_to(DAM).as_posix()
        if any(x in rel for x in ("__pycache__", ".venv", "tests/")) or rel == "paths.py":
            continue
        yield rel, p


def test_no_module_derives_its_own_data_root():
    """The regression guard. Reintroducing the old idiom silently re-breaks the
    harness split for whichever script does it, and nothing else would notice."""
    offenders = []
    for rel, p in _modules():
        if rel in ALLOWED:
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if re.search(r"parents\[\d+\]", line):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, (
        "these modules reach the monorepo root themselves instead of importing "
        "paths.py; add to ALLOWED only if the path is genuinely not benchmark "
        "data:\n  " + "\n  ".join(offenders))


def test_no_module_hardcodes_the_data_directory():
    """`data/benchmarks` should appear in paths.py and nowhere else, IN CODE.

    Checked over the AST rather than the raw text: comments are absent from it
    entirely and docstrings are excluded explicitly, because prose describing the
    default layout ("Writes data/benchmarks/.../results.jsonl") is documentation,
    not a hardcoded path. A text scan flags six such docstrings and would train
    people to ignore this test.
    """
    import ast
    offenders = []
    for rel, p in _modules():
        if rel in ALLOWED:
            continue
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError:
            continue
        docstrings = {id(ast.get_docstring(n, clean=False))
                      for n in ast.walk(tree)
                      if isinstance(n, (ast.Module, ast.FunctionDef,
                                        ast.AsyncFunctionDef, ast.ClassDef))}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and "data/benchmarks" in node.value
                    and id(node.value) not in docstrings):
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, ("hardcoded data directory in code outside paths.py: "
                           + ", ".join(offenders))


def _reimport(env: dict[str, str]) -> subprocess.CompletedProcess:
    """paths resolves at import, so overrides have to be tested in a fresh process."""
    return subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r); import paths; print(paths.DATA)" % str(DAM)],
        capture_output=True, text=True, env={**os.environ, **env})


def test_env_override_wins(tmp_path):
    r = _reimport({"DAM_DATA_ROOT": str(tmp_path)})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(tmp_path.resolve())


def test_explicit_override_may_point_at_a_missing_directory(tmp_path):
    """Seeding legitimately targets a root that does not exist yet, so an
    explicit override must not be second-guessed."""
    missing = tmp_path / "not-created-yet"
    r = _reimport({"DAM_DATA_ROOT": str(missing)})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(missing.resolve())


def test_missing_default_root_fails_with_a_useful_message(tmp_path, monkeypatch):
    """Without the guard this surfaces as a missing-file error several frames deep
    in whichever artifact a script happened to read first."""
    fake = tmp_path / "experiments" / "data-agent-mnist"
    fake.mkdir(parents=True)
    (fake / "paths.py").write_text((DAM / "paths.py").read_text())
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r); import paths" % str(fake)],
        capture_output=True, text=True,
        env={k: v for k, v in os.environ.items() if k != "DAM_DATA_ROOT"})
    assert r.returncode != 0
    assert "benchmark data root not found" in r.stderr
    assert "DAM_DATA_ROOT" in r.stderr and "dvc pull" in r.stderr


def test_subdirectory_scripts_import_when_run_as_scripts(tmp_path):
    """Run each subdirectory entrypoint the documented way, as a script.

    This exists because two earlier checks passed on a bug that this one catches.
    Importing a module, or runpy-ing it from the experiment root, leaves that root
    on sys.path and `import paths` resolves. Running `uv run sensitivity/foo.py`
    puts the SCRIPT'S directory on sys.path instead, so a `from paths import ...`
    placed above the module's own sys.path.insert crashes on import. The masking
    is the whole point: only a real subprocess invocation reproduces it.
    """
    scripts = sorted(p for p in DAM.rglob("*.py")
                     if p.parent != DAM
                     and not any(x in p.as_posix() for x in ("__pycache__", ".venv", "/tests/"))
                     and "from paths import" in p.read_text())
    # Having any is a property of the tree, not of the code. Every subdirectory
    # that holds one (sensitivity, maintenance, release) is board-only and stays
    # out of the harness repo, where this asserted itself into a failure on a
    # tree that is simply correct. Same shape as the retired-models
    # test in test_registry.
    if not scripts:
        pytest.skip("no subdirectory scripts import paths in this tree")
    broken = []
    for s in scripts:
        r = subprocess.run([sys.executable, str(s), "--help"], capture_output=True,
                           text=True, cwd=DAM,
                           env={**os.environ, "DAM_DATA_ROOT": str(tmp_path)})
        if "No module named 'paths'" in r.stderr:
            broken.append(s.relative_to(DAM).as_posix())
    assert not broken, (
        "these scripts import paths before putting the experiment root on "
        "sys.path, so they crash when run directly:\n  " + "\n  ".join(broken))
