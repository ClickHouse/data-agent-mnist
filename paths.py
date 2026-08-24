"""Where the benchmark data lives, resolved once instead of in thirty modules.

Every module used to rebuild the location from its own depth in the tree
(`Path(__file__).resolve().parents[2] / "data/benchmarks/..."`, or `parents[1]` and
`parents[3]` from the subdirectories), which hard-codes two separate assumptions:
that a monorepo root exists above this directory, and that the data sits at
`data/benchmarks` beneath it. Neither holds outside this repository, and thirty
copies is thirty places to change when it stops holding (AI-1858).

`DAM_DATA_ROOT` overrides the location. The default is the monorepo layout, so
nothing here needs configuring; an adopter points the variable at their own data
and changes nothing else. That is the whole mechanism: the rest of this change is
the thirty call sites, which had to stop computing the location themselves for
the variable to reach them.

A wrong root is worth failing on loudly. If the resolved root does not exist, the
scripts would otherwise report a confusing missing-file error from whichever
artifact they happened to read first, several frames deep. The check below names
the actual problem instead. It is skipped when the caller has set the variable
explicitly to somewhere not yet created, since seeding legitimately does that.
"""
from __future__ import annotations

import os
from pathlib import Path

_ENV = "DAM_DATA_ROOT"
_HERE = Path(__file__).resolve().parent

# Monorepo default: <repo>/data/benchmarks, two levels above this directory.
_DEFAULT = _HERE.parents[1] / "data" / "benchmarks"

_explicit = os.environ.get(_ENV)
DATA: Path = Path(_explicit).expanduser().resolve() if _explicit else _DEFAULT

if not _explicit and not DATA.exists():
    raise FileNotFoundError(
        f"benchmark data root not found at {DATA}.\n"
        f"  In the monorepo this means the DVC data is not checked out: run "
        f"`dvc pull data/benchmarks/...`.\n"
        f"  Outside it, set {_ENV} to wherever your datasets live.")


# Datasets are addressed as DATA / "<name>" at the point of use. There was a set
# of named constants here (SYNTH, CEILING, ...); nothing imported them, because
# the call sites read perfectly well without the indirection.

# The monorepo root. NOT part of the harness contract and deliberately not
# configurable: an adopter has no repository above their data, and nothing in the
# eval path may depend on this. It exists for the two monorepo-only tools that
# need the repository itself rather than the benchmark data (the DVC object cache,
# and recording the source commit in a bundle manifest), both of which stay
# private under the harness RFC.
#
# Exported anyway so the assumption lives in exactly one labelled place instead of
# being re-derived with parents[3] in the tools that use it.
REPO_ROOT: Path = _HERE.parents[1]
