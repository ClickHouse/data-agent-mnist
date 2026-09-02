"""Where the benchmark data lives, resolved once instead of in thirty modules.

Every module used to rebuild the location from its own depth in the tree
(`Path(__file__).resolve().parents[2] / "data/benchmarks/..."`, or `parents[1]` and
`parents[3]` from the subdirectories), which hard-codes two separate assumptions:
that a monorepo root exists above this directory, and that the data sits at
`data/benchmarks` beneath it. Neither holds outside this repository, and thirty
copies is thirty places to change when it stops holding.

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

# DAM_DATA_ROOT first, then fall back to inferring the monorepo layout. Order
# matters: the fallback can fail, and it must not be able to fail for someone who
# set the variable and therefore needs no inference at all.
_explicit = os.environ.get(_ENV)

if _explicit:
    DATA: Path = Path(_explicit).expanduser().resolve()
else:
    # Monorepo default: <repo>/data/benchmarks, two levels above this directory.
    #
    # Guarded, because parents[1] raises a bare `IndexError: 1` when this
    # directory is fewer than two levels from the filesystem root. That is not
    # hypothetical for the standalone harness: cloning it to a shallow path made
    # every module importing this one fail to collect, with a traceback pointing
    # at pathlib and no hint about the cause.
    if len(_HERE.parents) < 2:
        raise RuntimeError(
            f"cannot infer a benchmark data root: {_HERE} is too close to the "
            f"filesystem root to have a monorepo above it.\n"
            f"  Set {_ENV} to the directory holding the benchmark datasets, or "
            f"clone to a deeper path.")
    DATA = _HERE.parents[1] / "data" / "benchmarks"

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
#
# Resolved LAZILY, via module __getattr__, for two reasons. It cannot be computed
# at all when this directory is fewer than two levels from the filesystem root,
# and computing it eagerly made a shallow standalone clone fail on import even
# when DAM_DATA_ROOT was set and no repository root was wanted. And since nothing
# on the eval path may depend on it, nobody who does not ask for it should pay for
# it. `from paths import REPO_ROOT` still works; it just fails at the point of use
# rather than at import, with a message instead of an IndexError.
def __getattr__(name: str):
    if name == "REPO_ROOT":
        if len(_HERE.parents) < 2:
            raise RuntimeError(
                f"REPO_ROOT is not available: {_HERE} has no repository root "
                f"above it. It is monorepo-only and nothing on the eval path "
                f"may depend on it.")
        return _HERE.parents[1]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
