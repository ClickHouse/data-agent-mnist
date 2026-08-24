"""
Optional observability for the bench scripts: Langfuse tracing + MLflow metrics.

Both degrade to no-ops when their env keys / tracking URI are absent, so the
annotate and eval scripts run anywhere (CI, a laptop without secrets) without
guard clauses scattered through them. Extracted from nb05 cell 02.
"""
import os
from contextlib import contextmanager


class _NoopLangfuse:
    """Stand-in when Langfuse isn't configured; matches the v4 span API surface."""
    @contextmanager
    def start_as_current_observation(self, **kw):
        yield self

    def update(self, **kw):
        pass

    def flush(self):
        pass


def get_langfuse(prefer_research: bool = True):
    """Langfuse v4 client, or a no-op stand-in. Defaults to the *research* project
    keys (synthetic-bench traces), falling back to the default project keys."""
    pk = sk = None
    if prefer_research:
        pk = os.environ.get("LANGFUSE_RESEARCH_PUBLIC_KEY")
        sk = os.environ.get("LANGFUSE_RESEARCH_SECRET_KEY")
    pk = pk or os.environ.get("LANGFUSE_PUBLIC_KEY")
    sk = sk or os.environ.get("LANGFUSE_SECRET_KEY")
    if not (pk and sk):
        print("Langfuse disabled (no keys in env)")
        return _NoopLangfuse()
    try:
        import langfuse as _pkg
        from langfuse import Langfuse
        lf = Langfuse(public_key=pk, secret_key=sk,
                      host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"))
        print(f"Langfuse ready (SDK {_pkg.__version__})")
        return lf
    except Exception as e:
        print(f"WARNING: Langfuse init failed ({e}); tracing disabled")
        return _NoopLangfuse()


def setup_mlflow(experiment: str = "text2sqlbench-synthetic"):
    """The mlflow module wired to the configured experiment, or None if no tracking
    URI is set / the server is unreachable (metrics then silently skipped)."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri:
        print("MLflow disabled (MLFLOW_TRACKING_URI unset)")
        return None
    try:
        import mlflow
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        print(f"MLflow ready ({uri})")
        return mlflow
    except Exception as e:
        print(f"WARNING: MLflow unreachable ({e}); metrics disabled")
        return None
