"""Model-based judge audit over the cached panel votes (offline).

The panel scores each candidate answer against the reference answer, one
pass/tie/fail vote per seat, so the contest is candidate-vs-reference, not
candidate-vs-candidate. A pairwise Bradley-Terry (Bradley and Terry 1952) does
not apply directly; its rater-effects form does. This fits the many-facet Rasch
model (Rasch 1960; the many-facet rater extension, Linacre 1989):

    logit P(judge j passes candidate c) = theta_c - beta_j + gamma * family(c, j)

theta_c is candidate ability, beta_j is judge severity, and family(c, j) is 1
when the candidate and the judge share a provider, so gamma is the own-family
partisanship test. The fit replaces three separate descriptive audits (leniency
table, in-group residual, single-judge ablation) with one model that carries
uncertainty: candidate abilities and gamma get bootstrap intervals.

Everything comes from result_score.votes in the cached board results; no model
calls. Ties score 0.5 and enter as a fractional-response (quasi-Bernoulli)
target; --drop-ties reports the tie-free sensitivity variant.

    uv run 13_judge_bt.py [--bootstrap 500] [--emit judge_bt.json] [--plot out.pdf]
    uv run 13_judge_bt.py --selftest
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))

# bench and paths are imported lazily inside the functions that touch board data
# or provider config, so the numeric core and --selftest run with no credentials
# and no data root. bench builds provider clients at import; paths asserts the
# data root exists. Neither should be a precondition for the self-contained fit.

VOTE = {"pass": 1.0, "tie": 0.5, "fail": 0.0, "equivalent": 1.0, "not_equivalent": 0.0}

# Candidate vendor by name prefix, in the same namespace as the judge seat keys
# (anthropic/openai/google), so family(c, j) is a plain equality. Only the three
# judge families can match; every other vendor maps to a label no seat carries.
CAND_VENDOR_PREFIX = [
    ("opus", "anthropic"), ("sonnet", "anthropic"), ("haiku", "anthropic"),
    ("fable", "anthropic"),
    ("gpt", "openai"), ("o4", "openai"),
    ("gemini", "google"), ("gemma", "google"),
    ("nova", "amazon"), ("deepseek", "deepseek"), ("kimi", "moonshot"),
    ("qwen", "qwen"), ("glm", "zhipu"),
]


def cand_vendor(model: str) -> str:
    return next((v for pre, v in CAND_VENDOR_PREFIX if model.startswith(pre)), "other")


class Votes:
    """Long-form votes ready for the fit: parallel arrays plus the index maps."""

    def __init__(self, cand_idx, judge_idx, ci, ji, fam, y, trace):
        self.cand_idx = cand_idx          # {model_key: column}
        self.judge_idx = judge_idx        # {judge_key: column}
        self.ci = np.asarray(ci, dtype=np.intp)
        self.ji = np.asarray(ji, dtype=np.intp)
        self.fam = np.asarray(fam, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.trace = np.asarray(trace)

    @property
    def n_cand(self) -> int:
        return len(self.cand_idx)

    @property
    def n_judge(self) -> int:
        return len(self.judge_idx)


def load_votes(results_path: Path, drop_ties: bool = False) -> Votes:
    """Extract one row per (candidate, judge) vote. Mirrors judge_ablation.py:
    skip retired candidates and unjudged/turn-limited runs (which carry no votes),
    and read `outcome` or the older `verdict`."""
    from bench import JUDGE_PROVIDER, RETIRED_CANDIDATES
    cands: dict[str, int] = {}
    judges: dict[str, int] = {}
    ci, ji, fam, y, trace = [], [], [], [], []
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tid = r.get("trace_id")
        for m, c in (r.get("candidates") or {}).items():
            if m in RETIRED_CANDIDATES:
                continue
            for v in ((c.get("result_score") or {}).get("votes") or []):
                jkey = v.get("judge")
                prov = JUDGE_PROVIDER.get(jkey)
                o = v.get("outcome") or v.get("verdict")
                if not prov or o not in VOTE:
                    continue
                score = VOTE[o]
                if drop_ties and score == 0.5:
                    continue
                ci.append(cands.setdefault(m, len(cands)))
                ji.append(judges.setdefault(jkey, len(judges)))
                fam.append(1.0 if cand_vendor(m) == prov else 0.0)
                y.append(score)
                trace.append(tid)
    return Votes(cands, judges, ci, ji, fam, y, trace)


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))


def fit(v: Votes, ci=None, ji=None, fam=None, y=None, ridge: float = 1e-6):
    """Penalised MLE of (theta, beta, gamma). theta_c - beta_j has one global-shift
    degeneracy (add k to every theta and every beta and nothing changes); a tiny
    ridge on theta and beta pins it, and both groups are re-centred on their own
    mean for reporting, which is the identified quantity. gamma is identified and
    left unpenalised. The optional arrays let the bootstrap refit a resample over
    the same fixed candidate/judge columns."""
    ci = v.ci if ci is None else ci
    ji = v.ji if ji is None else ji
    fam = v.fam if fam is None else fam
    y = v.y if y is None else y
    nc, nj = v.n_cand, v.n_judge

    def unpack(x):
        return x[:nc], x[nc:nc + nj], x[nc + nj]

    def objective(x):
        theta, beta, gamma = unpack(x)
        eta = theta[ci] - beta[ji] + gamma * fam
        p = _sigmoid(eta)
        eps = 1e-12
        nll = -np.sum(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
        nll += ridge * (np.dot(theta, theta) + np.dot(beta, beta))
        resid = p - y
        g_theta = np.zeros(nc)
        np.add.at(g_theta, ci, resid)
        g_theta += 2.0 * ridge * theta
        g_beta = np.zeros(nj)
        np.add.at(g_beta, ji, -resid)
        g_beta += 2.0 * ridge * beta
        g_gamma = float(np.sum(fam * resid))
        return nll, np.concatenate([g_theta, g_beta, [g_gamma]])

    x0 = np.zeros(nc + nj + 1)
    res = minimize(objective, x0, jac=True, method="L-BFGS-B")
    theta, beta, gamma = unpack(res.x)
    return theta - theta.mean(), beta - beta.mean(), float(gamma)


def bootstrap(v: Votes, n: int, seed: int):
    """Nonparametric bootstrap clustered on trace_id: resample questions, refit.
    Candidates or judges absent from a resample are ridge-shrunk toward the mean,
    which widens rather than biases their interval."""
    rng = np.random.default_rng(seed)
    by_trace: dict[str, list[int]] = {}
    for i, t in enumerate(v.trace):
        by_trace.setdefault(t, []).append(i)
    traces = list(by_trace)
    thetas = np.empty((n, v.n_cand))
    betas = np.empty((n, v.n_judge))
    gammas = np.empty(n)
    for b in range(n):
        pick = rng.choice(len(traces), size=len(traces), replace=True)
        idx = np.concatenate([by_trace[traces[k]] for k in pick])
        th, be, ga = fit(v, v.ci[idx], v.ji[idx], v.fam[idx], v.y[idx])
        thetas[b], betas[b], gammas[b] = th, be, ga
    return thetas, betas, gammas


def _ci(samples: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def analyse(v: Votes, n_boot: int, seed: int) -> dict:
    from bench import JUDGE_PROVIDER
    theta, beta, gamma = fit(v)
    thetas, betas, gammas = (bootstrap(v, n_boot, seed) if n_boot
                             else (None, None, None))
    names = importlib.import_module("09_dds_analysis").NAMES
    cand_rows = []
    for m, j in sorted(v.cand_idx.items(), key=lambda kv: -theta[kv[1]]):
        lo, hi = _ci(thetas[:, j]) if n_boot else (None, None)
        cand_rows.append({"key": m, "name": names.get(m, m),
                          "ability": round(float(theta[j]), 3),
                          "ci95": [round(lo, 3), round(hi, 3)] if n_boot else None})
    judge_rows = []
    for jk, j in sorted(v.judge_idx.items(), key=lambda kv: kv[1]):
        lo, hi = _ci(betas[:, j]) if n_boot else (None, None)
        judge_rows.append({"judge": jk, "provider": JUDGE_PROVIDER.get(jk, "?"),
                           "severity": round(float(beta[j]), 3),
                           "ci95": [round(lo, 3), round(hi, 3)] if n_boot else None})
    judge_rows.sort(key=lambda r: r["severity"])
    g_ci = _ci(gammas) if n_boot else None
    return {
        "n_votes": int(v.y.size), "n_candidates": v.n_cand, "n_judges": v.n_judge,
        "n_bootstrap": n_boot,
        "family_coefficient": {"gamma": round(gamma, 3),
                               "ci95": [round(g_ci[0], 3), round(g_ci[1], 3)] if n_boot else None},
        "candidates": cand_rows, "judges": judge_rows,
    }


def _print(report: dict) -> None:
    print(f"{report['n_votes']} votes, {report['n_candidates']} candidates, "
          f"{report['n_judges']} judges, {report['n_bootstrap']} bootstrap resamples\n")
    print("candidate ability (higher is stronger, relative to the mean candidate):")
    for r in report["candidates"]:
        ci = f"  [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]" if r["ci95"] else ""
        print(f"  {r['name']:24} {r['ability']:+.3f}{ci}")
    print("\njudge severity (higher is harsher, relative to the mean judge):")
    for r in report["judges"]:
        ci = f"  [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]" if r["ci95"] else ""
        print(f"  {r['judge']:18} {r['provider']:10} {r['severity']:+.3f}{ci}")
    g = report["family_coefficient"]
    ci = f"  [{g['ci95'][0]:+.2f}, {g['ci95'][1]:+.2f}]" if g["ci95"] else ""
    print(f"\nown-family coefficient gamma: {g['gamma']:+.3f}{ci}  "
          f"(positive means a seat is more favourable to its own provider)")


def plot(report: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = report["candidates"][::-1]
    y = np.arange(len(rows))
    ab = [r["ability"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 0.32 * len(rows) + 1))
    if rows and rows[0]["ci95"]:
        lo = [r["ability"] - r["ci95"][0] for r in rows]
        hi = [r["ci95"][1] - r["ability"] for r in rows]
        ax.errorbar(ab, y, xerr=[lo, hi], fmt="o", color="#b25a00", ecolor="#999", capsize=2)
    else:
        ax.plot(ab, y, "o", color="#b25a00")
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in rows], fontsize=8)
    ax.set_xlabel("ability (log-odds, relative to mean candidate)")
    ax.set_title("Judge-model candidate ability with 95% bootstrap intervals",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def _selftest() -> bool:
    """Fit synthetic votes with planted ability, severity and family gaps and
    check recovery. Runs without any board data, so it is the CI-safe check."""
    rng = np.random.default_rng(0)
    n_cand, providers = 8, ["anthropic", "openai", "google"]
    true_theta = np.linspace(-1.5, 1.5, n_cand)
    vendor = [providers[i % 3] for i in range(n_cand)]
    judges = ["anthropic_j", "openai_j", "google_j"]
    true_beta = {"anthropic_j": -0.4, "openai_j": -0.9, "google_j": 0.6}  # google harshest
    jprov = {"anthropic_j": "anthropic", "openai_j": "openai", "google_j": "google"}
    gamma_true = 0.6
    ci, ji, fam, y, trace = [], [], [], [], []
    cidx = {f"cand{i}": i for i in range(n_cand)}
    jidx = {j: k for k, j in enumerate(judges)}
    for q in range(120):
        for i in range(n_cand):
            for jk in judges:
                f = 1.0 if vendor[i] == jprov[jk] else 0.0
                eta = true_theta[i] - true_beta[jk] + gamma_true * f
                p = 1.0 / (1.0 + np.exp(-eta))
                ci.append(i); ji.append(jidx[jk]); fam.append(f)
                y.append(float(rng.random() < p)); trace.append(f"q{q}")
    v = Votes(cidx, jidx, ci, ji, fam, y, trace)
    theta, beta, gamma = fit(v)
    corr = float(np.corrcoef(theta, true_theta - true_theta.mean())[0, 1])
    centred_beta = {j: beta[jidx[j]] for j in judges}
    order_ok = (centred_beta["openai_j"] < centred_beta["anthropic_j"] < centred_beta["google_j"])
    gamma_ok = abs(gamma - gamma_true) < 0.25
    ok = corr > 0.95 and order_ok and gamma_ok
    print(f"selftest: ability corr={corr:.3f} (>0.95), severity order ok={order_ok}, "
          f"gamma={gamma:.3f} vs {gamma_true} (ok={gamma_ok}) -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=None,
                    help="results.jsonl (default: the synthetic board results under the data root)")
    ap.add_argument("--bootstrap", type=int, default=500,
                    help="clustered-bootstrap resamples for intervals; 0 to skip")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-ties", action="store_true",
                    help="tie-free sensitivity variant (ties normally score 0.5)")
    ap.add_argument("--emit", type=Path, default=None, help="write the report as JSON")
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="fit synthetic votes and check recovery; no board data needed")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if _selftest() else 1)

    from paths import DATA
    results = args.results or (DATA / "text2sqlbench-synthetic/results.jsonl")
    v = load_votes(results, drop_ties=args.drop_ties)
    if v.y.size == 0:
        raise SystemExit("no usable votes in the results file")
    report = analyse(v, args.bootstrap, args.seed)
    _print(report)
    if args.emit:
        args.emit.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.emit}")
    if args.plot:
        plot(report, args.plot)


if __name__ == "__main__":
    main()
