#!/usr/bin/env python
"""Statistics for "does the modality order matter?" (paper Section 4.6).

Reads the score lines ("<tag>_s<seed> ... CCC = x") of logs/eval_*.log files, or of any text files with those lines.

2a  Training-time order (models trained with different fixed orders, tags ord_<order>_<corpus>):
    per-order mean/sd, one-way ANOVA, variance components (seed sd, order sd with an approximate 95% upper bound),
    and for every order a Welch TOST against the runs of all other orders with margin --margin.
2b  Inference-time order (one trained model scanned in other orders, tags probe_<order>_<corpus>_<probe>):
    per model, delta = CCC(probe) - CCC(own order). The probes of one model are averaged first, so the unit of
    analysis is the trained model (no pseudo-replication); then a one-sample TOST on those per-model deltas.

TOST: equivalence within +-margin is shown when both one-sided tests reject at alpha = 0.05 (p_TOST < 0.05),
which is the same as the 90% confidence interval lying inside (-margin, +margin). "Excluded beyond" is the
smallest margin that the data establish, i.e. the largest absolute end of that 90% interval.

usage (repo root): python scripts/order_stats.py logs/eval_ord_*.log logs/eval_probe_*.log [--margin 0.02]
"""

from __future__ import annotations

import argparse
import collections
import re

import numpy as np

from scipy import stats

ORDERS = ("blocked", "interleaved", "rand0", "rand1")
LINE = re.compile(r"^(\S+)_s(\d+) .*CCC = (-?[\d.]+)")


def read_scores(paths: list[str]) -> dict[str, dict[int, float]]:
    scores: dict[str, dict[int, float]] = collections.defaultdict(dict)
    for p in paths:
        for line in open(p):
            m = LINE.search(line)
            if m:
                scores[m.group(1)][int(m.group(2))] = float(m.group(3))
    return scores


def welch_tost(x: np.ndarray, y: np.ndarray, margin: float) -> tuple[float, float, float, float]:
    """-> (diff, p_tost, ci90_low, ci90_high) for mean(x) - mean(y)."""
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = np.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    d = x.mean() - y.mean()
    p = max(stats.t.sf((d + margin) / se, df), stats.t.cdf((d - margin) / se, df))
    h = stats.t.ppf(0.95, df) * se
    return d, p, d - h, d + h


def one_sample_tost(d: np.ndarray, margin: float) -> tuple[float, float, float, float]:
    se, df, m = d.std(ddof=1) / np.sqrt(len(d)), len(d) - 1, d.mean()
    p = max(stats.t.sf((m + margin) / se, df), stats.t.cdf((m - margin) / se, df))
    h = stats.t.ppf(0.95, df) * se
    return m, p, m - h, m + h


def training_order(scores: dict, corpus: str, margin: float) -> None:
    groups = {o: np.array(list(scores[f"ord_{o}_{corpus}"].values())) for o in ORDERS if f"ord_{o}_{corpus}" in scores}
    if len(groups) < 2:
        return
    print(f"\n== 2a  training-time order, {corpus} ==")
    for o, g in groups.items():
        print(f"  {o:12s} n={len(g)}  {g.mean():.4f} +- {g.std(ddof=1):.4f}")
    g = list(groups.values())
    F, p = stats.f_oneway(*g)
    a, n = len(g), np.mean([len(x) for x in g])  # n: mean group size (balanced design expected)
    msb = n * np.var([x.mean() for x in g], ddof=1)
    msw = np.mean([x.var(ddof=1) for x in g])
    s2a = max(0.0, (msb - msw) / n)
    s2a_ub = max(0.0, (msb * (a - 1) / stats.chi2.ppf(0.05, a - 1) - msw) / n)  # approximate: ignores the uncertainty of msw
    print(f"  ANOVA F={F:.2f} p={p:.3f}   seed sd {np.sqrt(msw):.4f}   order sd {np.sqrt(s2a):.4f} (approx. 95% upper bound {np.sqrt(s2a_ub):.4f})")
    print(f"  spread of order means (max - min): {max(x.mean() for x in g) - min(x.mean() for x in g):.4f}")
    for o, x in groups.items():
        rest = np.concatenate([y for k, y in groups.items() if k != o])
        d, pt, lo, hi = welch_tost(x, rest, margin)
        print(f"  {o:12s} vs other orders: diff {d:+.4f}  90% CI [{lo:+.4f}, {hi:+.4f}]  TOST(+-{margin}) p={pt:.3f}  excluded beyond {max(abs(lo), abs(hi)):.4f}")


def inference_order(scores: dict, corpus: str, margin: float) -> None:
    rows = []  # (order, seed, probe, delta)
    for tag, by_seed in scores.items():
        m = re.fullmatch(rf"probe_(\w+?)_{corpus}_(\w+)", tag)
        if not m:
            continue
        own = scores.get(f"ord_{m.group(1)}_{corpus}", {})
        for s, v in by_seed.items():
            if s in own:
                rows.append((m.group(1), s, m.group(2), v - own[s]))
    if not rows:
        return
    print(f"\n== 2b  inference-time order, {corpus}: delta = CCC(probe order) - CCC(trained order) ==")
    per_model = collections.defaultdict(list)
    for o, s, _, d in rows:
        per_model[(o, s)].append(d)
    for o in ORDERS:
        ds = [d for (oo, _, _, d) in rows if oo == o]
        if ds:
            print(f"  trained {o:12s} {len(ds):3d} probes: mean {np.mean(ds):+.4f}  sd {np.std(ds, ddof=1):.4f}  min {min(ds):+.4f}  max {max(ds):+.4f}")
    all_d = np.array([r[3] for r in rows])
    print(f"  all {len(all_d)} probes: mean {all_d.mean():+.4f}  mean |delta| {np.abs(all_d).mean():.4f}  max |delta| {np.abs(all_d).max():.4f}")
    model_mean = np.array([np.mean(v) for v in per_model.values()])
    model_worst = np.array([min(v) for v in per_model.values()])
    m, pt, lo, hi = one_sample_tost(model_mean, margin)
    print(f"  per-model mean delta ({len(model_mean)} models): {m:+.4f}  90% CI [{lo:+.4f}, {hi:+.4f}]  TOST(+-{margin}) p={pt:.4f}  excluded beyond {max(abs(lo), abs(hi)):.4f}")
    print(f"  worst probe per model: mean {model_worst.mean():+.4f}  min {model_worst.min():+.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--margin", type=float, default=0.02)
    args = ap.parse_args()
    scores = read_scores(args.files)
    for corpus in ("noxi", "noxij"):
        training_order(scores, corpus, args.margin)
        inference_order(scores, corpus, args.margin)


if __name__ == "__main__":
    main()
