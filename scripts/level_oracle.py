#!/usr/bin/env python
"""How much of the pooled CCC is about each person's overall engagement level, and how much about the changes within a session?

Reads validation predictions as written by inference.py (<pred_dir>/<corpus>/val/<session>/<role>.engagement.pred.csv) and the
ground truth, aligned exactly as src/evaluate.py does (truncated, non-finite dropped). A "person" is one role in one session.

Per corpus it prints:
  - label variance split into between-person (variance of the person means) and within-person parts
  - the actual pooled CCC of each prediction dir and of their frame-wise mean (ensemble, not rescaled)
  - three oracles that swap one part of the prediction for the truth:
      true level   pred - mean_p(pred) + mean_p(gt)   our within-person dynamics, perfect level
      level only   mean_p(gt) as a constant            perfect level, no dynamics at all
      true dynamics gt - mean_p(gt) + mean_p(pred)    perfect dynamics, our level
  - how well the level is predicted: Pearson r and mean absolute error of the person means; mean within-person Pearson r
With --lag-check, also the pooled CCC and mean within-person r (averaged over the single models, and for their mean) when the
predictions are shifted in time (positive lag = prediction delayed, as if the annotation lags behind the behaviour) or smoothed
with a centred moving average. Shifts and smoothing stay within one person and pad with the edge value, so lengths are unchanged.
With --bands, each person's deviation from their mean is split by timescale with centred moving averages (MA) of 2, 10 and 60 s:
fast = x - MA2 (< 2 s), mid = MA2 - MA10 (2-10 s), slow = MA10 - MA60 (10-60 s, longer than one window), very slow = MA60 - mean
(> 60 s); the four parts add up to x - mean exactly. Per band: share of the within-person label variance, mean within-person r
between the prediction's and the label's band, and the pooled CCC if that band of the prediction were replaced by the truth.

usage (repo root): python scripts/level_oracle.py --corpus NoXi --pred-dirs sweep_out/feat_noxi_t_besttau_s* [--data-root data] [--lag-check] [--bands]
"""

from __future__ import annotations

import argparse
import numpy as np
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluate import find_pred, load_csv, official_ccc
from read_data import ROLES


def load_people(gt_dir: Path, pred_dirs: list[Path]) -> list[tuple[str, np.ndarray, list[np.ndarray]]]:
    """-> [(person id, gt, [pred per dir])], aligned to a common length and finite in all dirs."""
    people = []
    for session_dir in sorted(p for p in gt_dir.iterdir() if p.is_dir()):
        for role in ROLES:
            gt_path = session_dir / f"{role}.engagement.annotation.csv"
            pred_paths = [find_pred(d / session_dir.name, role) for d in pred_dirs]
            if not gt_path.exists() or not all(p.exists() for p in pred_paths):
                continue
            gt = load_csv(gt_path)
            preds = [load_csv(p) for p in pred_paths]
            n = min([len(gt)] + [len(p) for p in preds])
            gt, preds = gt[:n], [p[:n] for p in preds]
            mask = np.isfinite(gt) & np.all([np.isfinite(p) for p in preds], axis=0)
            if mask.sum() > 1:
                people.append((f"{session_dir.name}.{role}", gt[mask], [p[mask] for p in preds]))
    return people


def report(name: str, gt: list[np.ndarray], pred: list[np.ndarray]) -> None:
    g, p = np.concatenate(gt), np.concatenate(pred)
    gm = np.array([x.mean() for x in gt])
    pm = np.array([x.mean() for x in pred])
    true_level = np.concatenate([x - x.mean() + y.mean() for x, y in zip(pred, gt)])
    level_only = np.concatenate([np.full(len(y), y.mean()) for y in gt])
    true_dyn = np.concatenate([y - y.mean() + x.mean() for x, y in zip(pred, gt)])
    within_r = [np.corrcoef(x, y)[0, 1] for x, y in zip(pred, gt) if x.std() > 0 and y.std() > 0]
    print(f"  {name:28s} CCC {official_ccc(p, g):.4f} | oracles: true level {official_ccc(true_level, g):.4f}  level only {official_ccc(level_only, g):.4f}  true dynamics {official_ccc(true_dyn, g):.4f}"
          f" | level r {np.corrcoef(pm, gm)[0, 1]:+.3f} MAE {np.abs(pm - gm).mean():.4f} | mean within-person r {np.mean(within_r):+.3f}")


FPS = 25
LAGS_S = (-4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6)
SMOOTH_S = (0, 0.5, 1, 2, 4, 8)


def shift(x: np.ndarray, k: int) -> np.ndarray:
    """y[t] = x[t - k], edge-padded."""
    if k == 0:
        return x
    return np.concatenate([np.full(k, x[0]), x[:-k]]) if k > 0 else np.concatenate([x[-k:], np.full(-k, x[-1])])


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    xp = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(xp, np.ones(w) / w, mode="valid")


def scores(gt: list[np.ndarray], pred: list[np.ndarray]) -> tuple[float, float]:
    r = [np.corrcoef(x, y)[0, 1] for x, y in zip(pred, gt) if x.std() > 0 and y.std() > 0]
    return official_ccc(np.concatenate(pred), np.concatenate(gt)), float(np.mean(r))


def lag_check(gt: list[np.ndarray], preds: list[list[np.ndarray]]) -> None:
    """preds: [model][person] -> prints one line per lag and per smoothing width."""
    ens = [np.mean([m[i] for m in preds], axis=0) for i in range(len(gt))] if len(preds) > 1 else None

    def line(label: str, f) -> None:
        single = np.array([scores(gt, [f(x) for x in m]) for m in preds])
        out = f"  {label:14s} single models: CCC {single[:, 0].mean():.4f}  within r {single[:, 1].mean():+.3f}"
        if ens is not None:
            c, r = scores(gt, [f(x) for x in ens])
            out += f" | mean of {len(preds)}: CCC {c:.4f}  within r {r:+.3f}"
        print(out)

    print("  -- shift (positive = prediction delayed) --")
    for s_ in LAGS_S:
        line(f"lag {s_:+.1f} s", lambda x, k=int(round(s_ * FPS)): shift(x, k))
    print("  -- centred moving average --")
    for s_ in SMOOTH_S:
        line(f"smooth {s_:.1f} s", lambda x, w=int(round(s_ * FPS)): smooth(x, w))


BANDS_S = (2, 10, 60)
BAND_NAMES = ("fast <2 s", "mid 2-10 s", "slow 10-60 s", "very slow >60 s")


def bands(x: np.ndarray) -> list[np.ndarray]:
    ma = [smooth(x, int(round(s_ * FPS))) for s_ in BANDS_S]
    return [x - ma[0], ma[0] - ma[1], ma[1] - ma[2], ma[2] - x.mean()]


def band_check(gt: list[np.ndarray], preds: list[list[np.ndarray]]) -> None:
    """preds: [model][person] -> one line per timescale band."""
    models = preds + ([[np.mean([m[i] for m in preds], axis=0) for i in range(len(gt))]] if len(preds) > 1 else [])
    gb = [bands(y) for y in gt]
    within = sum(y.var() * len(y) for y in gt)
    print("  -- timescale bands of the within-person deviation --")
    for b, name in enumerate(BAND_NAMES):
        share = sum(x[b].var() * len(x[b]) for x in gb) / within
        rs, oracle = [], []
        for m in models:
            pb = [bands(x) for x in m]
            rs.append(np.mean([np.corrcoef(p[b], g[b])[0, 1] for p, g in zip(pb, gb) if p[b].std() > 0 and g[b].std() > 0]))
            oracle.append(official_ccc(np.concatenate([x - p[b] + g[b] for x, p, g in zip(m, pb, gb)]), np.concatenate(gt)))
        out = f"  {name:16s} label variance {share:5.1%} | single models: band r {np.mean(rs[:len(preds)]):+.3f}  CCC with true band {np.mean(oracle[:len(preds)]):.4f}"
        if len(preds) > 1:
            out += f" | mean of {len(preds)}: band r {rs[-1]:+.3f}  CCC with true band {oracle[-1]:.4f}"
        print(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--pred-dirs", nargs="+", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--split", default="val")
    ap.add_argument("--lag-check", action="store_true")
    ap.add_argument("--bands", action="store_true")
    args = ap.parse_args()

    pred_dirs = [Path(d) / args.corpus / args.split for d in args.pred_dirs]
    people = load_people(Path(args.data_root) / args.corpus / args.split, pred_dirs)
    if not people:
        raise SystemExit("no person found with ground truth and predictions in every dir")
    gt = [x[1] for x in people]

    g = np.concatenate(gt)
    w = np.array([len(x) for x in gt], dtype=float)
    gm = np.array([x.mean() for x in gt])
    between = np.sum(w * (gm - g.mean()) ** 2) / w.sum()
    print(f"== {args.corpus}/{args.split}: {len(people)} people, {len(g)} frames ==")
    print(f"  label variance {g.var():.5f}: between-person {between / g.var():.1%}, within-person {1 - between / g.var():.1%}")
    print(f"  person means: min {gm.min():.3f} max {gm.max():.3f} sd {gm.std():.3f}; mean within-person sd {np.mean([x.std() for x in gt]):.3f}")

    for k, d in enumerate(args.pred_dirs):
        report(Path(d).name, gt, [x[2][k] for x in people])
    if len(args.pred_dirs) > 1:
        report(f"mean of {len(args.pred_dirs)} (not rescaled)", gt, [np.mean(x[2], axis=0) for x in people])
    if args.lag_check:
        lag_check(gt, [[x[2][k] for x in people] for k in range(len(args.pred_dirs))])
    if args.bands:
        band_check(gt, [[x[2][k] for x in people] for k in range(len(args.pred_dirs))])


if __name__ == "__main__":
    main()
