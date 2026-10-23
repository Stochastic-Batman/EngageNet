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

usage (repo root): python scripts/level_oracle.py --corpus NoXi --pred-dirs sweep_out/feat_noxi_t_besttau_s* [--data-root data]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--pred-dirs", nargs="+", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--split", default="val")
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


if __name__ == "__main__":
    main()
