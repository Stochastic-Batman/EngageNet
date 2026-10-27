#!/usr/bin/env python
"""Per-person comparison of two sets of validation predictions, e.g. without and with test-time adaptation.

A person is one role in one session (as in level_oracle.py). For every person present in all prediction dirs of both sets,
it computes the CCC and the Pearson r of that person's frames, and the pooled CCC as src/evaluate.py does. Each set may hold
several dirs (seeds); the per-person scores are then averaged over the seeds, and dir k of --base is paired with dir k of --new.

Printed: pooled CCC of each set (mean over seeds), mean per-person CCC and r of each set, mean change, how many people
improve, and a two-sided Wilcoxon signed-rank test over people of the change in per-person CCC.

usage (repo root): python scripts/person_eval.py --corpus NoXi --base sweep_out/ftb4_noxi_s{95,1,2} --new sweep_out/ftb4_noxi_ttap_s{95,1,2}
"""

from __future__ import annotations

import argparse
import numpy as np
import sys

from pathlib import Path
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import official_ccc
from level_oracle import load_people


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--new", nargs="+", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    if len(args.base) != len(args.new):
        raise SystemExit("--base and --new need the same number of dirs (one per seed, in the same order)")

    dirs = [Path(d) / args.corpus / args.split for d in args.base + args.new]
    people = load_people(Path(args.data_root) / args.corpus / args.split, dirs)
    if not people:
        raise SystemExit("no person found with ground truth and predictions in every dir")
    k = len(args.base)
    gt = [x[1] for x in people]

    def scores(idx: int) -> tuple[float, np.ndarray, np.ndarray]:
        pred = [x[2][idx] for x in people]
        ccc_p = np.array([official_ccc(p, y) for p, y in zip(pred, gt)])
        r_p = np.array([np.corrcoef(p, y)[0, 1] if p.std() > 0 and y.std() > 0 else np.nan for p, y in zip(pred, gt)])
        return official_ccc(np.concatenate(pred), np.concatenate(gt)), ccc_p, r_p

    base = [scores(i) for i in range(k)]
    new = [scores(k + i) for i in range(k)]
    pooled_b, pooled_n = np.array([s[0] for s in base]), np.array([s[0] for s in new])
    ccc_b, ccc_n = np.mean([s[1] for s in base], axis=0), np.mean([s[1] for s in new], axis=0)
    r_b, r_n = np.nanmean([s[2] for s in base], axis=0), np.nanmean([s[2] for s in new], axis=0)
    d = ccc_n - ccc_b

    print(f"== {args.corpus}/{args.split}: {len(people)} people, {k} seed(s) per set ==")
    print(f"  pooled CCC        base {pooled_b.mean():.4f}  new {pooled_n.mean():.4f}  change {pooled_n.mean() - pooled_b.mean():+.4f} (per seed: {' '.join(f'{x:+.4f}' for x in pooled_n - pooled_b)})")
    print(f"  per-person CCC    base {ccc_b.mean():.4f}  new {ccc_n.mean():.4f}  change {d.mean():+.4f}  improved {int((d > 0).sum())}/{len(d)}  Wilcoxon p {wilcoxon(d).pvalue:.3f}")
    print(f"  per-person r      base {np.nanmean(r_b):.4f}  new {np.nanmean(r_n):.4f}  change {np.nanmean(r_n - r_b):+.4f}")
    print(f"  largest changes   {' '.join(f'{people[i][0]} {d[i]:+.3f}' for i in np.argsort(np.abs(d))[::-1][:5])}")


if __name__ == "__main__":
    main()
