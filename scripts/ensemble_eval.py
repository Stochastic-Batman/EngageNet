"""Score one or more prediction folders against ground truth, with the official pooled CCC.

One folder  -> CCC plus its decomposition CCC = rho * C_b, and prediction vs. label mean/std.
Several     -> the same for the frame-wise AVERAGE of the folders (a seed/window ensemble).

Folders are anything inference.py wrote with --submission-dir (layout <dir>/<corpus>/<split>/<session>/).
CPU only, reads CSVs, needs no model.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluate import find_pred, load_csv, official_ccc  # noqa: E402
from read_data import ROLES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True, help="prediction folders (--submission-dir of each run)")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--label", default=None, help="name printed at the start of the output line")
    args = ap.parse_args()

    gt_dir = Path(args.data_root) / args.corpus / args.split
    members = [Path(m) / args.corpus / args.split for m in args.members]
    for m in members:
        if not m.exists():
            sys.exit(f"missing predictions: {m}")

    all_p, all_t = [], []
    for session_dir in sorted(p for p in gt_dir.iterdir() if p.is_dir()):
        for role in ROLES:
            gt_path = session_dir / f"{role}.engagement.annotation.csv"
            pred_paths = [find_pred(m / session_dir.name, role) for m in members]
            if not gt_path.exists() or not all(p.exists() for p in pred_paths):
                continue
            gt = load_csv(gt_path)
            preds = [load_csv(p) for p in pred_paths]
            n = min(len(gt), *(len(p) for p in preds))
            p = np.mean(np.stack([q[:n] for q in preds]), axis=0)
            t = gt[:n]
            mask = np.isfinite(p) & np.isfinite(t)
            all_p.append(p[mask])
            all_t.append(t[mask])

    p, t = np.concatenate(all_p), np.concatenate(all_t)
    c = official_ccc(t, p)
    rho = float(np.corrcoef(p, t)[0, 1])
    cb = c / rho if rho != 0 else float("nan")
    label = args.label or "+".join(Path(m).name for m in args.members)
    print(f"{label}: CCC={c:.4f} rho={rho:.4f} C_b={cb:.4f} | pred mean={p.mean():.3f} std={p.std():.3f} | label mean={t.mean():.3f} std={t.std():.3f} | n_members={len(members)} frames={len(p)}")


if __name__ == "__main__":
    main()
