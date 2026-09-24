"""Average several prediction folders frame by frame, optionally rescale, score and/or write the result.

One folder  -> CCC plus its decomposition CCC = rho * C_b, and prediction vs. label mean/std.
Several     -> the same for the frame-wise AVERAGE of the folders (a seed/window ensemble).

--rescale   Averaging shrinks the spread of the predictions. This stretches the ensemble back to the
            average mean and std of its members, pooled over all frames of the corpus split.
            Uses predictions only, never labels, so it is legitimate on test data.
--write-dir Write the ensemble as <write-dir>/<corpus>/<split>/<session>/<role>.engagement.pred.csv.
            Works on test splits too (scoring is skipped when there are no labels).

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
    ap.add_argument("--rescale", action="store_true", help="restore the members' average mean/std (label-free)")
    ap.add_argument("--write-dir", default=None, help="write the ensemble here in the official layout")
    args = ap.parse_args()

    members = [Path(m) / args.corpus / args.split for m in args.members]
    for m in members:
        if not m.exists():
            sys.exit(f"missing predictions: {m}")

    # 1. Frame-wise average per (session, role); keep every member's own series for the rescale statistics
    sessions = sorted(p.name for p in members[0].iterdir() if p.is_dir())
    ens: dict[tuple[str, str], np.ndarray] = {}
    member_series: list[list[np.ndarray]] = [[] for _ in members]
    for s in sessions:
        for role in ROLES:
            paths = [find_pred(m / s, role) for m in members]
            if not all(p.exists() for p in paths):
                continue
            preds = [load_csv(p) for p in paths]
            n = min(len(p) for p in preds)
            stacked = np.stack([q[:n] for q in preds])
            ens[(s, role)] = stacked.mean(axis=0)
            for i in range(len(members)):
                member_series[i].append(stacked[i])

    if not ens:
        sys.exit("no (session, role) pairs found in all members")

    # 2. Optional label-free rescale, pooled over the whole corpus split
    if args.rescale:
        pooled = np.concatenate(list(ens.values()))
        m_means = [np.nanmean(np.concatenate(ser)) for ser in member_series]
        m_stds = [np.nanstd(np.concatenate(ser)) for ser in member_series]
        mu, sd = np.nanmean(pooled), np.nanstd(pooled)
        tgt_mu, tgt_sd = float(np.mean(m_means)), float(np.mean(m_stds))
        scale = tgt_sd / sd if sd > 0 else 1.0
        ens = {k: np.clip(tgt_mu + (v - mu) * scale, 0.0, 1.0).astype(np.float32) for k, v in ens.items()}

    label = args.label or "+".join(Path(m).name for m in args.members)
    label += " [rescaled]" if args.rescale else ""

    # 3. Optional write in the official layout
    if args.write_dir:
        for (s, role), v in ens.items():
            out = Path(args.write_dir) / args.corpus / args.split / s
            out.mkdir(parents=True, exist_ok=True)
            np.savetxt(out / f"{role}.engagement.pred.csv", v, fmt="%.6f")
        print(f"{label}: wrote {len(ens)} files for {len(sessions)} sessions to {Path(args.write_dir) / args.corpus / args.split}")

    # 4. Score when labels exist (official pooled CCC, truncate to the shorter series, drop non-finite)
    gt_dir = Path(args.data_root) / args.corpus / args.split
    all_p, all_t = [], []
    for (s, role), v in ens.items():
        gt_path = gt_dir / s / f"{role}.engagement.annotation.csv"
        if not gt_path.exists():
            continue
        gt = load_csv(gt_path)
        n = min(len(gt), len(v))
        p, t = v[:n], gt[:n]
        mask = np.isfinite(p) & np.isfinite(t)
        all_p.append(p[mask])
        all_t.append(t[mask])

    if not all_p:
        print(f"{label}: no labels for {args.corpus}/{args.split}, not scored")
        return

    p, t = np.concatenate(all_p), np.concatenate(all_t)
    c = official_ccc(t, p)
    rho = float(np.corrcoef(p, t)[0, 1])
    cb = c / rho if rho != 0 else float("nan")
    print(f"{label}: CCC={c:.4f} rho={rho:.4f} C_b={cb:.4f} | pred mean={p.mean():.3f} std={p.std():.3f} | label mean={t.mean():.3f} std={t.std():.3f} | n_members={len(members)} frames={len(p)}")


if __name__ == "__main__":
    main()
