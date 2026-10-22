#!/usr/bin/env python
"""Read out the learned modality-ordering matrix P on the validation windows.

Same flags as inference.py (corpus, --checkpoint-dir, --active-modalities, ...). Prints:
  - the mean P over all validation windows (rows = positions, columns = streams)
  - sharpness: mean of the largest entry per row (1.0 = hard permutation) and the share of rows above 0.9
  - the most frequent rounded orders (argmax per row), and how many windows give a valid permutation when rounded

usage (repo root): python scripts/dump_P.py --corpus NoXi --checkpoint-dir $PWD/models_feat_noxi_t_s95 --active-modalities ...
"""

from __future__ import annotations

import collections
import jax
import numpy as np
import orbax.checkpoint as ocp
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import EngageNetConfig
from data_loader import iter_batches
from inference import find_latest_checkpoint
from train import _drop_meta, create_train_state
from read_data import log


def main() -> None:
    cnfg = EngageNetConfig.from_cli()
    if cnfg.fixed_order is not None:
        raise SystemExit("dump_P.py is for learned-ordering checkpoints; drop --fixed-order")

    state = create_train_state(cnfg, jax.random.PRNGKey(cnfg.seed))
    ckpt = find_latest_checkpoint(cnfg.checkpoint_dir)
    restored = ocp.StandardCheckpointer().restore(ckpt, {"params": state.params, "batch_stats": state.batch_stats})
    variables = {"params": restored["params"], "batch_stats": restored["batch_stats"]}

    @jax.jit
    def get_P(inputs):
        _, inter = state.apply_fn(variables, inputs, tau=cnfg.tau_min, rng=None, train=False, mutable=["batch_stats", "intermediates"])
        return inter["intermediates"]["inter_modal"]["P"][0]  # (B, M, M)

    Ps = []
    for batch in iter_batches(cnfg, split="val"):
        inputs = {k: v for k, v in _drop_meta(batch).items() if not (k.endswith(".engagement") or k.endswith(".gender"))}
        Ps.append(np.asarray(get_P(inputs)))
    P = np.concatenate(Ps, axis=0)  # (N, M, M)
    keys = sorted(f"{r}.{f}" for f in cnfg.modality_names for r in ("expert", "novice"))
    short = [k.replace("expert.", "E.").replace("novice.", "N.").replace("audio.", "").replace("_embeddings", "") for k in keys]

    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    log.info(f"checkpoint {ckpt}  windows {P.shape[0]}  tau {cnfg.tau_min}")
    log.info("streams (columns): " + "  ".join(f"{j}={s}" for j, s in enumerate(short)))
    log.info("mean P (rows = positions):\n" + str(P.mean(axis=0)))
    row_max = P.max(axis=-1)
    log.info(f"sharpness: mean row max {row_max.mean():.3f}, rows with max > 0.9: {(row_max > 0.9).mean():.1%}")

    argmax = P.argmax(axis=-1)  # (N, M)
    valid = np.array([len(set(a.tolist())) == len(a) for a in argmax])
    log.info(f"rounded orders that are valid permutations: {valid.mean():.1%}")
    counts = collections.Counter(tuple(a.tolist()) for a in argmax)
    for order, n in counts.most_common(5):
        log.info(f"{n / len(argmax):6.1%}  " + " ".join(short[j] for j in order))
    log.info(f"distinct rounded orders: {len(counts)}")


if __name__ == "__main__":
    main()
