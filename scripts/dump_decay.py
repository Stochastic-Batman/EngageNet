#!/usr/bin/env python
"""Read out how much state the inter-modal BiMamba keeps per step along the modality axis.

Same flags as inference.py. For the forward and the backward scan, prints the distribution of
A_bar = exp(delta * A) over all channels, state coordinates, positions and validation windows, and the
mean of A_bar^(M-1), the share of the state that survives the whole sequence of M streams.
A_bar near 1 everywhere means long memory (with shared forward/backward parameters and no convolution,
the scan would then not depend on the order); A_bar near 0 means each position sees little beyond itself.

usage (repo root): python scripts/dump_decay.py --corpus NoXi --checkpoint-dir $PWD/models_ord_blocked_noxi_s95 --fixed-order blocked --active-modalities ...
"""

from __future__ import annotations

import jax
import numpy as np
import orbax.checkpoint as ocp
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import EngageNetConfig
from data_loader import iter_batches
from inference import find_latest_checkpoint, resolve_tau
from train import _drop_meta, create_train_state
from read_data import log


def main() -> None:
    cnfg = EngageNetConfig.from_cli()
    state = create_train_state(cnfg, jax.random.PRNGKey(cnfg.seed))
    ckpt = find_latest_checkpoint(cnfg.checkpoint_dir)
    restored = ocp.StandardCheckpointer().restore(ckpt, {"params": state.params, "batch_stats": state.batch_stats})
    variables = {"params": restored["params"], "batch_stats": restored["batch_stats"]}
    tau = resolve_tau(cnfg)

    @jax.jit
    def get_stats(inputs):
        _, inter = state.apply_fn(variables, inputs, tau=tau, rng=None, train=False, mutable=["batch_stats", "intermediates"])
        blk = inter["intermediates"]["inter_modal"]["cross_modal_bimamba"]
        return {k: v[0] for k, v in blk.items()}

    hist = {"fwd": np.zeros(20), "bwd": np.zeros(20)}
    reach = {"fwd": [], "bwd": []}
    for batch in iter_batches(cnfg, split="val"):
        inputs = {k: v for k, v in _drop_meta(batch).items() if not (k.endswith(".engagement") or k.endswith(".gender"))}
        st = get_stats(inputs)
        for d in ("fwd", "bwd"):
            hist[d] += np.asarray(st[f"abar_hist_{d}"])
            reach[d].append(float(st[f"abar_reach_{d}"]))

    edges = np.linspace(0, 1, 21)
    log.info(f"checkpoint {ckpt}  tau {tau:.4f}")
    for d in ("fwd", "bwd"):
        h = hist[d] / hist[d].sum()
        cdf = np.cumsum(h)
        q = {p: edges[1 + int(np.searchsorted(cdf, p))] for p in (0.1, 0.5, 0.9)}
        log.info(f"{d}: A_bar <0.1: {h[:2].sum():.1%}  >0.9: {h[18:].sum():.1%}  quantiles (bin upper edges) 10% {q[0.1]:.2f} 50% {q[0.5]:.2f} 90% {q[0.9]:.2f}  mean A_bar^(M-1) {np.mean(reach[d]):.3f}")
        log.info(f"{d} histogram (20 bins on [0,1]): " + " ".join(f"{x:.3f}" for x in h))


if __name__ == "__main__":
    main()
