#!/usr/bin/env python
"""Where in the network is the information about engagement? Linear probes on the output of each stage.

Same flags as inference.py (corpus, --checkpoint-dir, --active-modalities, --fusion-temporal, ...). Runs the trained model on
the validation windows, keeps every --every-th frame (default 10; windows overlap by half, so frames are not repeated), and
records per frame the outputs of
  frontend   InitEncoder of every stream (local context of a few frames, no temporal modelling)
  intra      intra-modal BiMamba of every stream (temporal context within the stream)
  inter      fused representation H of the inter-modal block (same for both roles)
  inter+z    [H, z] with the temporal block after fusion, if the model has one (--fusion-temporal)
For each stage it fits a ridge regression to the engagement label, one per role, and evaluates it leave-one-session-out:
"own" uses only that person's streams, "both" both people's. The ridge strength is the best of a small grid by the same
leave-one-session-out score (slightly optimistic, equally for every stage). Reported: pooled CCC over the kept frames, and
the mean within-person Pearson r. The model's own prediction on the same frames is given for reference.

Reading: intra(both) ~ inter -> the inter-modal block adds no linearly readable information; frontend ~ intra -> the
temporal modelling within streams adds little; probe well below the model -> the heads use non-linear structure.

usage (repo root): python scripts/probe_stages.py --corpus NoXi --checkpoint-dir $PWD/models_feat_noxi_t_s95 --shuffle-windows --active-modalities ... [--every 10]
"""

from __future__ import annotations

import collections
import jax
import numpy as np
import orbax.checkpoint as ocp
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from beta_head import predictive_mean
from config import EngageNetConfig
from data_loader import iter_batches
from evaluate import official_ccc
from inference import find_latest_checkpoint, resolve_tau
from read_data import ROLES, log
from train import _drop_meta, create_train_state

ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4)
STAGES_OUT = ("frontend", "intra_modal", "inter_modal", "fusion_temporal")


def ridge_loso(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Leave-one-group-out ridge predictions for the best alpha of ALPHAS (by pooled CCC). X standardised, intercept unpenalised."""
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    X = np.concatenate([X, np.ones((len(X), 1), X.dtype)], axis=1).astype(np.float64)
    pen = np.ones(X.shape[1])
    pen[-1] = 0.0
    G_s = {g: X[groups == g].T @ X[groups == g] for g in np.unique(groups)}
    c_s = {g: X[groups == g].T @ y[groups == g] for g in G_s}
    G, c = sum(G_s.values()), sum(c_s.values())
    best, best_pred = -np.inf, None
    for a in ALPHAS:
        pred = np.empty_like(y)
        for g in G_s:
            w = np.linalg.solve(G - G_s[g] + a * np.diag(pen), c - c_s[g])
            pred[groups == g] = X[groups == g] @ w
        score = official_ccc(pred, y)
        if score > best:
            best, best_pred = score, pred
    return best_pred


def within_r(pred: np.ndarray, y: np.ndarray, person: np.ndarray) -> float:
    rs = [np.corrcoef(pred[person == p], y[person == p])[0, 1] for p in np.unique(person) if y[person == p].std() > 0 and pred[person == p].std() > 0]
    return float(np.mean(rs))


def main() -> None:
    argv = sys.argv[1:]
    every = 10
    if "--every" in argv:
        i = argv.index("--every")
        every = int(argv[i + 1])
        del sys.argv[1 + i:3 + i]
    cnfg = EngageNetConfig.from_cli()

    state = create_train_state(cnfg, jax.random.PRNGKey(cnfg.seed))
    ckpt = find_latest_checkpoint(cnfg.checkpoint_dir)
    restored = ocp.StandardCheckpointer().restore(ckpt, {"params": state.params, "batch_stats": state.batch_stats})
    variables = {"params": restored["params"], "batch_stats": restored["batch_stats"]}
    tau = resolve_tau(cnfg)
    keys = sorted(f"{r}.{f}" for f in cnfg.modality_names for r in ROLES)

    @jax.jit
    def run(inputs):
        (multimodal, _), inter = state.apply_fn(variables, inputs, tau=tau, rng=None, train=False, mutable=["batch_stats", "intermediates"],
                                               capture_intermediates=lambda mdl, name: name == "__call__" and mdl.name in STAGES_OUT)
        im = inter["intermediates"]
        out = {s: im[s]["__call__"][0] for s in STAGES_OUT if s in im}
        out["pred"] = {r: predictive_mean(a, b) for r, (a, b) in multimodal.items()}
        return out

    feats = collections.defaultdict(list)  # name -> list of (n, d) arrays
    lab = {r: [] for r in ROLES}
    pred = {r: [] for r in ROLES}
    sess = []
    for batch in iter_batches(cnfg, split="val"):
        inputs = {k: v for k, v in _drop_meta(batch).items() if not (k.endswith(".engagement") or k.endswith(".gender"))}
        out = jax.device_get(run(inputs))
        sl = slice(0, None, every)
        for k in keys:
            feats[f"frontend:{k}"].append(np.asarray(out["frontend"][k][:, sl]).reshape(-1, out["frontend"][k].shape[-1]).astype(np.float16))
            feats[f"intra:{k}"].append(np.asarray(out["intra_modal"][k][:, sl]).reshape(-1, out["intra_modal"][k].shape[-1]).astype(np.float16))
        feats["inter"].append(np.asarray(out["inter_modal"][:, sl]).reshape(-1, out["inter_modal"].shape[-1]).astype(np.float16))
        if "fusion_temporal" in out:
            feats["z"].append(np.asarray(out["fusion_temporal"][:, sl]).reshape(-1, out["fusion_temporal"].shape[-1]).astype(np.float16))
        n_t = len(range(*sl.indices(out["inter_modal"].shape[1])))
        for r in ROLES:
            lab[r].append(np.asarray(batch[f"{r}.engagement"])[:, sl].reshape(-1))
            pred[r].append(np.asarray(out["pred"][r])[:, sl].reshape(-1))
        sess.append(np.repeat(np.asarray(batch["session"]), n_t))

    feats = {k: np.concatenate(v) for k, v in feats.items()}
    lab = {r: np.concatenate(v).astype(np.float64) for r, v in lab.items()}
    pred = {r: np.concatenate(v).astype(np.float64) for r, v in pred.items()}
    sess = np.concatenate(sess)

    def stage(name: str, role: str) -> np.ndarray:
        partner = [r for r in ROLES if r != role][0]
        own = [k for k in keys if k.startswith(role + ".")]
        oth = [k for k in keys if k.startswith(partner + ".")]
        if name in ("frontend own", "intra own"):
            return np.concatenate([feats[f"{name.split()[0]}:{k}"] for k in own], axis=1)
        if name in ("frontend both", "intra both"):
            return np.concatenate([feats[f"{name.split()[0]}:{k}"] for k in own + oth], axis=1)
        if name == "inter":
            return feats["inter"]
        return np.concatenate([feats["inter"], feats["z"]], axis=1)  # inter+z

    names = ["frontend own", "frontend both", "intra own", "intra both", "inter"] + (["inter+z"] if "z" in feats else [])
    log.info(f"checkpoint {ckpt}  every {every}th frame  {len(sess)} frames per role  {len(np.unique(sess))} sessions")
    for name in [None] + names:
        ys, ps, persons = [], [], []
        for r in ROLES:
            ok = np.isfinite(lab[r])
            if name is None:
                p = pred[r][ok]
            else:
                X = stage(name, r)[ok].astype(np.float32)
                p = ridge_loso(X, lab[r][ok], sess[ok])
            ys.append(lab[r][ok])
            ps.append(p)
            persons.append(np.char.add(sess[ok].astype(str), "." + r))
        y, p, person = np.concatenate(ys), np.concatenate(ps), np.concatenate(persons)
        label = "model prediction" if name is None else f"probe {name}"
        log.info(f"{label:22s} CCC {official_ccc(p, y):.4f}  within-person r {within_r(p, y, person):+.3f}")


if __name__ == "__main__":
    main()
