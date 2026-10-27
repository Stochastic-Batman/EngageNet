from __future__ import annotations

import dataclasses
import functools
import json
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregator import aggregate_windows
from beta_head import predictive_mean, predictive_variance
from config import EngageNetConfig
from dataset import EngageNetDataset
from model import EngageNet
from read_data import ROLES, load_session, log
from train import TrainState, create_train_state
from tta import count_adapted, make_tta_tx, select_windows, tta_step, window_variance


# Default targets the challenge submission (test has no labels, so evaluate.py
# cannot score it). Use --submission-split val to produce scoreable predictions.
SUBMISSION_CORPORA = [
    ("NoXi", "test-base"),
    ("NoXi", "test-additional"),
    ("NoXi+J", "test"),
]


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    best = checkpoint_dir / "best"
    if best.exists():
        return best
    ckpts = sorted(checkpoint_dir.glob("EngageNet_*"), key=lambda p: int(p.name.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")
    return ckpts[-1]


# Temperature the model is run at: --infer-tau, else the tau of the epoch the best checkpoint came from, else tau_min.
# Before best_tau.json existed every model was run at tau_min, although it was trained and selected at a higher tau.
def resolve_tau(cnfg: EngageNetConfig) -> float:
    if cnfg.infer_tau is not None:
        return cnfg.infer_tau
    f = cnfg.checkpoint_dir / "best_tau.json"
    if f.exists():
        return float(json.loads(f.read_text())["tau"])
    return cnfg.tau_min


# Fewer windows than this and batch quantiles are meaningless, so the last short batch is never adapted on
TTA_MIN_BATCH = 4


# -> (multimodal: dict{role: (mean, variance)}, unimodal: dict{"{role}.{feat}": (mean, variance)}), each (B, L')
# mc_samples = 0: one deterministic pass, Beta mean and variance (uncertainty method 1).
# mc_samples = T > 0: T passes with dropout on (BatchNorm stays in eval mode); mean of the T means, and the total variance of
# the mixture = mean of the T Beta variances (aleatoric) + variance of the T means (epistemic) (method 3).
@functools.partial(jax.jit, static_argnames=("mc_samples",))
def predict_batch(state: TrainState, stream_inputs: dict[str, jax.Array], tau: float, mc_samples: int = 0, rng: jax.Array | None = None):
    variables = {"params": state.params, "batch_stats": state.batch_stats}

    def moments(heads):
        return {k: (predictive_mean(a, b), predictive_variance(a, b)) for k, (a, b) in heads.items()}

    if mc_samples == 0:
        (multimodal, unimodal), _ = state.apply_fn(variables, stream_inputs, tau=tau, rng=None, train=False, mutable=["batch_stats"])
        return moments(multimodal), moments(unimodal)

    def one_pass(key):
        (multimodal, unimodal), _ = state.apply_fn(variables, stream_inputs, tau=tau, rng=None, train=False, mutable=["batch_stats"], rngs={"dropout": key})
        return moments(multimodal), moments(unimodal)

    passes = jax.lax.map(one_pass, jax.random.split(rng, mc_samples))  # every leaf gets a leading axis of length T

    def total(heads):
        return {k: (mu.mean(0), var.mean(0) + mu.var(0)) for k, (mu, var) in heads.items()}

    return total(passes[0]), total(passes[1])


def _stack(windows: list[dict]) -> dict[str, jax.Array]:
    # Stream tensors only: engagement / gender / session are not model inputs
    keys = [k for k, v in windows[0].items() if isinstance(v, np.ndarray) and not k.endswith(".engagement") and not k.endswith(".gender")]
    return {k: jnp.asarray(np.stack([w[k] for w in windows], axis=0)) for k in keys}


# tta_state: None -> plain inference with `state`; otherwise a fresh TTA state (reset per session by the caller)
def run_session(session_dir: Path, state: TrainState, cnfg: EngageNetConfig, tta_state: TrainState | None = None, tau: float | None = None) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    if tta_state is None or not cnfg.tta_person:
        return _run_session(session_dir, state, cnfg, tta_state, tau)
    # One adaptation per person: every role starts again from the trained model, only that role's heads select windows
    # and give the loss, and only that role's predictions are kept from the pass
    result: dict[str, np.ndarray] = {}
    stats = {"windows": 0, "selected": 0, "steps": 0}
    for role in ROLES:
        preds, role_stats = _run_session(session_dir, state, cnfg, tta_state, tau, roles=(role,))
        result[role] = preds[role]
        for k in stats:
            stats[k] += role_stats[k]
    return result, stats


# roles: None -> TTA filter and loss over both people; else only these roles' heads
def _run_session(session_dir: Path, state: TrainState, cnfg: EngageNetConfig, tta_state: TrainState | None, tau: float | None, roles: tuple[str, ...] | None = None) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    ds = EngageNetDataset(cnfg, split=None, session_dirs=[session_dir])

    window_preds: dict[str, list[np.ndarray]] = {role: [] for role in ROLES}
    window_starts: list[int] = []
    stats = {"windows": 0, "selected": 0, "steps": 0}
    st = tta_state
    tau = cnfg.tau_min if tau is None else tau
    mc_key = jax.random.PRNGKey(cnfg.seed)
    n_batches = 0

    def flush(buf: list[dict]) -> None:
        nonlocal st, n_batches
        inputs = _stack(buf)
        key = jax.random.fold_in(mc_key, n_batches)  # dropout masks of the MC passes (unused without --mc-samples)
        n_batches += 1

        if st is not None and len(buf) >= TTA_MIN_BATCH:
            multimodal, unimodal = predict_batch(st, inputs, tau, cnfg.mc_samples, key)
            if roles is not None:
                multimodal = {r: multimodal[r] for r in roles}
                unimodal = {k: v for k, v in unimodal.items() if k.split(".", 1)[0] in roles}
            mask = select_windows(window_variance(multimodal), window_variance(unimodal), multi_pct=cnfg.tta_multi_pct, uni_pct=cnfg.tta_uni_pct)
            n_sel = int(mask.sum())
            if n_sel > 0:
                st, _loss = tta_step(st, inputs, mask, tau, lam=cnfg.tta_lambda, roles=roles)
                stats["selected"] += n_sel
                stats["steps"] += 1

        # Predictions always come from AFTER the update, in eval mode
        multimodal, _ = predict_batch(st if st is not None else state, inputs, tau, cnfg.mc_samples, jax.random.fold_in(key, 1))
        for role in ROLES:
            means = np.asarray(multimodal[role][0])  # (B, L')
            window_preds[role].extend(list(means))
        stats["windows"] += len(buf)

    buf: list[dict] = []
    for start, window in ds.iter_windows_with_starts():
        window_starts.append(start)
        buf.append(window)
        if len(buf) == cnfg.infer_batch:
            flush(buf)
            buf = []
    if buf:
        flush(buf)

    # Determine total session length from engagement annotations or stream length
    session = load_session(session_dir)
    total_frames = min(
        session[role].get("engagement").shape[0] if session[role].get("engagement") is not None
        else max(s["data"].shape[0] for s in session[role].get("streams", {}).values())
        for role in ROLES
    )

    result = {}
    for role in ROLES:
        result[role] = aggregate_windows(window_preds[role], window_starts, total_frames)

    return result, stats


def main():
    cnfg = EngageNetConfig.from_cli()

    rng = jax.random.PRNGKey(cnfg.seed)
    rng, rng_init = jax.random.split(rng)

    state = create_train_state(cnfg, rng_init)

    ckpt_path = find_latest_checkpoint(cnfg.checkpoint_dir)
    log.info(f"Loading checkpoint: {ckpt_path}")
    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(ckpt_path, {"params": state.params, "batch_stats": state.batch_stats})
    state = state.replace(params=restored["params"], batch_stats=restored["batch_stats"])

    tau = resolve_tau(cnfg)
    log.info(f"Inference temperature tau = {tau:.4f}")
    if cnfg.mc_samples > 0:
        if cnfg.dropout <= 0:
            raise SystemExit("--mc-samples needs a model trained with --dropout > 0 (and the same --dropout at inference)")
        log.info(f"MC dropout: {cnfg.mc_samples} passes at rate {cnfg.dropout}; predictions are the mean of the passes, the TTA filter uses the total variance")

    tta_template = None
    if cnfg.tta:
        n_adapt = count_adapted(state.params)
        if n_adapt == 0:
            raise RuntimeError("TTA: surgical_mask matched no parameters - check layer names in tta.surgical_mask")
        tta_template = TrainState.create(apply_fn=state.apply_fn, params=state.params, tx=make_tta_tx(state.params, cnfg.tta_lr), batch_stats=state.batch_stats)
        log.info(f"TTA on: adapting {n_adapt:,} params, lr={cnfg.tta_lr}, lambda={cnfg.tta_lambda}, batch={cnfg.infer_batch}, pct=({cnfg.tta_multi_pct}, {cnfg.tta_uni_pct}), reset per {'person' if cnfg.tta_person else 'session'}")
    else:
        log.info(f"TTA off, batch={cnfg.infer_batch}")

    cnfg.submission_dir.mkdir(parents=True, exist_ok=True)

    corpora = [(c, cnfg.submission_split or s) for c, s in SUBMISSION_CORPORA]
    corpora = list(dict.fromkeys(corpora))

    for corpus_name, split in corpora:
        sub_cnfg = dataclasses.replace(cnfg, corpus=corpus_name)
        split_path = sub_cnfg.split_dir(split)

        if not split_path.exists():
            log.info(f"Skipping {corpus_name}/{split} (not found)")
            continue

        session_dirs = sorted([p for p in split_path.iterdir() if p.is_dir()])
        log.info(f"{corpus_name}/{split}: {len(session_dirs)} sessions")

        for session_dir in session_dirs:
            # tta_template is immutable, so passing it again resets params and optimizer state for every session
            preds, stats = run_session(session_dir, state, sub_cnfg, tta_template, tau)

            out_dir = cnfg.submission_dir / corpus_name / split / session_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            for role, pred_arr in preds.items():
                out_path = out_dir / f"{role}.engagement.pred.csv"
                np.savetxt(out_path, pred_arr, fmt="%.6f", delimiter=";")

            tta_info = f", TTA: {stats['selected']}/{stats['windows']} windows selected, {stats['steps']} steps" if cnfg.tta else ""
            log.info(f"  {session_dir.name}: {preds[ROLES[0]].shape[0]} frames written{tta_info}")

    log.info("Inference complete")


if __name__ == "__main__":
    main()
