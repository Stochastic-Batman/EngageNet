from __future__ import annotations

import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregator import aggregate_windows
from beta_head import predictive_mean
from config import EngageNetConfig
from dataset import EngageNetDataset
from model import EngageNet
from read_data import ROLES, load_session, log
from train import TrainState, create_train_state
from tta import sample_filter, tta_step


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


def run_session(session_dir: Path, state: TrainState, cnfg: EngageNetConfig, rng: jax.Array) -> dict[str, np.ndarray]:  # {role: (T,) predictions}
    ds = EngageNetDataset(cnfg, split=None, session_dirs=[session_dir])

    window_preds: dict[str, list[np.ndarray]] = {role: [] for role in ROLES}
    window_starts: list[int] = []

    idx = 0
    for window in ds.iter_windows():
        start = idx * cnfg.window_stride
        window_starts.append(start)

        stream_inputs = {}
        for key, val in window.items():
            if isinstance(val, np.ndarray) and not key.endswith(".engagement"):
                stream_inputs[key] = jnp.array(val[np.newaxis])

        variables = {"params": state.params, "batch_stats": state.batch_stats}
        (alpha, beta, unimodal), _ = state.apply_fn(variables, stream_inputs, tau=cnfg.tau_min, rng=None, train=False, mutable=["batch_stats"])

        # Pool over time for TTA sample filter (needs scalar uncertainty per sample)
        alpha_pooled = alpha.mean(axis=-1)
        beta_pooled = beta.mean(axis=-1)
        unimodal_pooled = {k: (a.mean(axis=-1), b.mean(axis=-1)) for k, (a, b) in unimodal.items()}
        mask = sample_filter(alpha_pooled, beta_pooled, unimodal_pooled)
        if mask.any():
            rng, rng_tta = jax.random.split(rng)
            state, _loss, alpha, beta, unimodal = tta_step(state, stream_inputs, tau=cnfg.tau_min, rng=rng_tta)

        pred = np.array(predictive_mean(alpha, beta)[0])  # (L',)

        for role in ROLES:
            window_preds[role].append(pred)

        idx += 1

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

    return result


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
            rng, rng_session = jax.random.split(rng)
            preds = run_session(session_dir, state, sub_cnfg, rng_session)

            out_dir = cnfg.submission_dir / corpus_name / split / session_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            for role, pred_arr in preds.items():
                out_path = out_dir / f"{role}.engagement.annotation.csv"
                np.savetxt(out_path, pred_arr, fmt="%.6f", delimiter=";")

            log.info(f"  {session_dir.name}: {preds[ROLES[0]].shape[0]} frames written")

    log.info("Inference complete")


if __name__ == "__main__":
    main()
