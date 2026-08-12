from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from flax.training import train_state
from functools import lru_cache
from pathlib import Path

from beta_head import nll_loss, predictive_mean
from config import EngageNetConfig
from data_loader import iter_batches
from evaluate import build_gender_map
from metrics import cdd_gender, cdd_language, ccc
from model import EngageNet
from read_data import ROLES, log


class TrainState(train_state.TrainState):
    batch_stats: dict


def anneal_tau(tau_init: float, tau_min: float, decay: float, epoch: int) -> float:
    return max(tau_min, tau_init * (decay ** epoch))


@lru_cache(maxsize=None)
def _session_meta_cached(session_path: str) -> tuple[str, str]:
    """(raw gender code, language) for a session; "" gender means unknown.

    Gender is per participant but the training target averages expert+novice, so a
    single code is only meaningful when both roles share it - otherwise return ""
    rather than attribute one participant's gender to a blended target.
    """
    from read_data import read_scalar_annotation

    d = Path(session_path)
    codes = set()
    for role in ROLES:
        p = d / f"{role}.gender.annotation.csv"
        if p.exists():
            codes.add(read_scalar_annotation(p).strip().lower())
    gender = codes.pop() if len(codes) == 1 else ""

    p = d / "language.annotation.csv"
    language = read_scalar_annotation(p).strip() if p.exists() else "unknown"

    return gender, language


def _session_meta(cnfg: EngageNetConfig, session_name: str) -> tuple[str, str]:
    return _session_meta_cached(str(cnfg.split_dir("val") / session_name))


def _drop_meta(batch: dict) -> dict[str, jax.Array]:
    return {k: v for k, v in batch.items() if k != "session"}


# batch: dict{str: (B, C_i, L)}; ... -> (state, loss)
@jax.jit
def train_step(state: TrainState, batch: dict[str, jax.Array], rng: jax.Array, tau: float):
    # Per-frame targets: average expert + novice engagement -> (B, L)
    targets = []
    for role in ROLES:
        key = f"{role}.engagement"
        if key in batch:
            targets.append(batch[key])
    target = jnp.stack(targets, axis=0).mean(axis=0)  # (B, L)

    stream_inputs = {k: v for k, v in batch.items() if not k.endswith(".engagement")}

    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        (alpha, beta, unimodal), updates = state.apply_fn(variables, stream_inputs, tau=tau, rng=rng, train=True, mutable=["batch_stats"])
        loss = nll_loss(alpha, beta, target)

        for _key, (a_i, b_i) in unimodal.items():
            loss = loss + 0.5 * nll_loss(a_i, b_i, target)

        return loss, updates["batch_stats"]

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, new_batch_stats), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_batch_stats)

    return state, loss


@jax.jit
def eval_step(state: TrainState, batch: dict[str, jax.Array], tau: float) -> jax.Array:
    stream_inputs = {k: v for k, v in batch.items() if not k.endswith(".engagement")}
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    (alpha, beta, _), _ = state.apply_fn(variables, stream_inputs, tau=tau, rng=None, train=False, mutable=["batch_stats"])
    return predictive_mean(alpha, beta)  # (B, L')


def val_metrics(state: TrainState, cnfg: EngageNetConfig, tau: float) -> tuple[float, float | None, dict[str, float]]:
    """Return (CCC, CDD_G, CDD_L) over the val split.

    The forward pass already runs every epoch, so the two CDD metrics cost only
    the bookkeeping of carrying gender/language labels alongside each window.
    """
    all_preds = []
    all_targets = []
    all_genders = []
    all_languages = []

    for batch in iter_batches(cnfg, split="val"):
        preds = eval_step(state, _drop_meta(batch), tau)
        n_per_window = int(np.array(preds).reshape(preds.shape[0], -1).shape[1])
        for name in batch["session"]:
            g, lang = _session_meta(cnfg, name)
            all_genders.append(np.full(n_per_window, g, dtype=object))
            all_languages.append(np.full(n_per_window, lang, dtype=object))
        all_preds.append(np.array(preds.reshape(-1)))

        targets = []
        for role in ROLES:
            key = f"{role}.engagement"
            if key in batch:
                targets.append(np.array(batch[key]))
        target = np.stack(targets, axis=0).mean(axis=0)  # (B, L)
        all_targets.append(target.reshape(-1))

    preds = np.concatenate(all_preds)
    targs = np.concatenate(all_targets)
    v_ccc = ccc(preds, targs)

    raw_genders = np.concatenate(all_genders) if all_genders else np.array([], dtype=object)
    v_cdd_g = None
    gmap = build_gender_map(set(raw_genders.tolist())) if raw_genders.size else {}
    if gmap:
        genders = np.array([gmap.get(c, -1) for c in raw_genders], dtype=np.int8)
        n = min(genders.size, preds.size)
        k = genders[:n] >= 0
        if k.any():
            v_cdd_g = cdd_gender(preds[:n][k], targs[:n][k], genders[:n][k])

    langs = np.concatenate(all_languages) if all_languages else np.array([], dtype=object)
    v_cdd_l: dict[str, float] = {}
    if langs.size and len(np.unique(langs)) > 1:
        n = min(langs.size, preds.size)
        v_cdd_l = cdd_language(preds[:n], targs[:n], langs[:n])

    return v_ccc, v_cdd_g, v_cdd_l


def create_train_state(cnfg: EngageNetConfig, rng: jax.Array) -> TrainState:
    model = EngageNet(cnfg=cnfg)

    dummy = {}
    for feat in cnfg.modality_names:
        c_in = cnfg.input_dim(feat)
        for role in ROLES:
            dummy[f"{role}.{feat}"] = jnp.zeros((cnfg.batch_size, c_in, cnfg.window_len))

    rng_init, rng_gumbel = jax.random.split(rng)
    variables = model.init(rng_init, dummy, tau=1.0, rng=rng_gumbel, train=False)

    tx = optax.adamw(learning_rate=cnfg.lr, weight_decay=cnfg.weight_decay)

    return TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx, batch_stats=variables.get("batch_stats", {}))


def main() -> None:
    cnfg = EngageNetConfig.from_cli()

    rng = jax.random.PRNGKey(cnfg.seed)
    rng, rng_init = jax.random.split(rng)

    state = create_train_state(cnfg, rng_init)
    param_count = sum(x.size for x in jax.tree.leaves(state.params))
    log.info(f"Total params: {param_count:,}")

    cnfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()

    best_val_ccc = -1.0
    epochs_without_improvement = 0

    for epoch in range(cnfg.n_epochs):
        tau = anneal_tau(cnfg.tau_init, cnfg.tau_min, cnfg.tau_decay, epoch)
        rng, rng_epoch = jax.random.split(rng)

        epoch_loss = 0.0
        n_batches = 0

        for batch in iter_batches(cnfg, split="train", shuffle=True, seed=cnfg.seed + epoch):
            rng, rng_step = jax.random.split(rng)
            state, loss = train_step(state, _drop_meta(batch), rng_step, tau)
            epoch_loss += float(loss)
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        if (epoch + 1) % cnfg.eval_every == 0:
            v_ccc, v_cdd_g, v_cdd_l = val_metrics(state, cnfg, tau)
            extra = ""
            if v_cdd_g is not None:
                extra += f"  cdd_g={v_cdd_g:+.4f}"
            if v_cdd_l:
                extra += "  cdd_l[" + " ".join(f"{k}={v:+.3f}" for k, v in sorted(v_cdd_l.items())) + "]"
            log.info(f"Epoch {epoch+1:3d}/{cnfg.n_epochs}  tau={tau:.4f}  loss={avg_loss:.4f}  val_ccc={v_ccc:.4f}{extra}")

            if v_ccc > best_val_ccc:
                best_val_ccc = v_ccc
                epochs_without_improvement = 0
                checkpointer.save(cnfg.checkpoint_dir / "best", {"params": state.params, "batch_stats": state.batch_stats}, force=True)
                log.info(f"  New best val CCC: {best_val_ccc:.4f}")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= cnfg.patience:
                log.info(f"Early stopping at epoch {epoch+1} (patience={cnfg.patience})")
                break
        else:
            log.info(f"Epoch {epoch+1:3d}/{cnfg.n_epochs}  tau={tau:.4f}  loss={avg_loss:.4f}")

        if (epoch + 1) % cnfg.checkpoint_every == 0:
            ckpt_path = cnfg.checkpoint_dir / f"EngageNet_{epoch+1}"
            checkpointer.save(ckpt_path, {"params": state.params, "batch_stats": state.batch_stats}, force=True)
            log.info(f"Checkpoint saved: {ckpt_path}")

    # Flush any in-flight async writes before the interpreter tears down its thread pool
    checkpointer.wait_until_finished()
    checkpointer.close()

    log.info(f"Training complete. Best val CCC: {best_val_ccc:.4f}")


if __name__ == "__main__":
    main()
