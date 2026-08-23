from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from flax.training import train_state
from functools import lru_cache, partial
from pathlib import Path

from beta_head import beta_nll_weights, nll_loss, predictive_mean
from config import EngageNetConfig
from data_loader import iter_batches
from metrics import ccc_jnp, cdd_gender, cdd_language, ccc, fairness_penalty
from model import EngageNet
from read_data import log, ROLES, read_scalar_annotation


class TrainState(train_state.TrainState):
    batch_stats: dict


def anneal_tau(tau_init: float, tau_min: float, decay: float, epoch: int) -> float:
    return max(tau_min, tau_init * (decay ** epoch))


@lru_cache(maxsize=None)
def _session_language_cached(session_path: str) -> str:
    p = Path(session_path) / "language.annotation.csv"
    return read_scalar_annotation(p).strip() if p.exists() else "unknown"


def _session_language(cnfg: EngageNetConfig, session_name: str) -> str:
    return _session_language_cached(str(cnfg.split_dir("val") / session_name))


def _drop_meta(batch: dict) -> dict[str, jax.Array]:
    return {k: v for k, v in batch.items() if k != "session"}


# batch: dict{str: (B, C_i, L)} plus "{role}.engagement" (B, L') and "{role}.gender" (B,) -> (state, loss)
@partial(jax.jit, static_argnames=("n_bins",))
def train_step(state: TrainState, batch: dict[str, jax.Array], rng: jax.Array, tau: float, beta_w: float, lambda_ccc: float, lambda_uni: float, lambda_fair: float, n_bins: int = 10):
    targets = {r: batch[f"{r}.engagement"] for r in ROLES if f"{r}.engagement" in batch}
    groups = {r: batch.get(f"{r}.gender") for r in ROLES}
    stream_inputs = {k: v for k, v in batch.items() if not (k.endswith(".engagement") or k.endswith(".gender"))}

    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        (multimodal, unimodal), updates = state.apply_fn(variables, stream_inputs, tau=tau, rng=rng, train=True, mutable=["batch_stats"])

        # 1. Per-frame fit: reweighted Beta NLL, one multimodal head per role
        loss = 0.0
        preds, tgts, grps = [], [], []
        for role, (a, b) in multimodal.items():
            if role not in targets:
                continue
            y = targets[role]
            loss = loss + nll_loss(a, b, y, weights=beta_nll_weights(a, b, beta_w))
            preds.append(predictive_mean(a, b).reshape(-1))
            tgts.append(y.reshape(-1))
            g = groups[role] if groups[role] is not None else -jnp.ones(y.shape[0], jnp.int8)
            grps.append(jnp.broadcast_to(g[:, None], y.shape).reshape(-1))
        loss = loss / max(len(multimodal), 1)

        p = jnp.concatenate(preds)
        t = jnp.concatenate(tgts)
        g = jnp.concatenate(grps)

        # 2. Ordering: CCC over the pooled batch
        loss = loss + lambda_ccc * (1.0 - ccc_jnp(p, t))

        # 3. Per-modality supervision: each head against its own role's target, normalised by M
        uni = 0.0
        for key, (a_i, b_i) in unimodal.items():
            role_i = key.split(".", 1)[0]
            if role_i not in targets:
                continue
            y_i = targets[role_i]
            uni = uni + nll_loss(a_i, b_i, y_i, weights=beta_nll_weights(a_i, b_i, beta_w))
        loss = loss + (lambda_uni / max(len(unimodal), 1)) * uni

        # 4. Fairness: squared CDD over ground-truth quantile bins
        loss = loss + lambda_fair * fairness_penalty(p, t, g, n_bins)

        return loss, updates["batch_stats"]

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, new_batch_stats), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_batch_stats)

    return state, loss


@jax.jit
def eval_step(state: TrainState, batch: dict[str, jax.Array], tau: float) -> dict[str, jax.Array]:
    stream_inputs = {k: v for k, v in batch.items() if not (k.endswith(".engagement") or k.endswith(".gender"))}
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    (multimodal, _), _ = state.apply_fn(variables, stream_inputs, tau=tau, rng=None, train=False, mutable=["batch_stats"])
    return {role: predictive_mean(a, b) for role, (a, b) in multimodal.items()}


def val_metrics(state: TrainState, cnfg: EngageNetConfig, tau: float) -> tuple[float, float | None, dict[str, float]]:
    """Return (CCC, CDD_G, CDD_L) over the val split, pooled across both roles.

    Gender is carried per participant in the batch, so a session whose two participants differ is no longer discarded.
    """
    all_preds, all_targets, all_genders, all_languages = [], [], [], []

    for batch in iter_batches(cnfg, split="val"):
        preds = eval_step(state, _drop_meta(batch), tau)  # dict{role: (B, L')}
        langs = np.array([_session_language(cnfg, n) for n in batch["session"]], dtype=object)

        for role in ROLES:
            key = f"{role}.engagement"
            if key not in batch or role not in preds:
                continue
            p = np.array(preds[role])  # (B, L')
            t = np.array(batch[key])  # (B, L')
            g = np.array(batch[f"{role}.gender"])  # (B,)
            L = p.shape[1]

            all_preds.append(p.reshape(-1))
            all_targets.append(t.reshape(-1))
            all_genders.append(np.repeat(g, L).astype(np.int8))
            all_languages.append(np.repeat(langs, L))

    preds = np.concatenate(all_preds)
    targs = np.concatenate(all_targets)
    v_ccc = ccc(preds, targs)

    genders = np.concatenate(all_genders)
    known = genders >= 0
    v_cdd_g = cdd_gender(preds[known], targs[known], genders[known]) if known.any() and len(np.unique(genders[known])) == 2 else None

    langs_all = np.concatenate(all_languages)
    v_cdd_l = cdd_language(preds, targs, langs_all) if len(np.unique(langs_all)) > 1 else {}

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
            state, loss = train_step(state, _drop_meta(batch), rng_step, tau, cnfg.beta_w, cnfg.lambda_ccc, cnfg.lambda_uni, cnfg.lambda_fair, cnfg.cdd_bins)
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
