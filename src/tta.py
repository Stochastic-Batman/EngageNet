from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import optax

from jax.scipy.special import betaln, digamma

from beta_head import nll_loss, predictive_mean, predictive_variance
from read_data import ROLES, log


# u_multi: (B,) ; u_uni: (B,) -> mask: (B,) boolean
def select_windows(u_multi: jax.Array, u_uni: jax.Array, *, multi_pct: float = 0.3, uni_pct: float = 0.7) -> jax.Array:
    """Keep windows where the fused head is confident but the single-stream heads are not.

    Quantiles are taken over the batch, so the batch must hold several windows;
    with one window q(u) == u and the strict inequality can never hold.
    """
    multi_thresh = jnp.percentile(u_multi, multi_pct * 100)
    uni_thresh = jnp.percentile(u_uni, uni_pct * 100)
    return (u_multi < multi_thresh) & (u_uni > uni_thresh)


# multimodal: dict{role: (alpha, beta)} each (B, L') -> (B,)
def window_uncertainty(heads: dict[str, tuple[jax.Array, jax.Array]]) -> jax.Array:
    """Mean per-frame Beta variance, averaged over time and over heads -> one scalar per window."""
    return jnp.stack([predictive_variance(a, b).mean(axis=-1) for a, b in heads.values()], axis=0).mean(axis=0)


def beta_kl(a1: jax.Array, b1: jax.Array, a2: jax.Array, b2: jax.Array) -> jax.Array:
    return (betaln(a2, b2) - betaln(a1, b1) + (a1 - a2) * digamma(a1) + (b1 - b2) * digamma(b1) + (a2 - a1 + b2 - b1) * digamma(a1 + b1))


# alpha_multi, beta_multi: (B, L') ; unimodal: dict{str: (alpha, beta)} ; weights: (B,) -> scalar
def tta_loss(alpha_multi: jax.Array, beta_multi: jax.Array, unimodal: dict[str, tuple[jax.Array, jax.Array]], *, weights: jax.Array, lam: float = 1.0) -> jax.Array:
    """Loss over the SELECTED windows only (weights = 0/1 mask broadcast over time)."""
    w = jnp.broadcast_to(weights[:, None], alpha_multi.shape).astype(alpha_multi.dtype)
    denom = w.sum() + 1e-8

    # Mutual information sharing: KL(unimodal_i || multimodal), gradient flows both ways as in the paper
    mis_loss = jnp.zeros(())
    for _key, (a_i, b_i) in unimodal.items():
        mis_loss = mis_loss + (w * beta_kl(a_i, b_i, alpha_multi, beta_multi)).sum() / denom

    # Multimodal mean is a pseudo-label: a fixed target, so the teacher is not pulled towards the students
    tgt = jax.lax.stop_gradient(predictive_mean(alpha_multi, beta_multi))
    tgt_loss = jnp.zeros(())
    for _key, (a_i, b_i) in unimodal.items():
        tgt_loss = tgt_loss + nll_loss(a_i, b_i, tgt, weights=w)

    return mis_loss + lam * tgt_loss


# True for surgical layers (BatchNorm, conv1 in InitEncoder, first Dense in the cross-modal BiMamba)
def surgical_mask(params: dict) -> dict:
    def _mask(path: tuple, _leaf):
        path_str = "/".join(str(getattr(p, "key", p)) for p in path).lower()
        if "batch_norm" in path_str or "batchnorm" in path_str:
            return True
        if "frontend" in path_str and "conv1" in path_str:
            return True
        if "inter_modal" in path_str and "cross_modal_bimamba" in path_str and "dense_0" in path_str:
            return True
        return False

    return jax.tree_util.tree_map_with_path(_mask, params)


def make_tta_tx(params: dict, lr: float) -> optax.GradientTransformation:
    """Adam on the surgical layers, exact zero update everywhere else (no weight decay leaking into frozen weights)."""
    labels = jax.tree_util.tree_map(lambda m: "adapt" if m else "frozen", surgical_mask(params))
    return optax.multi_transform({"adapt": optax.adam(lr), "frozen": optax.set_to_zero()}, labels)


def count_adapted(params: dict) -> int:
    mask = surgical_mask(params)
    return int(sum(p.size for p, m in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(mask)) if m))


# state: TrainState whose tx comes from make_tta_tx ; stream_inputs: dict{str: (B, C_i, L)} ; mask: (B,) bool
@partial(jax.jit, static_argnames=("lam",))
def tta_step(state, stream_inputs: dict[str, jax.Array], mask: jax.Array, tau: float, lam: float = 1.0):
    weights = mask.astype(jnp.float32)

    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        # rng=None: no Gumbel noise at test time; train=True only switches BatchNorm to batch statistics
        (multimodal, unimodal), updates = state.apply_fn(variables, stream_inputs, tau=tau, rng=None, train=True, mutable=["batch_stats"])

        # Each role's multimodal head teaches only the unimodal heads reading that role's streams
        loss = jnp.zeros(())
        for role, (a, b) in multimodal.items():
            uni_role = {k: v for k, v in unimodal.items() if k.split(".", 1)[0] == role}
            loss = loss + tta_loss(a, b, uni_role, weights=weights, lam=lam)
        loss = loss / max(len(multimodal), 1)

        return loss, updates["batch_stats"]

    (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_batch_stats)

    return state, loss
