from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from flax.training import train_state
from jax.scipy.special import betaln, digamma

from beta_head import nll_loss, predictive_mean, predictive_variance
from config import EngageNetConfig
from read_data import ROLES, log


# alpha_multi: (B,) ; beta_multi: (B,) ; unimodal: dict{str: (alpha, beta)} -> mask: (B,) boolean
def sample_filter(alpha_multi: jax.Array, beta_multi: jax.Array, unimodal: dict[str, tuple[jax.Array, jax.Array]], *, multi_pct: float = 0.3, uni_pct: float = 0.7) -> jax.Array:
    u_multi = predictive_variance(alpha_multi, beta_multi)  # (B,)

    u_unis = []
    for _key, (a, b) in unimodal.items():
        u_unis.append(predictive_variance(a, b))
    u_uni_avg = jnp.stack(u_unis, axis=0).mean(axis=0)  # (B,)

    multi_thresh = jnp.percentile(u_multi, multi_pct * 100)
    uni_thresh = jnp.percentile(u_uni_avg, uni_pct * 100)

    # Keep samples where multimodal is confident (low uncertainty) but unimodals disagree (high uncertainty)
    mask = (u_multi < multi_thresh) & (u_uni_avg > uni_thresh)
    return mask


def beta_kl(a1: jax.Array, b1: jax.Array, a2: jax.Array, b2: jax.Array) -> jax.Array:
    return (betaln(a2, b2) - betaln(a1, b1) + (a1 - a2) * digamma(a1) + (b1 - b2) * digamma(b1) + (a2 - a1 + b2 - b1) * digamma(a1 + b1))


# alpha_multi: (B,) ; beta_multi: (B,) ; unimodal: dict{str: (alpha, beta)} ; lam: float -> scalar
def tta_loss(alpha_multi: jax.Array, beta_multi: jax.Array, unimodal: dict[str, tuple[jax.Array, jax.Array]], *, lam: float = 1.0) -> jax.Array:
    # Mutual information sharing: KL(unimodal_i || multimodal) summed over modalities
    mis_loss = jnp.zeros(())
    for _key, (a_i, b_i) in unimodal.items():
        mis_loss = mis_loss + beta_kl(a_i, b_i, alpha_multi, beta_multi).mean()

    # multimodal mean as target for each unimodal head
    tgt = predictive_mean(alpha_multi, beta_multi)  # (B,)
    tgt_loss = jnp.zeros(())
    for _key, (a_i, b_i) in unimodal.items():
        tgt_loss = tgt_loss + nll_loss(a_i, b_i, tgt)

    return mis_loss + lam * tgt_loss


# Returns a param mask; True for surgical layers (BatchNorm, conv1 in InitEncoder, first Dense in inter-modal BiMamba). This method was completed using Claude Sonnet 4.6
def surgical_mask(params: dict) -> dict:
    def _mask(path: tuple, _leaf):
        path_str = "/".join(str(p) for p in path)
        if "batch_norm" in path_str:
            return True
        if "frontend" in path_str and "conv1" in path_str:
            return True
        if "inter_modal" in path_str and "cross_modal_bimamba" in path_str and "Dense_0" in path_str:
            return True
        return False

    return jax.tree_util.tree_map_with_path(_mask, params)


def tta_step(state, stream_inputs: dict[str, jax.Array], *, tau: float, rng: jax.Array, lam: float = 1.0):
    mask = surgical_mask(state.params)

    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        (alpha, beta, unimodal), updates = state.apply_fn(variables, stream_inputs, tau=tau, rng=rng, train=True, mutable=["batch_stats"])
        loss = tta_loss(alpha, beta, unimodal, lam=lam)
        return loss, (updates["batch_stats"], alpha, beta, unimodal)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (new_batch_stats, alpha, beta, unimodal)), grads = grad_fn(state.params)
    grads = jax.tree_util.tree_map(lambda g, m: g if m else jnp.zeros_like(g), grads, mask)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_batch_stats)

    return state, loss, alpha, beta, unimodal