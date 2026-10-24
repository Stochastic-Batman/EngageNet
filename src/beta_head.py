from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from jax.scipy.stats import beta as beta_dist
from read_data import ROLES


# alpha: (B, ...) ; beta: (B, ...) -> (B, ...)
def predictive_mean(alpha: jax.Array, beta: jax.Array) -> jax.Array:
    return alpha / (alpha + beta)


# alpha: (B, ...) ; beta: (B, ...) -> (B, ...)
def predictive_variance(alpha: jax.Array, beta: jax.Array) -> jax.Array:
    apb = alpha + beta
    return (alpha * beta) / (apb * apb * (apb + 1))


# alpha: (B, ...) ; beta: (B, ...) ; targets: (B, ...) ; weights: (B, ...) or None -> scalar
def nll_loss(alpha: jax.Array, beta: jax.Array, targets: jax.Array, weights: jax.Array | None = None) -> jax.Array:
    targets = jnp.clip(targets, 1e-6, 1.0 - 1e-6)
    ll = beta_dist.logpdf(targets, alpha, beta)
    if weights is None:
        return -ll.mean()
    return -(weights * ll).sum() / (weights.sum() + 1e-8)


# alpha: (B, ...) ; beta: (B, ...) -> (B, ...)
def beta_nll_weights(alpha: jax.Array, beta: jax.Array, beta_w: float) -> jax.Array:
    """Detached kappa^{-beta_w}; cancels the precision factor in d(-log p)/d(mu).

    beta_w = 0 -> plain NLL; beta_w = 1 -> update independent of confidence.
    Detached so the model cannot lower the loss by inflating kappa.
    """
    kappa = jax.lax.stop_gradient(alpha + beta)
    return kappa ** (-beta_w)


class BetaHead(nn.Module):
    hidden_dim: int = 128

    # y: (B, ..., D) -> (alpha: (B, ...), beta: (B, ...))
    @nn.compact
    def __call__(self, y: jax.Array) -> tuple[jax.Array, jax.Array]:
        y_proj = nn.silu(nn.Dense(features=self.hidden_dim)(y))

        alpha_logit = nn.Dense(features=1)(y_proj)
        beta_logit = nn.Dense(features=1)(y_proj)

        alpha, beta = jax.nn.softplus(alpha_logit) + 1, jax.nn.softplus(beta_logit) + 1

        return alpha.squeeze(-1), beta.squeeze(-1)


class MultiHeadBeta(nn.Module):
    hidden_dim: int = 128

    # fused: (B, L', MC') ; per_modality: dict{"{role}.{feat}": (B, L', C')}
    # -> (multimodal: dict{role: (alpha, beta)}, unimodal: dict{"{role}.{feat}": (alpha, beta)})
    @nn.compact
    def __call__(self, fused: jax.Array, per_modality: dict[str, jax.Array]) -> tuple[dict[str, tuple[jax.Array, jax.Array]], dict[str, tuple[jax.Array, jax.Array]]]:
        # One multimodal head per role, both reading the same fused representation
        multimodal = {role: BetaHead(hidden_dim=self.hidden_dim, name=f"multi_head_{role}")(fused) for role in ROLES}

        feats = sorted(set(key.split(".", 1)[1] for key in per_modality))
        unimodal: dict[str, tuple[jax.Array, jax.Array]] = {}

        for feat in feats:
            head = BetaHead(hidden_dim=self.hidden_dim, name=f"uni_head_{feat.replace('.', '_')}")
            for key in sorted(per_modality):
                if key.split(".", 1)[1] == feat:
                    unimodal[key] = head(per_modality[key])

        return multimodal, unimodal
