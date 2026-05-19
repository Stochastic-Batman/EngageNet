from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from jax.scipy.stats import beta as beta_dist


# alpha: (B,) ; beta: (B,) -> (B,)
def predictive_mean(alpha: jax.Array, beta: jax.Array) -> jax.Array:
    return alpha / (alpha + beta)


# alpha: (B,) ; beta: (B,) -> (B,)
def predictive_variance(alpha: jax.Array, beta: jax.Array) -> jax.Array:
    apb = alpha + beta
    return (alpha * beta) / (apb * apb * (apb + 1))


# alpha: (B,) ; beta: (B,) ; targets: (B,) -> scalar
def nll_loss(alpha: jax.Array, beta: jax.Array, targets: jax.Array) -> jax.Array:
    targets = jnp.clip(targets, 1e-6, 1.0 - 1e-6)
    return -beta_dist.logpdf(targets, alpha, beta).mean()


class BetaHead(nn.Module):
    hidden_dim: int = 128

    # y: (B, D) -> (alpha: (B,), beta: (B,))
    @nn.compact
    def __call__(self, y: jax.Array) -> tuple[jax.Array, jax.Array]:
        y_proj = nn.silu(nn.Dense(features=self.hidden_dim)(y))

        alpha_logit = nn.Dense(features=1)(y_proj)
        beta_logit = nn.Dense(features=1)(y_proj)

        alpha, beta = jax.nn.softplus(alpha_logit) + 1, jax.nn.softplus(beta_logit) + 1

        return alpha.squeeze(-1), beta.squeeze(-1)


class MultiHeadBeta(nn.Module):
    hidden_dim: int = 128

    # fused: (B, L', MC') ; per_modality: dict{str: (B, L', C')} -> (multimodal_alpha, multimodal_beta, dict{str: (alpha, beta)})
    @nn.compact
    def __call__(self, fused: jax.Array, per_modality: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array, dict[str, tuple[jax.Array, jax.Array]]]:
        multimodal_alpha, multimodal_beta = BetaHead(hidden_dim=self.hidden_dim, name="multi_head")(fused.mean(axis=1))

        feats = sorted(set(key.split(".", 1)[1] for key in per_modality))
        unimodal: dict[str, tuple[jax.Array, jax.Array]] = {}

        for feat in feats:
            head = BetaHead(hidden_dim=self.hidden_dim, name=f"uni_head_{feat.replace('.', '_')}")
            for key in sorted(per_modality):
                if key.split(".", 1)[1] == feat:
                    unimodal[key] = head(per_modality[key].mean(axis=1))

        return multimodal_alpha, multimodal_beta, unimodal

