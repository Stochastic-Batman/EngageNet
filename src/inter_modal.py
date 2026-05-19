from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from bimamba import BiMambaBlock


# scores: (B, M, M) -> doubly-stochastic matrix P: (B, M, M)
def sinkhorn(scores: jax.Array, n_iters: int = 10) -> jax.Array:
    P = jnp.exp(scores)
    for _ in range(n_iters):
        P = P / P.sum(axis=-1, keepdims=True)
        P = P / P.sum(axis=-2, keepdims=True)
    return P


class GumbelSinkhorn(nn.Module):
    M: int          # number of modalities
    d_key: int      # query/key projection dim
    n_iters: int    # Sinkhorn iterations

    # summaries: (B, M, C') ; tau: scalar temperature ; rng: PRNGKey or None -> P: (B, M, M) doubly-stochastic
    @nn.compact
    def __call__(self, summaries: jax.Array, *, tau: float = 1.0, rng: jax.Array | None = None) -> jax.Array:
        Q = nn.Dense(self.d_key, use_bias=False, name="query")(summaries)  # (B, M, d_key)
        K = nn.Dense(self.d_key, use_bias=False, name="key")(summaries)    # (B, M, d_key)

        # Score matrix: entry (i,j) = fitness of placing modality j at position i
        Z = jnp.matmul(Q, K.transpose(0, 2, 1))  # (B, M, M)

        # Gumbel perturbation for stochastic exploration during training
        if rng is not None:
            gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(rng, Z.shape, minval=1e-6, maxval=1.0 - 1e-6)))
            Z = Z + gumbel_noise

        P = sinkhorn(Z / tau, n_iters=self.n_iters)  # (B, M, M)
        return P


class InterModalBiMamba(nn.Module):
    D: int                  # BiMamba hidden dim
    N: int                  # SSM state dim
    D_C: int                # depthwise conv kernel size
    GS_dim: int = 64         # Gumbel-Sinkhorn projection dim
    n_iters: int = 10       # Sinkhorn iterations

    # hiddens: dict{str: (B, L', C')} ; tau: Gumbel temperature ; rng: PRNGKey or None -> H: (B, L', MC')
    @nn.compact
    def __call__(self, hiddens: dict[str, jax.Array], *, tau: float = 1.0, rng: jax.Array | None = None, train: bool = True) -> jax.Array:
        keys = sorted(hiddens.keys())
        M = len(keys)
        stacked = jnp.stack([hiddens[k] for k in keys], axis=1)  # (B, M, L', C')
        B, _, Lp, Cp = stacked.shape

        summaries = stacked.mean(axis=2)  # Temporal mean-pool each modality -> summaries (B, M, C')

        gs = GumbelSinkhorn(M=M, d_key=self.GS_dim, n_iters=self.n_iters, name="gumbel_sinkhorn")
        P = gs(summaries, tau=tau, rng=rng)  # (B, M, M)

        reordered = jnp.einsum("bij,bjlc->bilc", P, stacked)  # reordered[b,i] = sum_j P[b,i,j] * stacked[b,j] -> (B, M, L', C')

        fused = reordered.reshape(B, M, Lp, Cp).transpose(0, 2, 1, 3).reshape(B, Lp, M * Cp)

        block = BiMambaBlock(D=self.D, N=self.N, D_C=self.D_C, name="cross_modal_bimamba")
        H = block(fused, train=train)  # (B, L', MC')

        return H
