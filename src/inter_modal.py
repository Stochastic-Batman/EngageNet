from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from jax.scipy.special import logsumexp

from bimamba import BiMambaBlock


# scores: (B, M, M) -> doubly-stochastic matrix P: (B, M, M)
def sinkhorn(scores: jax.Array, n_iters: int = 10) -> jax.Array:
    """Sinkhorn normalisation carried out in log-space.

    Normalising in probability space needs exp(scores) up front, which overflows
    to +inf once the scores get large (real feature magnitudes reach 1e4-1e5);
    the subsequent inf/inf then poisons the whole model with NaN. Alternating
    log-domain normalisation is mathematically identical but numerically stable.
    """
    log_P = scores
    for _ in range(n_iters):
        log_P = log_P - logsumexp(log_P, axis=-1, keepdims=True)
        log_P = log_P - logsumexp(log_P, axis=-2, keepdims=True)
    return jnp.exp(log_P)


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


FIXED_ORDERS = ("blocked", "interleaved", "rand0", "rand1")


# keys: sorted stream names "{role}.{feat}" ; name: one of FIXED_ORDERS -> perm with perm[i] = index of the stream placed at position i
def fixed_permutation(keys: list[str], name: str) -> np.ndarray:
    M = len(keys)
    if name == "blocked":  # sorted order = all expert streams, then all novice streams
        return np.arange(M)
    if name == "interleaved":  # expert and novice stream of each feature type side by side
        feats = sorted({k.split(".", 1)[1] for k in keys})
        roles = sorted({k.split(".", 1)[0] for k in keys})
        return np.array([keys.index(f"{r}.{f}") for f in feats for r in roles if f"{r}.{f}" in keys])
    if name in ("rand0", "rand1"):  # drawn once from a fixed seed, independent of the training seed
        return np.random.default_rng(int(name[-1])).permutation(M)
    raise ValueError(f"unknown fixed order {name!r}; expected one of {FIXED_ORDERS}")


# perm: (M,) -> P: (M, M) permutation matrix with P[i, perm[i]] = 1
def permutation_matrix(perm: np.ndarray) -> np.ndarray:
    M = len(perm)
    P = np.zeros((M, M), dtype=np.float32)
    P[np.arange(M), perm] = 1.0
    return P


class InterModalBiMamba(nn.Module):
    D: int                  # BiMamba hidden dim
    N: int                  # SSM state dim
    D_C: int                # depthwise conv kernel size
    GS_dim: int = 64         # Gumbel-Sinkhorn projection dim
    n_iters: int = 10       # Sinkhorn iterations
    fixed_order: str | None = None  # None = learned (Gumbel-Sinkhorn); else a name from FIXED_ORDERS -> constant P, no meta-network
    dt_min: float = 1e-3
    dt_max: float = 1e-1

    # hiddens: dict{str: (B, L', C')} ; tau: Gumbel temperature ; rng: PRNGKey or None -> H: (B, L', MC')
    @nn.compact
    def __call__(self, hiddens: dict[str, jax.Array], *, tau: float = 1.0, rng: jax.Array | None = None, train: bool = True) -> jax.Array:
        keys = sorted(hiddens.keys())
        M = len(keys)
        stacked = jnp.stack([hiddens[k] for k in keys], axis=1)  # (B, M, L', C')
        B, _, Lp, Cp = stacked.shape

        summaries = stacked.mean(axis=2)  # Temporal mean-pool each modality -> summaries (B, M, C')

        if self.fixed_order is None:
            gs = GumbelSinkhorn(M=M, d_key=self.GS_dim, n_iters=self.n_iters, name="gumbel_sinkhorn")
            P = gs(summaries, tau=tau, rng=rng)  # (B, M, M)
        else:
            P_fixed = jnp.asarray(permutation_matrix(fixed_permutation(keys, self.fixed_order)))  # (M, M)
            P = jnp.broadcast_to(P_fixed, (B, M, M)).astype(stacked.dtype)  # (B, M, M)
        self.sow("intermediates", "P", P)  # read out with mutable=["intermediates"]; a no-op otherwise

        reordered = jnp.einsum("bij,bjlc->bilc", P, stacked)  # reordered[b,i] = sum_j P[b,i,j] * stacked[b,j] -> (B, M, L', C')

        # Fuse along the MODALITY axis: sequence length M, feature width C'.
        # Folding time into the batch applies the same block, with shared weights, at every step.
        seq = reordered.transpose(0, 2, 1, 3).reshape(B * Lp, M, Cp)  # (B * L', M, C')

        block = BiMambaBlock(D=self.D, N=self.N, D_C=self.D_C, dt_min=self.dt_min, dt_max=self.dt_max, name="cross_modal_bimamba")
        
        fused = block(seq, train=train)  # (B * L', M, C')

        H = fused.reshape(B, Lp, M * Cp)  # (B, L', MC')

        return H
