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
    position_queries: bool = False  # rows of Z from one learned query per POSITION (Z_ij = p_i . k_j) instead of per stream
    scale_scores: bool = False      # Z / sqrt(d_key), as in scaled dot-product attention

    # summaries: (B, M, C') ; tau: scalar temperature ; rng: PRNGKey or None -> P: (B, M, M) doubly-stochastic
    @nn.compact
    def __call__(self, summaries: jax.Array, *, tau: float = 1.0, rng: jax.Array | None = None) -> jax.Array:
        K = nn.Dense(self.d_key, use_bias=False, name="key")(summaries)    # (B, M, d_key)
        if self.position_queries:
            pos = self.param("position_queries", nn.initializers.normal(stddev=1.0), (self.M, self.d_key))  # (M, d_key)
            Q = jnp.broadcast_to(pos, K.shape)  # (B, M, d_key): row i scores how well each stream fits position i
        else:
            Q = nn.Dense(self.d_key, use_bias=False, name="query")(summaries)  # (B, M, d_key): row i belongs to stream i

        # Score matrix: entry (i,j) = fitness of placing modality j at position i
        Z = jnp.matmul(Q, K.transpose(0, 2, 1))  # (B, M, M)
        if self.scale_scores:
            Z = Z / jnp.sqrt(jnp.asarray(self.d_key, Z.dtype))
        self.sow("intermediates", "Z", Z)  # score magnitudes, for diagnostics only

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
    if name.startswith("rand") and name[4:].isdigit():  # rand<k>: drawn once from seed k, independent of the training seed
        return np.random.default_rng(int(name[4:])).permutation(M)
    raise ValueError(f"unknown fixed order {name!r}; expected one of {FIXED_ORDERS} or rand<k>")


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
    position_queries: bool = False
    scale_scores: bool = False
    unsort: bool = False            # map fused slots back to stream order with P^T, so the heads see a fixed layout
    probe_order: str | None = None  # inference-only probe of a fixed-order model: scan in this order, then map back to the trained layout
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
            gs = GumbelSinkhorn(M=M, d_key=self.GS_dim, n_iters=self.n_iters, position_queries=self.position_queries, scale_scores=self.scale_scores, name="gumbel_sinkhorn")
            P = gs(summaries, tau=tau, rng=rng)  # (B, M, M)
        else:
            P_fixed = jnp.asarray(permutation_matrix(fixed_permutation(keys, self.fixed_order)))  # (M, M)
            P = jnp.broadcast_to(P_fixed, (B, M, M)).astype(stacked.dtype)  # (B, M, M)
        self.sow("intermediates", "P", P)  # read out with mutable=["intermediates"]; a no-op otherwise

        P_scan = P
        if self.probe_order is not None:
            if self.fixed_order is None:
                raise ValueError("probe_order needs a fixed-order model (fixed_order set)")
            P_scan = jnp.broadcast_to(jnp.asarray(permutation_matrix(fixed_permutation(keys, self.probe_order))), (B, M, M)).astype(stacked.dtype)

        reordered = jnp.einsum("bij,bjlc->bilc", P_scan, stacked)  # reordered[b,i] = sum_j P[b,i,j] * stacked[b,j] -> (B, M, L', C')

        # Fuse along the MODALITY axis: sequence length M, feature width C'.
        # Folding time into the batch applies the same block, with shared weights, at every step.
        seq = reordered.transpose(0, 2, 1, 3).reshape(B * Lp, M, Cp)  # (B * L', M, C')

        block = BiMambaBlock(D=self.D, N=self.N, D_C=self.D_C, dt_min=self.dt_min, dt_max=self.dt_max, record_decay=True, name="cross_modal_bimamba")
        
        fused = block(seq, train=train)  # (B * L', M, C')

        if self.probe_order is not None and not self.unsort:
            # Slot i of the probe scan holds stream probe[i]; the heads were trained on slot k = stream trained[k].
            # P P_scan^T maps probe slots back to the trained layout, so only the scan order differs.
            fused = jnp.einsum("bik,bkj,bljc->blic", P, P_scan.transpose(0, 2, 1), fused.reshape(B, Lp, M, Cp)).reshape(B * Lp, M, Cp)
            P_scan = P  # nothing left to unsort

        if self.unsort:
            # Slot i holds sum_j P_ij u_j; stream j gets back sum_i P_ij slot_i (= P^T, the inverse of a permutation),
            # so the heads always read the streams in the same layout whatever order the scan used
            fused = jnp.einsum("bij,blic->bljc", P_scan, fused.reshape(B, Lp, M, Cp)).reshape(B * Lp, M, Cp)

        H = fused.reshape(B, Lp, M * Cp)  # (B, L', MC')

        return H
