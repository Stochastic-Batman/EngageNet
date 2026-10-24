import jax
import jax.numpy as jnp
from jax import lax


# delta: (B, L, D); A: (D, N); B: (B, L, N)
def discretize(delta: jax.Array, A: jax.Array, B: jax.Array) -> tuple[jax.Array, jax.Array]:
    delta_reshaped = delta[..., None]
    A_reshaped = A[None, None, :, :]
    B_reshaped = B[:, :, None, :]
    
    A_bar = jnp.exp(delta_reshaped * A_reshaped)
    B_bar = delta_reshaped * B_reshaped
    
    return A_bar, B_bar


# a: (A_cumulative, x_cumulative) each (L, B, D, N); b: (A_t, b_t) each (L, B, D, N)
def _scan_fn(a: tuple[jax.Array, jax.Array], b: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
    # f_j(f_i(x)) = A_bar_j * (A_bar_i * x + b_i) + b_j = (A_bar_j * A_bar_i) * x + (A_bar_j * b_i + b_j)
    A1, b1 = a
    A2, b2 = b
    return A2 * A1, A2 * b1 + b2


# u: (B, L, D); delta: (B, L, D); A: (D, N); B: (B, L, N); C: (B, L, N)
def ssm(u: jax.Array, delta: jax.Array, A: jax.Array, B: jax.Array, C: jax.Array) -> jax.Array:
    A_bar, B_bar = discretize(delta, A, B)
    b = B_bar * u[..., None]

    A_bar_T = A_bar.transpose((1, 0, 2, 3))
    b_T = b.transpose((1, 0, 2, 3))
    
    _, xs = lax.associative_scan(_scan_fn, (A_bar_T, b_T), axis=0)
    xs = xs.transpose((1, 0, 2, 3))
    y = (C[:, :, None, :] * xs).sum(-1)
    
    return y


# a: (M_cumulative, x_cumulative) with M (L, B, D, N, 2, 2), x (L, B, D, N, 2); b: (M_t, b_t) likewise
def _coupled_scan_fn(a: tuple[jax.Array, jax.Array], b: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
    # f_j(f_i(s)) = M_j (M_i s + b_i) + b_j = (M_j M_i) s + (M_j b_i + b_j)
    M1, b1 = a
    M2, b2 = b
    return jnp.einsum("...ij,...jk->...ik", M2, M1), jnp.einsum("...ij,...j->...i", M2, b1) + b2


# Two coupled SSMs, one per person (leading axis 2 = role). u, delta, kappa: (2, B, L, D); A: (2, D, N); B, C: (2, B, L, N)
# h^r_t = a^r_t h^r_{t-1} + (1 - a^r_t) kappa^r_t h^{r'}_{t-1} + B_bar^r_t u^r_t, with r' the partner.
# Each row of the 2x2 transition sums to a + (1 - a) kappa <= 1, so the coupled state stays bounded; kappa = 0 gives two independent SSMs.
def coupled_ssm(u: jax.Array, delta: jax.Array, A: jax.Array, B: jax.Array, C: jax.Array, kappa: jax.Array) -> jax.Array:
    A_bar = jnp.exp(delta[..., None] * A[:, None, None])  # (2, B, L, D, N)
    b = delta[..., None] * B[:, :, :, None, :] * u[..., None]  # (2, B, L, D, N)
    g = (1.0 - A_bar) * kappa[..., None]  # (2, B, L, D, N): weight of the partner's previous state

    M = jnp.stack([jnp.stack([A_bar[0], g[0]], axis=-1), jnp.stack([g[1], A_bar[1]], axis=-1)], axis=-2)  # (B, L, D, N, 2, 2)
    b = jnp.moveaxis(b, 0, -1)  # (B, L, D, N, 2)

    _, xs = lax.associative_scan(_coupled_scan_fn, (M.transpose((1, 0, 2, 3, 4, 5)), b.transpose((1, 0, 2, 3, 4))), axis=0)
    xs = jnp.moveaxis(xs.transpose((1, 0, 2, 3, 4)), -1, 0)  # (2, B, L, D, N)
    y = (C[:, :, :, None, :] * xs).sum(-1)  # (2, B, L, D)

    return y
