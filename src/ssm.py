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
