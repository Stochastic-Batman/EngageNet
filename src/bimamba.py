from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from read_data import ROLES
from ssm import ssm



def _a_log_init(key: jax.Array, shape: tuple[int, ...], dtype=jnp.float32) -> jax.Array:
    """S4D-Real initialisation: a_{d,n} = -(n+1), stored as log|A|.

    Gives every channel a geometric ladder of decay rates instead of a single shared one,
    so the N state coordinates cover a range of timescales rather than N copies of one.
    """
    _, N = shape
    n = jnp.arange(1, N + 1, dtype=dtype)  # 1, 2, ..., N
    return jnp.broadcast_to(jnp.log(n), shape).astype(dtype)


def _s_delta_init(dt_min: float, dt_max: float):
    """Initialise s_delta so that softplus(s_delta) is log-uniform on [dt_min, dt_max].

    tau_eff = 1 / (delta * |A|) steps, so this sets the block's starting receptive field.
    Zeros would give delta = softplus(0) = ln 2, i.e. tau_eff ~ 1.4 steps (~58 ms at 25 Hz).
    """
    log_lo, log_hi = jnp.log(dt_min), jnp.log(dt_max)

    def init(key: jax.Array, shape: tuple[int, ...], dtype=jnp.float32) -> jax.Array:
        u = jax.random.uniform(key, shape, dtype=dtype)
        dt = jnp.exp(u * (log_hi - log_lo) + log_lo)
        return jnp.log(jnp.expm1(dt)).astype(dtype)  # softplus^{-1}(dt)

    return init


class BiMambaBlock(nn.Module):
    D: int  # D
    N: int  # N
    D_C:  int  # depthwise convolution kernel size
    pre_norm: bool = True  # LayerNorm on the residual stream
    dt_min: float = 1e-3  # smallest delta at init -> longest receptive field
    dt_max: float = 1e-1  # largest delta at init -> shortest receptive field

    # h: (B, L', D) -> (B, L', D)
    @nn.compact
    def __call__(self, h: jax.Array, *, train: bool = True) -> jax.Array:
        D, N = self.D, self.N

        h_in = nn.LayerNorm(name="pre_norm")(h) if self.pre_norm else h
        g = nn.silu(nn.Dense(D)(h_in))  # (B, L', D)
        x = nn.Dense(D)(h_in)  # (B, L', D)
        
        x_fwd = nn.silu(nn.Conv(features=D, kernel_size=(self.D_C,), padding="SAME", feature_group_count=D)(x))
        # A stored in log-space so exp keeps it negative -> A_bar = exp(delta*A) in (0,1), stable
        A_fwd = -jnp.exp(self.param("A_log_fwd", _a_log_init, (D, N)))
        delta_fwd = nn.softplus(self.param("s_delta_fwd", _s_delta_init(self.dt_min, self.dt_max), (D,)) + nn.Dense(D)(x_fwd))
        B_fwd = nn.Dense(N, use_bias=False)(x_fwd)  # (B, L', N)
        C_fwd = nn.Dense(N, use_bias=False)(x_fwd)  # (B, L', N)
        h_fwd = g * ssm(x_fwd, delta_fwd, A_fwd, B_fwd, C_fwd)  # (B, L', D)

        x_bwd = nn.silu(nn.Conv(features=D, kernel_size=(self.D_C,), padding="SAME", feature_group_count=D)(jnp.flip(x, axis=1)))
        A_bwd = -jnp.exp(self.param("A_log_bwd", _a_log_init, (D, N)))
        delta_bwd = nn.softplus(self.param("s_delta_bwd", _s_delta_init(self.dt_min, self.dt_max), (D,)) + nn.Dense(D)(x_bwd))
        B_bwd = nn.Dense(N, use_bias=False)(x_bwd)  # (B, L', N)
        C_bwd = nn.Dense(N, use_bias=False)(x_bwd)  # (B, L', N)
        h_bwd = g * jnp.flip(ssm(x_bwd, delta_bwd, A_bwd, B_bwd, C_bwd), axis=1)  # (B, L', D)

        u = h + nn.Dense(D)((h_fwd + h_bwd) / 2)  # (B, L', D)

        return u


class IntraModalBiMamba(nn.Module):
    D: int
    N: int
    D_C:  int

    # hiddens: dict{str: (B, L', C')} -> dict{str: (B, L', C')}
    @nn.compact
    def __call__(self, hiddens: dict[str, jax.Array], *, train: bool = True) -> dict[str, jax.Array]:
        # One BiMambaBlock per feature type, shared across roles - same weight-sharing logic as ModalityFrontend
        # sorted(set(...)) gives stable parameter creation order.
        feats = sorted(set(key.split(".", 1)[1] for key in hiddens))
        u: dict[str, jax.Array] = {}

        for feat in feats:
            block = BiMambaBlock(D=self.D, N=self.N, D_C=self.D_C, name=f"bimamba_{feat.replace('.', '_')}")
            for role in ROLES:
                key = f"{role}.{feat}"
                if key not in hiddens:
                    continue
                u[key] = block(hiddens[key], train=train)  # (B, L', C')

        return u
