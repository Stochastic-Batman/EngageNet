from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from read_data import ROLES
from ssm import coupled_ssm, ssm



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
    record_decay: bool = False  # sow statistics of A_bar = exp(delta*A) (diagnostics, read with mutable=["intermediates"])

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

        if self.record_decay:
            L = h.shape[1]
            for name, delta, A in (("fwd", delta_fwd, A_fwd), ("bwd", delta_bwd, A_bwd)):
                a_bar = jnp.exp(delta[..., None] * A)  # (B, L', D, N): fraction of the state kept per step
                self.sow("intermediates", f"abar_hist_{name}", jnp.histogram(a_bar, bins=20, range=(0.0, 1.0))[0])
                self.sow("intermediates", f"abar_reach_{name}", jnp.mean(a_bar ** (L - 1)))  # share kept over the whole sequence

        u = h + nn.Dense(D)((h_fwd + h_bwd) / 2)  # (B, L', D)

        return u


class DyadicBiMambaBlock(nn.Module):
    """BiMamba over time for the two people of a session, with one SSM state per person and selective coupling between them.

    Same layout as BiMambaBlock, with role-specific parameters. In each direction the state of person r takes in a share
    (1 - A_bar) * kappa of the partner's previous state; kappa in (0, 1) is input-dependent per channel and computed from both
    people's inputs. coupled=False fixes kappa = 0, i.e. two independent per-person temporal BiMambas (the ablation).
    """
    D: int
    N: int
    D_C: int
    coupled: bool = True
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    kappa_bias: float = -1.0  # sigmoid(-1) ~ 0.27 at initialisation

    # h: (2, B, L', D), role axis in ROLES order -> (2, B, L', D)
    @nn.compact
    def __call__(self, h: jax.Array, *, train: bool = True) -> jax.Array:
        D, N = self.D, self.N
        R = range(len(ROLES))

        h_in = [nn.LayerNorm(name=f"pre_norm_{ROLES[r]}")(h[r]) for r in R]
        g = [nn.silu(nn.Dense(D, name=f"gate_{ROLES[r]}")(h_in[r])) for r in R]  # (B, L', D) each
        x = [nn.Dense(D, name=f"in_{ROLES[r]}")(h_in[r]) for r in R]

        ys = {}
        for direction in ("fwd", "bwd"):
            xd = [x[r] if direction == "fwd" else jnp.flip(x[r], axis=1) for r in R]
            xc = [nn.silu(nn.Conv(features=D, kernel_size=(self.D_C,), padding="SAME", feature_group_count=D, name=f"conv_{direction}_{ROLES[r]}")(xd[r])) for r in R]
            A = jnp.stack([-jnp.exp(self.param(f"A_log_{direction}_{ROLES[r]}", _a_log_init, (D, N))) for r in R])  # (2, D, N)
            delta = jnp.stack([nn.softplus(self.param(f"s_delta_{direction}_{ROLES[r]}", _s_delta_init(self.dt_min, self.dt_max), (D,)) + nn.Dense(D, name=f"delta_{direction}_{ROLES[r]}")(xc[r])) for r in R])
            Bm = jnp.stack([nn.Dense(N, use_bias=False, name=f"B_{direction}_{ROLES[r]}")(xc[r]) for r in R])  # (2, B, L', N)
            Cm = jnp.stack([nn.Dense(N, use_bias=False, name=f"C_{direction}_{ROLES[r]}")(xc[r]) for r in R])
            if self.coupled:
                # Coupling of person r depends on both people's current input: when does the partner's state matter for r?
                kappa = jnp.stack([nn.sigmoid(nn.Dense(D, bias_init=nn.initializers.constant(self.kappa_bias), name=f"kappa_{direction}_{ROLES[r]}")(jnp.concatenate([xc[r], xc[1 - r]], axis=-1))) for r in R])
                self.sow("intermediates", f"kappa_mean_{direction}", kappa.mean(axis=(1, 2, 3)))  # (2,) per role
            else:
                kappa = jnp.zeros_like(delta)
            y = coupled_ssm(jnp.stack(xc), delta, A, Bm, Cm, kappa)  # (2, B, L', D)
            ys[direction] = y if direction == "fwd" else jnp.flip(y, axis=2)

        return jnp.stack([h[r] + nn.Dense(D, name=f"out_{ROLES[r]}")(g[r] * (ys["fwd"][r] + ys["bwd"][r]) / 2) for r in R])


class IntraModalBiMamba(nn.Module):
    D: int
    N: int
    D_C:  int
    dt_min: float = 1e-3
    dt_max: float = 1e-1

    # hiddens: dict{str: (B, L', C')} -> dict{str: (B, L', C')}
    @nn.compact
    def __call__(self, hiddens: dict[str, jax.Array], *, train: bool = True) -> dict[str, jax.Array]:
        # One BiMambaBlock per feature type, shared across roles - same weight-sharing logic as ModalityFrontend
        # sorted(set(...)) gives stable parameter creation order.
        feats = sorted(set(key.split(".", 1)[1] for key in hiddens))
        u: dict[str, jax.Array] = {}

        for feat in feats:
            block = BiMambaBlock(D=self.D, N=self.N, D_C=self.D_C, dt_min=self.dt_min, dt_max=self.dt_max, name=f"bimamba_{feat.replace('.', '_')}")
            for role in ROLES:
                key = f"{role}.{feat}"
                if key not in hiddens:
                    continue
                u[key] = block(hiddens[key], train=train)  # (B, L', C')

        return u
