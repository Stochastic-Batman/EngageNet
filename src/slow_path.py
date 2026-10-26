"""SlowPath - the second, slow rate of the two-rate model.

The window-level model sees 10 s at 25 Hz. This path sees the WHOLE session at 1/slow_bin of the frame rate (2 s bins by
default) through the same three stages - per-feature projection, intra-modal BiMamba over time, inter-modal BiMamba over the
M streams, then one BiMamba over time on the fused state - and returns its output interpolated at the frames of the window,
so the heads of the fast path can read slow, session-level context next to H.

Inputs (from dataset.pad_slow): "slow.{role}.{feat}" (B, S, C_i) bin means of the standardised streams, "slow_mask" (B, S)
with 1 for real bins, "window_start" (B,) first frame of the window in the session.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from bimamba import BiMambaBlock, IntraModalBiMamba
from config import EngageNetConfig
from inter_modal import InterModalBiMamba
from read_data import ROLES


# s: (B, S, C) ; t: (B, L) frame indices ; n_bins: (B,) real bins per row ; bin_len: frames per bin -> (B, L, C)
def interpolate_bins(s: jax.Array, t: jax.Array, n_bins: jax.Array, bin_len: int) -> jax.Array:
    """Linear interpolation between bin centres (bin k covers frames [k*bin_len, (k+1)*bin_len)); constant beyond the ends."""
    pos = (t.astype(s.dtype) - (bin_len - 1) / 2) / bin_len  # (B, L), bin coordinate of each frame
    i0 = jnp.floor(pos)
    w = (pos - i0)[..., None]  # (B, L, 1)
    hi = (n_bins - 1)[:, None]  # (B, 1)
    i0c = jnp.clip(i0.astype(jnp.int32), 0, hi)
    i1c = jnp.clip(i0.astype(jnp.int32) + 1, 0, hi)
    s0 = jnp.take_along_axis(s, i0c[..., None], axis=1)  # (B, L, C)
    s1 = jnp.take_along_axis(s, i1c[..., None], axis=1)
    return (1 - w) * s0 + w * s1


class SlowPath(nn.Module):
    cnfg: EngageNetConfig

    # inputs: dict with the slow keys above ; n_frames: L' of the window -> (B, L', C_s)
    @nn.compact
    def __call__(self, inputs: dict[str, jax.Array], *, n_frames: int, train: bool = True) -> jax.Array:
        cnfg = self.cnfg
        C = cnfg.slow_dim
        if any(cnfg.stride(f) != 1 for f in cnfg.modality_names):
            raise ValueError("slow_path assumes stride-1 encoders (window frame k = session frame start + k)")
        mask = inputs["slow_mask"]  # (B, S)
        m = mask[..., None]  # (B, S, 1)

        def session_centre(x: jax.Array) -> jax.Array:
            # Subtract the mean over the real bins of each session: a constant per session cannot pass
            return (x - (x * m).sum(1, keepdims=True) / jnp.maximum(m.sum(1, keepdims=True), 1.0)) * m

        # Per-feature projection, shared by both roles (as in ModalityFrontend; no BatchNorm, which padding would bias)
        hiddens: dict[str, jax.Array] = {}
        for feat in cnfg.modality_names:
            proj = nn.Dense(C, name=f"proj_{feat.replace('.', '_')}")
            for role in ROLES:
                key = f"{role}.{feat}"
                if f"slow.{key}" in inputs:
                    x = inputs[f"slow.{key}"]  # (B, S, C_i)
                    hiddens[key] = proj(session_centre(x) if cnfg.slow_center else x)  # (B, S, C_s)

        u = IntraModalBiMamba(D=C, N=cnfg.ssm_state_dim, D_C=cnfg.conv_kernel, dt_min=cnfg.dt_min, dt_max=cnfg.dt_max, name="intra_modal")(hiddens, train=train, mask=mask)
        # Order of the streams has no measurable effect (Section 4.8), so the slow path uses a fixed one
        H = InterModalBiMamba(D=C, N=cnfg.ssm_state_dim, D_C=cnfg.cross_modal_conv_kernel, fixed_order="blocked", dt_min=cnfg.dt_min, dt_max=cnfg.dt_max, name="inter_modal")(u, train=train)  # (B, S, MC_s)
        s = BiMambaBlock(D=C, N=cnfg.ssm_state_dim, D_C=cnfg.conv_kernel, dt_min=cnfg.dt_min, dt_max=cnfg.dt_max, name="temporal")(nn.Dense(C, name="fusion_proj")(H), train=train, mask=mask)  # (B, S, C_s)

        if cnfg.slow_center:
            s = session_centre(s)  # the slow path can only shift the prediction within a session, not set its level

        t = inputs["window_start"].astype(jnp.int32)[:, None] + jnp.arange(n_frames)[None]  # (B, L')
        n_bins = jnp.maximum(mask.sum(-1).astype(jnp.int32), 1)
        return interpolate_bins(s, t, n_bins, cnfg.slow_bin)  # (B, L', C_s)
