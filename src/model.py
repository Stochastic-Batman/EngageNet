from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from beta_head import MultiHeadBeta
from bimamba import BiMambaBlock, DyadicBiMambaBlock, IntraModalBiMamba
from inter_modal import InterModalBiMamba
from config import EngageNetConfig
from read_data import ROLES
from modality_frontend import ModalityFrontend


class EngageNet(nn.Module):
    cnfg: EngageNetConfig

    # inputs: dict{str: (B, C_i, L)}; ... -> (dict{role: (alpha, beta)}, dict{"{role}.{feat}": (alpha, beta)})
    @nn.compact
    def __call__(self, inputs: dict[str, jax.Array], *, tau: float = 1.0, rng: jax.Array | None = None, train: bool = True):
        cnfg = self.cnfg
        C = cnfg.shared_dim

        hiddens = ModalityFrontend(cnfg=cnfg, name="frontend")(inputs, train=train)  # dict{str: (B, L', C')}

        u = IntraModalBiMamba(D=C, N=cnfg.ssm_state_dim, D_C=cnfg.conv_kernel, dt_min=cnfg.dt_min, dt_max=cnfg.dt_max, name="intra_modal")(hiddens, train=train)  # dict{str: (B, L', C')}
        
        H = InterModalBiMamba(D=C, N=cnfg.ssm_state_dim, D_C=cnfg.cross_modal_conv_kernel, GS_dim=cnfg.gs_dim, n_iters=cnfg.gs_iters, fixed_order=cnfg.fixed_order, position_queries=cnfg.position_queries, scale_scores=cnfg.scale_scores, unsort=cnfg.unsort, probe_order=cnfg.probe_order, dt_min=cnfg.dt_min, dt_max=cnfg.dt_max, name="inter_modal")(u, tau=tau, rng=rng, train=train)  # (B, L', MC')
        
        if cnfg.fusion_temporal:
            # The inter-modal block fuses each frame on its own; this scans the fused state of both people over time
            z = BiMambaBlock(D=C, N=cnfg.ssm_state_dim, D_C=cnfg.conv_kernel, dt_min=cnfg.dt_min, dt_max=cnfg.dt_max, name="fusion_temporal")(nn.Dense(C, name="fusion_proj")(H), train=train)  # (B, L', C')
            H = jnp.concatenate([H, z], axis=-1)  # (B, L', MC' + C')

        if cnfg.dyadic_ssm:
            # Each person's own streams, projected per frame, then scanned over time with one state per person, coupled to the partner's
            proj = nn.Dense(C, name="dyadic_proj")  # shared by both roles
            x = jnp.stack([proj(jnp.concatenate([u[f"{r}.{f}"] for f in cnfg.modality_names], axis=-1)) for r in ROLES])  # (2, B, L', C')
            z = DyadicBiMambaBlock(D=C, N=cnfg.ssm_state_dim, D_C=cnfg.conv_kernel, coupled=not cnfg.dyadic_uncoupled, dt_min=cnfg.dt_min, dt_max=cnfg.dt_max, name="dyadic")(x, train=train)
            H = {r: jnp.concatenate([H, z[i]], axis=-1) for i, r in enumerate(ROLES)}  # dict{role: (B, L', MC' + C')}

        multimodal, unimodal = MultiHeadBeta(hidden_dim=cnfg.beta_hidden, name="beta_heads")(H, u)  # (dict{role: (alpha, beta)}, dict{"{role}.{feat}": (alpha, beta)})

        return multimodal, unimodal
