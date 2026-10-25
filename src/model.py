from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from beta_head import MultiHeadBeta
from bimamba import BiMambaBlock, IntraModalBiMamba
from inter_modal import InterModalBiMamba
from config import EngageNetConfig
from modality_frontend import ModalityFrontend
from slow_path import SlowPath


class EngageNet(nn.Module):
    cnfg: EngageNetConfig

    # inputs: dict{str: (B, C_i, L)}; ... -> (dict{role: (alpha, beta)}, dict{"{role}.{feat}": (alpha, beta)})
    @nn.compact
    def __call__(self, inputs: dict[str, jax.Array], *, tau: float = 1.0, rng: jax.Array | None = None, train: bool = True):
        cnfg = self.cnfg
        C = cnfg.shared_dim

        hiddens = ModalityFrontend(cnfg=cnfg, name="frontend")(inputs, train=train)  # dict{str: (B, L', C')}

        u = IntraModalBiMamba(D=C, N=cnfg.ssm_state_dim, D_C=cnfg.conv_kernel, dt_min=cnfg.dt_min, dt_max=cnfg.dt_max, n_layers=cnfg.intra_layers, name="intra_modal")(hiddens, train=train)  # dict{str: (B, L', C')}
        
        H = InterModalBiMamba(D=C, N=cnfg.ssm_state_dim, D_C=cnfg.cross_modal_conv_kernel, GS_dim=cnfg.gs_dim, n_iters=cnfg.gs_iters, fixed_order=cnfg.fixed_order, position_queries=cnfg.position_queries, scale_scores=cnfg.scale_scores, unsort=cnfg.unsort, probe_order=cnfg.probe_order, dt_min=cnfg.dt_min, dt_max=cnfg.dt_max, name="inter_modal")(u, tau=tau, rng=rng, train=train)  # (B, L', MC')
        
        if cnfg.fusion_temporal:
            # The inter-modal block fuses each frame on its own; this scans the fused state of both people over time
            z = BiMambaBlock(D=C, N=cnfg.ssm_state_dim, D_C=cnfg.conv_kernel, dt_min=cnfg.dt_min, dt_max=cnfg.dt_max, name="fusion_temporal")(nn.Dense(C, name="fusion_proj")(H), train=train)  # (B, L', C')
            H = jnp.concatenate([H, z], axis=-1)  # (B, L', MC' + C')

        if cnfg.slow_path:
            # Second rate: the whole session in slow_bin-frame bins, read back at this window's frames
            s = SlowPath(cnfg=cnfg, name="slow_path")(inputs, n_frames=H.shape[1], train=train)  # (B, L', C_s)
            H = jnp.concatenate([H, s], axis=-1)

        multimodal, unimodal = MultiHeadBeta(hidden_dim=cnfg.beta_hidden, name="beta_heads")(H, u)  # (dict{role: (alpha, beta)}, dict{"{role}.{feat}": (alpha, beta)})

        return multimodal, unimodal
