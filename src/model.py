from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from beta_head import MultiHeadBeta
from bimamba import IntraModalBiMamba
from inter_modal import InterModalBiMamba
from config import EngageNetConfig
from modality_frontend import ModalityFrontend


class EngageNet(nn.Module):
    cnfg: EngageNetConfig

    # inputs: dict{str: (B, C_i, L)}; ... -> (multimodal_alpha: (B, L'), multimodal_beta: (B, L'), unimodal: dict{str: (alpha, beta)})
    @nn.compact
    def __call__(self, inputs: dict[str, jax.Array], *, tau: float = 1.0, rng: jax.Array | None = None, train: bool = True):
        cnfg = self.cnfg
        C = cnfg.shared_dim

        hiddens = ModalityFrontend(cnfg=cnfg, name="frontend")(inputs, train=train)  # dict{str: (B, L', C')}

        u = IntraModalBiMamba(D=C, N=cnfg.ssm_state_dim, D_C=cnfg.conv_kernel, dt_min=cnfg.dt_min, dt_max=cnfg.dt_max, name="intra_modal")(hiddens, train=train)  # dict{str: (B, L', C')}
        
        H = InterModalBiMamba(D=C, N=cnfg.ssm_state_dim, D_C=cnfg.cross_modal_conv_kernel, GS_dim=cnfg.gs_dim, n_iters=cnfg.gs_iters, name="inter_modal")(u, tau=tau, rng=rng, train=train)  # (B, L', MC')
        
        multimodal_alpha, multimodal_beta, unimodal = MultiHeadBeta(hidden_dim=cnfg.beta_hidden, name="beta_heads")(H, u)  # (multimodal_alpha: (B, L'), multimodal_beta: (B, L'), unimodal: dict{str: ((B, L'), (B, L'))})

        return multimodal_alpha, multimodal_beta, unimodal

