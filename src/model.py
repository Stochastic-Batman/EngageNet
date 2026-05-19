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
    N: int = 16          # SSM state dim
    D_C: int = 4         # depthwise conv kernel size
    GS_dim: int = 64     # Gumbel-Sinkhorn key dim
    n_iters: int = 10    # Sinkhorn iterations

    # inputs: dict{str: (B, C_i, L)}; ... -> (multimodal_alpha: (B,), multimodal_beta: (B,), unimodal: dict{str: (alpha, beta)})
    @nn.compact
    def __call__(self, inputs: dict[str, jax.Array], *, tau: float = 1.0, rng: jax.Array | None = None, train: bool = True):
        C = self.cnfg.shared_dim
        M = len(self.cnfg.modality_names) * 2  # features x 2 roles

        hiddens = ModalityFrontend(cnfg=self.cnfg, name="frontend")(inputs, train=train)  # dict{str: (B, L', C')}
        u = IntraModalBiMamba(D=C, N=self.N, D_C=self.D_C, name="intra_modal")(hiddens, train=train)  # dict{str: (B, L', C')}
        H = InterModalBiMamba(D=M * C, N=self.N, D_C=self.D_C, GS_dim=self.GS_dim, n_iters=self.n_iters, name="inter_modal")(u, tau=tau, rng=rng, train=train)  # (B, L', MC')
        multimodal_alpha, multimodal_beta, unimodal = MultiHeadBeta(name="beta_heads")(H, u)  # (multimodal_alpha: (B,), multimodal_beta: (B,), unimodal: dict{str: (alpha, beta)})

        return multimodal_alpha, multimodal_beta, unimodal

