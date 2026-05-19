"""ModalityFrontend - runs every InitEncoder and the uniform channel projection.

Inputs: per-modality tensors {role}.{feat} each of shape (B, C_i, L)
Outputs: per-modality hidden {role}.{feat} each of shape (B, L', C')

Weight sharing: expert and novice for the SAME feature type share the same InitEncoder (and projection), since the underlying signal semantics are identical.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from config import EngageNetConfig
from init_encoder import InitEncoder
from read_data import ROLES


class ModalityFrontend(nn.Module):
    """Top-level module that owns all InitEncoders + channel projections."""

    cnfg: EngageNetConfig

    @nn.compact
    def __call__(self, inputs: dict[str, jnp.ndarray], *, train: bool = True) -> dict[str, jnp.ndarray]:
        """
        inputs : dict
            Keys are "{role}.{feat}" with values of shape (B, C_i, L).
            Missing modalities are silently skipped.
        train : bool
            Forwarded to BatchNorm layers inside each InitEncoder.

        Returns
        hiddens : dict
            Same keys as inputs (minus missing ones), values (B, L', C').
        """
        cnfg = self.cnfg
        hiddens: dict[str, jnp.ndarray] = {}

        # Only iterate over active modalities (respects cnfg.active_modalities filter)
        for feat, (_, c_out, ks, stride) in cnfg.active_specs.items():

            # One encoder per feature type, shared across roles
            encoder = InitEncoder(out_channels=c_out, kernel_size=ks, stride=stride, name=f"enc_{feat}")

            # Uniform channel projection  f_i : R^{C_i'} -> R^{C'}
            proj = nn.Dense(features=cnfg.shared_dim, name=f"proj_{feat}")

            for role in ROLES:
                key = f"{role}.{feat}"
                if key not in inputs:
                    continue

                h = encoder(inputs[key], train=train)        # (B, L', C_i')
                h = proj(h)                                  # (B, L', C')
                hiddens[key] = h

        return hiddens  # shape: (B, L', C')

