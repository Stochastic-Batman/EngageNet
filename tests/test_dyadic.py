#!/usr/bin/env python
"""Checks for the coupled dyadic SSM (src/ssm.py coupled_ssm, src/bimamba.py DyadicBiMambaBlock).

Usage (from repo root):
    python tests/test_dyadic.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bimamba import DyadicBiMambaBlock
from read_data import log
from ssm import coupled_ssm, ssm


def inputs(key, B=2, L=30, D=8, N=4):
    k = jax.random.split(key, 6)
    u = jax.random.normal(k[0], (2, B, L, D))
    delta = jax.nn.softplus(jax.random.normal(k[1], (2, B, L, D)))
    A = -jnp.exp(jax.random.normal(k[2], (2, D, N)))
    Bm = jax.random.normal(k[3], (2, B, L, N))
    Cm = jax.random.normal(k[4], (2, B, L, N))
    kappa = jax.nn.sigmoid(jax.random.normal(k[5], (2, B, L, D)))
    return u, delta, A, Bm, Cm, kappa


def sequential(u, delta, A, Bm, Cm, kappa):
    u, delta, A, Bm, Cm, kappa = map(np.asarray, (u, delta, A, Bm, Cm, kappa))
    _, B, L, D = u.shape
    h = np.zeros((2, B, D, A.shape[-1]))
    ys = []
    for t in range(L):
        a = np.exp(delta[:, :, t, :, None] * A[:, None])  # (2, B, D, N)
        g = (1 - a) * kappa[:, :, t, :, None]
        b = delta[:, :, t, :, None] * Bm[:, :, t, None, :] * u[:, :, t, :, None]
        h = np.stack([a[0] * h[0] + g[0] * h[1] + b[0], a[1] * h[1] + g[1] * h[0] + b[1]])
        ys.append((Cm[:, :, t, None, :] * h).sum(-1))
    return np.stack(ys, axis=2)  # (2, B, L, D)


def main() -> None:
    u, delta, A, Bm, Cm, kappa = inputs(jax.random.PRNGKey(0))

    y = coupled_ssm(u, delta, A, Bm, Cm, kappa)
    err = np.abs(np.asarray(y) - sequential(u, delta, A, Bm, Cm, kappa)).max()
    log.info(f"coupled scan vs sequential loop: max error {err:.2e}")
    assert err < 1e-4

    y0 = coupled_ssm(u, delta, A, Bm, Cm, jnp.zeros_like(kappa))
    for r in range(2):
        err0 = np.abs(np.asarray(y0[r] - ssm(u[r], delta[r], A[r], Bm[r], Cm[r]))).max()
        assert err0 < 1e-5, err0
    log.info("kappa = 0 reproduces two independent SSMs")

    # The partner influences person 0 only when coupled
    u2 = u.at[1].add(1.0)
    assert np.abs(np.asarray(coupled_ssm(u2, delta, A, Bm, Cm, jnp.zeros_like(kappa))[0] - y0[0])).max() < 1e-6
    assert np.abs(np.asarray(coupled_ssm(u2, delta, A, Bm, Cm, kappa)[0] - y[0])).max() > 1e-3
    log.info("partner input reaches person 0 only through the coupling")

    h = jax.random.normal(jax.random.PRNGKey(1), (2, 2, 30, 16))
    for coupled in (True, False):
        blk = DyadicBiMambaBlock(D=16, N=4, D_C=4, coupled=coupled)
        params = blk.init(jax.random.PRNGKey(2), h)
        out = blk.apply(params, h)
        assert out.shape == h.shape and np.isfinite(np.asarray(out)).all()
        n = sum(x.size for x in jax.tree.leaves(params))
        log.info(f"DyadicBiMambaBlock coupled={coupled}: params {n:,}, output {out.shape}")

    log.info("All checks passed!")


if __name__ == "__main__":
    main()
