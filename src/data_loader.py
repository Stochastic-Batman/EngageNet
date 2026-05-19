"""Batch-yielding data loader built on top of `dataset.EngageNetDataset`.

Works as a pure-Python generator so no extra framework dependency is needed.
Batches are constructed lazily: only one session at a time is in memory.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from typing import Iterator

from config import EngageNetConfig
from dataset import EngageNetDataset


def _collate(windows: list[dict[str, np.ndarray]]) -> dict[str, jnp.ndarray]:
    """Stack a list of window dicts into a single batched dict of JAX arrays.

    Numeric arrays are stacked along a new leading batch axis.
    The "session" key is kept as a plain Python list of strings.
    """
    batch: dict[str, jnp.ndarray] = {}
    keys = [k for k in windows[0].keys() if k != "session"]
    
    for k in keys:
        batch[k] = jnp.array(np.stack([w[k] for w in windows], axis=0))
        
    batch["session"] = [w["session"] for w in windows]
    
    return batch


def iter_batches(cnfg: EngageNetConfig, split: str, *, shuffle: bool = False, seed: int | None = None) -> Iterator[dict[str, jnp.ndarray]]:
    """Yield batches of size `cnfg.batch_size` from the given `split`.

    cnfg : EngageNetConfig
    split : One of "train", "val", "test".
    shuffle : whether to shuffle session order each epoch
    seed : RNG seed (used only when `shuffle` is True)
    """
    ds = EngageNetDataset(cnfg, split)
    rng_key = jax.random.PRNGKey(seed if seed is not None else 0) if shuffle else None

    buf: list[dict[str, np.ndarray]] = []
    for window in ds.iter_windows(shuffle_sessions=shuffle, rng_key=rng_key):
        buf.append(window)
        if len(buf) == cnfg.batch_size:
            yield _collate(buf)
            buf.clear()
    # yield the last (possibly smaller) batch
    if buf:
        yield _collate(buf)

