"""Lazy, memory-efficient dataset that streams windows from disk.

Each session's streams are loaded on demand via `read_data.load_session`, resampled to a common sample rate, sliced into fixed-length windows, and yielded one window at a time so the full corpus never sits in RAM.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import math
import numpy as np

from pathlib import Path
from scipy.interpolate import interp1d
from typing import Iterator, Optional

from config import EngageNetConfig
from read_data import ROLES, STREAM_FEATURES, load_session, log



def _resample(data: np.ndarray, src_sr: float, tgt_sr: float) -> np.ndarray:
    """Resample `data` (T_src, D) from `src_sr` to `tgt_sr` via linear interpolation."""
    if src_sr == tgt_sr:
        return data
    
    t_src = data.shape[0]
    t_tgt = max(1, round(t_src * tgt_sr / src_sr))
    x_old = np.linspace(0, 1, t_src)
    x_new = np.linspace(0, 1, t_tgt)
    fn = interp1d(x_old, data, axis=0, kind="linear", fill_value="extrapolate")
    
    return fn(x_new).astype(data.dtype)


def _pad_or_trim(arr: np.ndarray, length: int) -> np.ndarray:
    """Ensure temporal axis (axis-0) is exactly `length`; zero-pad if short."""
    t = arr.shape[0]
    if t >= length:
        return arr[:length]
    
    # Unpack remaining dimensions (e.g. channels) so the padding shape matches the input
    pad = np.zeros((length - t, *arr.shape[1:]), dtype=arr.dtype)
    
    return np.concatenate([arr, pad], axis=0)



class EngageNetDataset:
    """Lazy window iterator over a corpus split.

    cnfg : EngageNetConfig
        Global configuration (paths, window settings, modality specs, ...).
    split : str
        One of "train", "val", "test".
    """

    def __init__(self, cnfg: EngageNetConfig, split: str) -> None:
        self.cnfg = cnfg
        if split not in ["train", "test", "val"]:
            raise ValueError(f"Unknown split {split}")
        self.split = split
        
        split_path = cnfg.split_dir(split)
        self.session_dirs: list[Path] = sorted([p for p in split_path.iterdir() if p.is_dir()])
        
        if not self.session_dirs:
            raise FileNotFoundError(f"No session directories found in {split_path}")
        
        log.info(f"EngageNetDataset: {split} - {len(self.session_dirs)} sessions")


    def iter_windows(self, shuffle_sessions: bool = False, rng_key: Optional[jax.Array] = None) -> Iterator[dict[str, np.ndarray]]:
        """Yield one window dict at a time, loading each session lazily.

        Each yielded dict contains:
        - "{role}.{feat}"  -> np.ndarray  shape (C_i, L)  (channels-first)
        - "{role}.engagement" -> np.ndarray shape (L,)   (only train/val)
        - "session"  -> session directory name (str)
        """
        order = list(range(len(self.session_dirs)))
        if shuffle_sessions:
            if rng_key is None:
                rng_key = jax.random.PRNGKey(0)
            order = jax.random.permutation(rng_key, jnp.array(order)).tolist()

        for idx in order:
            yield from self._windows_from_session(self.session_dirs[idx])


    def _windows_from_session(self, session_dir: Path) -> Iterator[dict[str, np.ndarray]]:
        session = load_session(session_dir)
        cnfg = self.cnfg

        # 1. Resample only active streams to target_sr -> dict[key] = (T, D)
        active_feats = cnfg.modality_names
        resampled: dict[str, np.ndarray] = {}

        for role in ROLES:
            streams = session[role].get("streams", {})
            for feat in active_feats:
                entry = streams.get(feat)
                if entry is None:
                    continue
                data = _resample(entry["data"], entry["sr"], cnfg.target_sr)
                resampled[f"{role}.{feat}"] = data

        if not resampled:
            log.warning(f"Session {session_dir.name}: no streams found, skipping")
            return

        # 2. Find the common temporal length across all resampled streams
        common_T = min(v.shape[0] for v in resampled.values())

        # 3. Also resample engagement labels (25 Hz -> target_sr, usually identity)
        engagement: dict[str, Optional[np.ndarray]] = {}
        for role in ROLES:
            eng = session[role].get("engagement")
            if eng is not None:
                arr = eng.values.astype(np.float32)
                # engagement is annotated at 25 Hz
                arr = _resample(arr.reshape(-1, 1), 25.0, cnfg.target_sr).squeeze(-1)
                engagement[role] = arr[:common_T]
            else:
                engagement[role] = None

        # 4. Slice into windows
        W = cnfg.window_len
        S = cnfg.window_stride
        n_windows = max(1, 1 + (common_T - W) // S)

        for w in range(n_windows):
            start = w * S
            end = start + W
            sample: dict[str, np.ndarray] = {"session": session_dir.name}  # type: ignore[dict-item]
            for key, arr in resampled.items():
                chunk = _pad_or_trim(arr[start:end], W)
                # Transpose to channels-first: (L, C_i) -> (C_i, L)
                sample[key] = chunk.T.astype(np.float32)
            for role in ROLES:
                if engagement[role] is not None:
                    sample[f"{role}.engagement"] = engagement[role][start:end]
            yield sample
