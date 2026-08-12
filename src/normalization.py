"""Per-channel input standardisation.

The raw streams live on wildly different scales - eGeMAPS reaches +-2.6e5 and
OpenFace2 +-3e4, while w2v-BERT sits in +-4. Feeding those straight into the
network makes activations blow up by 1e14 in the cross-modal block and the Beta
heads emit alpha 1e14, so the NLL is astronomically large and training is
useless. Standardising each channel to zero mean / unit variance fixes this.

Statistics are computed once over the TRAIN split (never val/test, to avoid
leakage), then cached to disk and reused.
"""

from __future__ import annotations

import numpy as np

from pathlib import Path

from read_data import ROLES, read_stream, log


def _stats_path(cnfg) -> Path:
    safe_corpus = cnfg.corpus.replace("/", "_")
    return cnfg.data_root / f"norm_stats_{safe_corpus}.npz"


def compute_stats(cnfg, split: str = "train") -> dict[str, tuple[np.ndarray, np.ndarray]]:
    split_path = cnfg.split_dir(split)
    session_dirs = sorted([p for p in split_path.iterdir() if p.is_dir()])

    # float64 accumulators: sum of x and x^2 are large for 1e5-scale channels
    count: dict[str, float] = {}
    total: dict[str, np.ndarray] = {}
    total_sq: dict[str, np.ndarray] = {}

    # Roles share encoder weights, so they must share normalisation statistics too.
    for session_dir in session_dirs:
        for feat in cnfg.modality_names:
            for role in ROLES:
                p = session_dir / f"{role}.{feat}.stream"
                if not p.exists():
                    continue
                data, _sr = read_stream(p)
                d64 = data.astype(np.float64)

                if feat not in total:
                    total[feat] = np.zeros(d64.shape[1], dtype=np.float64)
                    total_sq[feat] = np.zeros(d64.shape[1], dtype=np.float64)
                    count[feat] = 0.0

                total[feat] += d64.sum(axis=0)
                total_sq[feat] += (d64 * d64).sum(axis=0)
                count[feat] += d64.shape[0]

    stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for feat in total:
        n = max(count[feat], 1.0)
        mean = total[feat] / n
        var = np.maximum(total_sq[feat] / n - mean * mean, 0.0)
        std = np.sqrt(var)
        std[std < 1e-8] = 1.0  # Constant channels (std == 0) must not divide by zero; leave them centred only.
        stats[feat] = (mean.astype(np.float32), std.astype(np.float32))
        log.info(f"norm stats {feat}: mean|.|max={np.abs(mean).max():.4g} std max={std.max():.4g} (n={int(n)} frames)")

    return stats


def load_or_compute_stats(cnfg) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    path = _stats_path(cnfg)

    if path.exists():
        with np.load(path) as z:
            cached = {k[:-5]: (z[k], z[k.replace("_mean", "_std")]) for k in z.files if k.endswith("_mean")}
        if all(feat in cached for feat in cnfg.modality_names):
            log.info(f"loaded normalisation stats from {path}")
            return cached
        log.info(f"{path} missing some active modalities - recomputing")

    log.info("computing normalisation statistics over the train split (one-off)...")
    stats = compute_stats(cnfg, split="train")

    flat: dict[str, np.ndarray] = {}
    for feat, (mean, std) in stats.items():
        flat[f"{feat}_mean"] = mean
        flat[f"{feat}_std"] = std
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **flat)
    log.info(f"saved normalisation stats to {path}")

    return stats


def apply_stats(data: np.ndarray, stats: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    """Standardise (T, C) using (mean, std); a no-op when stats are unavailable."""
    if stats is None:
        return data
    mean, std = stats
    if data.shape[1] != mean.shape[0]:
        log.warning(f"normalisation shape mismatch: data C={data.shape[1]} vs stats C={mean.shape[0]}; skipping")
        return data
    return ((data - mean) / std).astype(np.float32)
