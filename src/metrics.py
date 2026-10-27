from __future__ import annotations

import jax.numpy as jnp
import numpy as np


# predictions: (N,) ; targets: (N,) -> scalar in [-1, 1]
def ccc(predictions: np.ndarray, targets: np.ndarray) -> float:
    mu_p, mu_t = predictions.mean(), targets.mean()
    var_p, var_t = predictions.var(), targets.var()
    cov_pt = ((predictions - mu_p) * (targets - mu_t)).mean()
    return float(2 * cov_pt / (var_p + var_t + (mu_p - mu_t) ** 2 + 1e-8))


# per_domain_cccs: dict{domain_name: ccc_value} -> scalar
def combined_ccc(per_domain_cccs: dict[str, float]) -> float:
    return float(np.mean(list(per_domain_cccs.values())))


# predictions: (N,) ; targets: (N,) ; genders: (N,) -> scalar
def cdd_gender(predictions: np.ndarray, targets: np.ndarray, genders: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.quantile(targets, np.linspace(0, 1, n_bins + 1))
    bin_indices = np.digitize(targets, bin_edges[1:-1])

    cdd = 0.0
    total = 0
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        male = (genders[mask] == 1)
        female = (genders[mask] == 0)
        if male.sum() == 0 or female.sum() == 0:
            continue
        diff = predictions[mask][male].mean() - predictions[mask][female].mean()
        cdd += diff * mask.sum()
        total += mask.sum()

    return float(cdd / max(total, 1))


# predictions: (N,) ; targets: (N,) ; languages: (N,) -> dict{lang: cdd_value}
def cdd_language(predictions: np.ndarray, targets: np.ndarray, languages: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    bin_edges = np.quantile(targets, np.linspace(0, 1, n_bins + 1))
    bin_indices = np.digitize(targets, bin_edges[1:-1])

    unique_langs = np.unique(languages)
    result: dict[str, float] = {}

    for lang in unique_langs:
        cdd = 0.0
        total = 0
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() == 0:
                continue
            lang_mask = languages[mask] == lang
            if lang_mask.sum() == 0 or (~lang_mask).sum() == 0:
                continue
            diff = predictions[mask][lang_mask].mean() - predictions[mask].mean()
            cdd += diff * mask.sum()
            total += mask.sum()
        result[str(lang)] = float(cdd / max(total, 1))

    return result


# predictions: (N,) ; targets: (N,) -> scalar, differentiable
def ccc_jnp(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    mu_p, mu_t = predictions.mean(), targets.mean()
    var_p, var_t = predictions.var(), targets.var()
    cov_pt = ((predictions - mu_p) * (targets - mu_t)).mean()
    return 2 * cov_pt / (var_p + var_t + (mu_p - mu_t) ** 2 + 1e-8)


# predictions: (N,) ; targets: (N,) ; groups: (N,) in {0, 1}, -1 = unknown -> scalar, differentiable
def fairness_penalty(predictions: jnp.ndarray, targets: jnp.ndarray, groups: jnp.ndarray, n_bins: int = 10) -> jnp.ndarray:
    """Squared CDD: bin-weighted mean of the squared group-mean gap within target quantile bins.

    Zero when both groups receive the same prediction at the same target level.
    Bins missing either group contribute to neither numerator nor denominator.
    """
    qs = jnp.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = jnp.quantile(targets, qs)
    idx = jnp.searchsorted(edges, targets)  # (N,) in [0, n_bins-1]

    num = 0.0
    den = 0.0
    for b in range(n_bins):  # unrolled at trace time; n_bins is static
        in_bin = (idx == b)
        a = in_bin & (groups == 1)
        c = in_bin & (groups == 0)
        n_a, n_c = a.sum(), c.sum()
        mean_a = (predictions * a).sum() / jnp.maximum(n_a, 1)
        mean_c = (predictions * c).sum() / jnp.maximum(n_c, 1)
        valid = (n_a > 0) & (n_c > 0)
        w = jnp.where(valid, in_bin.sum(), 0.0)
        num = num + w * (mean_a - mean_c) ** 2
        den = den + w

    return num / jnp.maximum(den, 1.0)


# predictions: (N,) ; targets: (N,) -> scalar
def pearson(predictions: np.ndarray, targets: np.ndarray) -> float:
    if predictions.std() < 1e-8 or targets.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(predictions, targets)[0, 1])
