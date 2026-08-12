"""Run ONE real batch through the model and report where NaN/inf first appears."""
import jax, jax.numpy as jnp
import logging
import sys

sys.path.insert(0, "src")
logging.getLogger("EngageNet-Logger").setLevel(logging.WARNING)

from beta_head import nll_loss
from config import CORE_MODALITIES, EngageNetConfig
from data_loader import iter_batches
from model import EngageNet
from read_data import ROLES
from train import create_train_state, _drop_meta


cnfg = EngageNetConfig(active_modalities=CORE_MODALITIES)
rng = jax.random.PRNGKey(0)
rng, ri = jax.random.split(rng)
state = create_train_state(cnfg, ri)
batch = _drop_meta(next(iter_batches(cnfg, split="train")))

print("=== RAW INPUT STATS ===")
for k in sorted(batch):
    v = batch[k]
    print(f"  {k:40s} min={float(jnp.min(v)):12.4g} max={float(jnp.max(v)):12.4g} "
          f"nan={int(jnp.isnan(v).sum()):d}")

stream_inputs = {k: v for k, v in batch.items() if not k.endswith(".engagement")}
model = EngageNet(cnfg=cnfg)
variables = {"params": state.params, "batch_stats": state.batch_stats}

(alpha, beta, uni), mods = model.apply(
    variables, stream_inputs, tau=1.0, rng=jax.random.PRNGKey(1), train=True,
    mutable=["batch_stats", "intermediates"], capture_intermediates=True)


print("\n=== INTERMEDIATES (first bad one wins) ===")
flat = jax.tree_util.tree_flatten_with_path(mods.get("intermediates", {}))[0]
bad = []

for path, v in flat:
    if not hasattr(v, "shape"):
        continue
    n_nan = int(jnp.isnan(v).sum()); n_inf = int(jnp.isinf(v).sum())
    name = "/".join(str(p.key) if hasattr(p, "key") else str(p.idx) for p in path)
    fin = v[jnp.isfinite(v)]
    mx = float(jnp.max(jnp.abs(fin))) if fin.size else float("nan")
    flag = "  <-- BAD" if (n_nan or n_inf) else ""
    if True:
        bad.append(f"  {name:70s} absmax={mx:12.4g} nan={n_nan} inf={n_inf}{flag}")

print("\n".join(bad) if bad else "  (no NaN/inf and nothing above 1e6 in captured intermediates)")

print(f"\nalpha: nan={int(jnp.isnan(alpha).sum())} min={float(jnp.nanmin(alpha)):.4g} max={float(jnp.nanmax(alpha)):.4g}")
print(f"beta : nan={int(jnp.isnan(beta).sum())} min={float(jnp.nanmin(beta)):.4g} max={float(jnp.nanmax(beta)):.4g}")

tg = jnp.stack([batch[f"{r}.engagement"] for r in ROLES if f"{r}.engagement" in batch]).mean(0)
print(f"target: nan={int(jnp.isnan(tg).sum())} min={float(jnp.min(tg)):.4g} max={float(jnp.max(tg)):.4g}")
print(f"LOSS (forward only, before any update) = {float(nll_loss(alpha, beta, tg)):.6g}")
