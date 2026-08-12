"""Did the model collapse to a constant? Compare prediction spread vs target spread on val."""
import jax, numpy as np
import logging
import orbax.checkpoint as ocp
import sys

sys.path.insert(0, "src")
logging.getLogger("EngageNet-Logger").setLevel(logging.WARNING)

from config import CORE_MODALITIES, EngageNetConfig
from data_loader import iter_batches
from read_data import ROLES
from train import create_train_state, eval_step, _drop_meta


cnfg = EngageNetConfig(active_modalities=CORE_MODALITIES)
state = create_train_state(cnfg, jax.random.PRNGKey(0))
ckpt = cnfg.checkpoint_dir / "best"
restored = ocp.StandardCheckpointer().restore(ckpt, {"params": state.params, "batch_stats": state.batch_stats})
state = state.replace(params=restored["params"], batch_stats=restored["batch_stats"])
print(f"loaded {ckpt}")

preds, targs = [], []
for i, batch in enumerate(iter_batches(cnfg, split="val")):
    b = _drop_meta(batch)
    p = eval_step(state, b, 0.5688)
    preds.append(np.array(p).reshape(-1))
    t = np.stack([np.array(b[f"{r}.engagement"]) for r in ROLES if f"{r}.engagement" in b]).mean(0)
    targs.append(t.reshape(-1))
    if i >= 40:
        break


p = np.concatenate(preds); t = np.concatenate(targs)

print(f"\nPREDICTIONS: mean={p.mean():.4f}  std={p.std():.6f}  min={p.min():.4f}  max={p.max():.4f}")
print(f"TARGETS    : mean={t.mean():.4f}  std={t.std():.6f}  min={t.min():.4f}  max={t.max():.4f}")
print(f"\nprediction spread is {t.std()/max(p.std(),1e-12):.0f}x SMALLER than the targets'")
print(f"correlation: {np.corrcoef(p, t)[0,1]:.4f}")
print(f"\nCCC if we just predicted the constant target-mean: 0.0 (for reference)")
