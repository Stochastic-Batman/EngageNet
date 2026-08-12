"""Time train_step on synthetic batches - no disk I/O - to separate GPU compute cost from data-loading cost."""
import jax, jax.numpy as jnp
import sys, time

sys.path.insert(0, "src")

from config import CORE_MODALITIES, EngageNetConfig
from read_data import ROLES
from train import create_train_state, train_step


cnfg = EngageNetConfig(active_modalities=CORE_MODALITIES)
rng = jax.random.PRNGKey(0)
rng, ri = jax.random.split(rng)
state = create_train_state(cnfg, ri)

batch = {}
for feat in cnfg.modality_names:
    for role in ROLES:
        batch[f"{role}.{feat}"] = jnp.zeros((cnfg.batch_size, cnfg.input_dim(feat), cnfg.window_len))
for role in ROLES:
    batch[f"{role}.engagement"] = jnp.full((cnfg.batch_size, cnfg.window_len), 0.5)

t0 = time.time()
rng, rs = jax.random.split(rng)
state, loss = train_step(state, batch, rs, 1.0)
jax.block_until_ready(loss)
print(f"first step (includes compile): {time.time()-t0:.1f}s")

N = 10
t0 = time.time()
for _ in range(N):
    rng, rs = jax.random.split(rng)
    state, loss = train_step(state, batch, rs, 1.0)

jax.block_until_ready(loss)
dt = (time.time() - t0) / N

print(f"steady-state: {dt:.3f}s per batch of {cnfg.batch_size}")
