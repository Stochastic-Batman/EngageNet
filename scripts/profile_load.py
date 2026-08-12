"""Where does the ~110s per session actually go? Time disk read vs windowing separately."""
import logging
import sys, time

from pathlib import Path

sys.path.insert(0, "src")
logging.getLogger("EngageNet-Logger").setLevel(logging.WARNING)

from config import CORE_MODALITIES, EngageNetConfig
from dataset import EngageNetDataset
from read_data import ROLES, read_stream, load_session


sess = Path(sys.argv[1])
cnfg = EngageNetConfig(active_modalities=CORE_MODALITIES)

# 1. raw disk read of the 4 core streams, both roles
t0 = time.time()
tot_bytes = 0
for role in ROLES:
    for feat in CORE_MODALITIES:
        p = sess / f"{role}.{feat}.stream"
        d, sr = read_stream(p)
        tot_bytes += d.nbytes
t_read = time.time() - t0
print(f"1. raw stream read      : {t_read:6.1f}s  ({tot_bytes/1e9:.2f} GB in RAM, {tot_bytes/1e6/max(t_read,.01):.0f} MB/s)")

# 2. full load_session (read + annotations)
t0 = time.time()
s = load_session(sess, features=CORE_MODALITIES)
t_load = time.time() - t0
print(f"2. load_session total   : {t_load:6.1f}s")

# 3. windowing: iterate all windows of this one session
ds = EngageNetDataset(cnfg, "train", session_dirs=[sess])
t0 = time.time()
n = 0
for w in ds.iter_windows():
    n += 1
t_win = time.time() - t0
print(f"3. windowing ({n:4d} win) : {t_win:6.1f}s   <- includes another load_session")
print(f"   -> windowing alone   : {t_win - t_load:6.1f}s  ({(t_win-t_load)/max(n,1)*1000:.0f} ms/window)")
