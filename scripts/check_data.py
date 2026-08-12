"""Look for NaN/inf and check value scales in the real streams + engagement labels."""
import sys
import numpy as np, logging

from pathlib import Path

sys.path.insert(0, "src")
logging.getLogger("EngageNet-Logger").setLevel(logging.WARNING)

from config import CORE_MODALITIES
from read_data import ROLES, read_stream, read_engagement


root = Path("data/NoXi+J/train")
sessions = sorted([p for p in root.iterdir() if p.is_dir()])[:5]


print(f"{'session':>6} {'stream':>32} {'min':>12} {'max':>12} {'mean':>10} {'nan':>8} {'inf':>7}")

for s in sessions:
    for feat in CORE_MODALITIES:
        for role in ROLES[:1]:
            p = s / f"{role}.{feat}.stream"
            if not p.exists():
                continue
            d, sr = read_stream(p)
            n_nan = int(np.isnan(d).sum()); n_inf = int(np.isinf(d).sum())
            fin = d[np.isfinite(d)]
            print(f"{s.name:>6} {feat:>32} {fin.min():12.3g} {fin.max():12.3g} {fin.mean():10.3g} {n_nan:8d} {n_inf:7d}")

    e = read_engagement(s / "expert.engagement.annotation.csv").values.astype(np.float32)
    print(f"{s.name:>6} {'ENGAGEMENT':>32} {np.nanmin(e):12.3g} {np.nanmax(e):12.3g} {np.nanmean(e):10.3g} {int(np.isnan(e).sum()):8d} {int(np.isinf(e).sum()):7d}")
    print()
