#!/usr/bin/env python
"""Write best_tau.json for a checkpoint trained before train.py recorded it.

Reads the training log, takes the temperature of the last epoch that produced "New best val CCC"
(that is the epoch the "best" checkpoint holds) and writes {"tau": ..., "epoch": ...} next to it,
so that inference.py and dump_P.py run the model at the temperature it was selected at.

usage (repo root): python scripts/best_tau.py logs/train_<tag>_s<seed>.log models_<tag>_s<seed>
"""

from __future__ import annotations

import json
import re
import sys

from pathlib import Path


def main() -> None:
    log_path, model_dir = Path(sys.argv[1]), Path(sys.argv[2])
    epoch = tau = best = None
    for line in log_path.read_text().splitlines():
        m = re.search(r"Epoch\s+(\d+)/\d+\s+tau=([\d.]+)", line)
        if m:
            epoch, tau = int(m.group(1)), float(m.group(2))
        elif "New best val CCC" in line and epoch is not None:
            best = (epoch, tau)
    if best is None:
        raise SystemExit(f"no 'New best val CCC' line in {log_path}")
    out = model_dir / "best_tau.json"
    out.write_text(json.dumps({"tau": best[1], "epoch": best[0]}))
    print(f"{model_dir.name}: best epoch {best[0]}, tau {best[1]:.4f} -> {out}")


if __name__ == "__main__":
    main()
