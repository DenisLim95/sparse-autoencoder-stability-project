"""Exercise safe_hist against the exact data shapes that crashed the 100M run."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

lines = open("prelim_experiments_update.py").read().splitlines()
start = next(i for i, l in enumerate(lines) if l.startswith("def safe_hist"))
end = next(i for i, l in enumerate(lines) if i > start and l.startswith("# Visualize"))
ns = {"np": np}
exec(compile("\n".join(lines[start:end]), "safe_hist", "exec"), ns)
safe_hist = ns["safe_hist"]

fig, ax = plt.subplots()
cases = {
    "exactly constant 1.0 (what the 8B run had)": np.ones(1616, dtype=np.float32),
    "1.0 +/- float32 last bits (what crashed)":
        (np.ones(1616, dtype=np.float32)
         + np.random.default_rng(0).integers(-3, 4, 1616) * np.finfo(np.float32).eps),
    "constant zero": np.zeros(50),
    "constant large value": np.full(50, 1e6),
    "single element": np.array([0.7]),
    "empty": np.array([]),
    "all NaN": np.full(10, np.nan),
    "some NaN and inf": np.array([1.0, 2.0, np.nan, np.inf, 3.0]),
    "genuinely varying (must still bin normally)": np.random.default_rng(1).normal(size=500),
    "tiny but real spread at small scale": np.random.default_rng(2).normal(0, 1e-9, 200),
}
for label, v in cases.items():
    try:
        before = len(ax.patches)
        safe_hist(ax, v, 30, alpha=0.5)
        drew = len(ax.patches) - before
        print(f"  ok    {label}  (bars drawn: {drew})")
    except Exception as e:
        print(f"  FAIL  {label}: {type(e).__name__}: {e}")

# the non-degenerate case must not be collapsed to one bar
ax2 = plt.subplots()[1]
safe_hist(ax2, np.random.default_rng(3).normal(size=1000), 30)
print(f"\n  varying data still gets many bins: {len(ax2.patches)} bars")
