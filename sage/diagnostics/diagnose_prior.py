"""Plot the FULL parameter prior of the dummy replica (== o3b_dummy_1 gwconfig).
CPU only. Draws directly from the DistributionSampler (no waveforms)."""
import sys, os, math
sys.path.insert(0, "/home/nagarajan/research/sage")
sys.path.insert(0, "/home/nagarajan/research/sage/runs/o3b")
os.chdir("/home/nagarajan/research/sage/runs/o3b")
import config; config.set_configs()
from sage.core.config import get_cfg
cfg = get_cfg()
try: cfg.device = "cpu"
except Exception: pass
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sage.data.waveform import read_from_config

ps = read_from_config("./gwconfig.yaml", seed=150914)
# force CPU sampling
try: ps.device = torch.device("cpu")
except Exception: pass
try: ps.generator = torch.Generator(device="cpu"); ps.generator.manual_seed(150914)
except Exception: pass

N = 20000
with torch.no_grad():
    p = ps.forward(N).cpu().numpy()
pidx = ps.param_index
print("param_index:", {k: pidx[k] for k in sorted(pidx, key=lambda x: pidx[x])})

want = ["mass1", "mass2", "mchirp", "q", "spin1_a", "spin2_a", "spin1z", "spin2z",
        "chirp_distance", "distance", "tc", "ra", "dec", "inclination",
        "coa_phase", "polarization", "injection_time"]
avail = [w for w in want if w in pidx]

ncol = 4; nrow = math.ceil(len(avail) / ncol)
fig, ax = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.8 * nrow))
ax = np.atleast_1d(ax).ravel()
for i, name in enumerate(avail):
    c = pidx[name]; col = p[:, c]
    ax[i].hist(col, bins=60, color="C0")
    ax[i].set_title(f"{name}  [{col.min():.3g}, {col.max():.3g}]", fontsize=9)
for j in range(len(avail), len(ax)): ax[j].axis("off")
plt.suptitle(f"Prior (dummy replica gwconfig, N={N}) — distance via chirp_distance_to_distance", fontsize=11)
plt.tight_layout()
out = "/home/nagarajan/research/sage/sage/diagnostics/plots/prior.png"
plt.savefig(out, dpi=100); print("saved", out)

print("\n=== prior summary ===")
for name in avail:
    c = pidx[name]; col = p[:, c]
    print(f"  {name:14s} min={col.min():.4g}  med={np.median(col):.4g}  "
          f"mean={col.mean():.4g}  max={col.max():.4g}")
