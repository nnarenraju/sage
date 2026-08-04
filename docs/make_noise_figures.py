"""Generate the two figures for the synthetic-noise docs page."""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

from sage.data.noise import sample_synthetic_noise
from sage.data.noise.white_noise import resolve_asd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_static")

# Palette (validated reference instance, light mode)
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False,
    "savefig.facecolor": SURFACE,
})


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, which="major", alpha=1.0)
    ax.grid(True, which="minor", alpha=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)


def log_asd_ticks(ax, ticks):
    """Labelled ticks on a narrow log ASD axis, where decades are too sparse."""
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:.0e}".replace("e-2", "e-2") for v in ticks])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())


def log_freq_ticks(ax):
    """Readable labelled decades on a log frequency axis."""
    ticks = [20, 50, 100, 200, 500, 1000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())


FS, FLOW = 2048.0, 15.0

# ── Figure 1: the ASD models you can request ────────────────────────────────
MODELS = [
    ("aLIGOZeroDetHighPower",  "aLIGO Zero-Det High Power", S1),
    ("AdVO4T1800545",          "AdV O4",                    S2),
    ("KAGRA128MpcT1800545",    "KAGRA 128 Mpc",             S3),
]

n = int(8 * FS)
freqs = torch.fft.rfftfreq(n, d=1.0 / FS, dtype=torch.float64)
f = freqs.numpy()

fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=150)
for name, label, colour in MODELS:
    asd = resolve_asd(name, freqs=freqs, sample_rate=FS,
                      low_frequency_cutoff=FLOW, dtype=torch.float64).numpy()
    # pycbc's from_string zeroes the Nyquist bin; drop non-positive bins so the
    # log axis does not show a spurious plunge at the right edge.
    band = (f >= FLOW) & (f <= 1000) & (asd > 0)
    ax.loglog(f[band], asd[band], color=colour, lw=2.0, label=label,
              solid_capstyle="round")

ax.set_xlim(FLOW, 1000)
ax.set_ylim(3e-24, 3e-22)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel(r"ASD  [strain / $\sqrt{\mathrm{Hz}}$]")
ax.set_title("Analytic detector ASD models", color=INK, fontsize=12,
             fontweight="bold", loc="left", pad=10)
ax.legend(loc="upper right", labelcolor=INK2, fontsize=9.5,
          handlelength=1.6, borderpad=0.4)
style(ax)
log_freq_ticks(ax)
log_asd_ticks(ax, [4e-24, 1e-23, 3e-23, 1e-22, 3e-22])
fig.text(0.008, 0.015,
         "available_asds() lists 94 models; any name is accepted by sample_synthetic_noise",
         color=MUTED, fontsize=8.5)
fig.tight_layout(rect=(0, 0.035, 1, 1))
fig.savefig(f"{OUT}/noise_asd_models.png", bbox_inches="tight")
plt.close(fig)
print("wrote noise_asd_models.png")

# ── Figure 2: what a sample looks like, and that it matches the request ─────
MODEL = "aLIGOZeroDetHighPower"
DUR = 8.0
x = sample_synthetic_noise(DUR, MODEL, batch=256, seed=0,
                           low_frequency_cutoff=FLOW, dtype=torch.float64)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.6), dpi=150)

# top: one second of the strain itself
seg = x[0, : int(FS)].numpy()
t = np.arange(seg.size) / FS
ax1.plot(t, seg * 1e21, color=S1, lw=0.9, solid_capstyle="round")
ax1.set_xlim(0, 1)
ax1.set_xlabel("Time [s]")
ax1.set_ylabel(r"Strain  [$\times 10^{-21}$]")
ax1.set_title("One second of the generated strain",
              color=INK, fontsize=12, fontweight="bold", loc="left", pad=10)
ax1.grid(True, axis="y")
style(ax1)
ax1.grid(False, which="minor")

# bottom: measured ASD of the batch against the model that was requested
seg_len = int(4 * FS)
flat = x.reshape(-1).numpy()
nseg = flat.size // seg_len
win = np.hanning(seg_len)
psd = np.zeros(seg_len // 2 + 1)
for i in range(nseg):
    chunk = flat[i * seg_len:(i + 1) * seg_len] * win
    psd += np.abs(np.fft.rfft(chunk)) ** 2
psd *= 2.0 / (nseg * FS * (win ** 2).sum())
fm = np.fft.rfftfreq(seg_len, d=1.0 / FS)

freqs_m = torch.from_numpy(fm).to(torch.float64)
target = resolve_asd(MODEL, freqs=freqs_m, sample_rate=FS,
                     low_frequency_cutoff=FLOW, dtype=torch.float64).numpy()

m = (fm >= FLOW) & (fm <= 1000) & (target > 0)
ax2.loglog(fm[m], np.sqrt(psd[m]), color=S1, lw=1.6,
           label="Measured from 256 samples", solid_capstyle="round")
ax2.loglog(fm[m], target[m], color=INK, lw=1.6, ls=(0, (4, 3)),
           label="Requested model", solid_capstyle="round")
ax2.set_xlim(FLOW, 1000)
ax2.set_ylim(3e-24, 6e-23)
ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylabel(r"ASD  [strain / $\sqrt{\mathrm{Hz}}$]")
ax2.set_title("Measured spectrum against the requested model",
              color=INK, fontsize=12, fontweight="bold", loc="left", pad=10)
ax2.legend(loc="upper right", labelcolor=INK2, fontsize=9.5,
           handlelength=2.2, borderpad=0.4)
style(ax2)
log_freq_ticks(ax2)
log_asd_ticks(ax2, [4e-24, 6e-24, 1e-23, 2e-23, 4e-23])

fig.tight_layout(h_pad=2.6)
fig.savefig(f"{OUT}/noise_synthetic_example.png", bbox_inches="tight")
plt.close(fig)
print("wrote noise_synthetic_example.png")

band_ratio = np.median(np.sqrt(psd[m]) / target[m])
print(f"median measured/requested ASD ratio over {FLOW}-1024 Hz: {band_ratio:.4f}")
