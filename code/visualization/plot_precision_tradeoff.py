import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = REPO_ROOT / 'data' / 'figures'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Times-like serif to match a LaTeX pdflatex manuscript
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------
# Data: full 166,667-step trajectory
#   - fp64 throughput from v4 run (v4_precision_full.json)
#   - fp32 + the four sub-fp32 isolations from precision_isolation.json
#   - dX/dZ are RMSE increase vs the all-fp32 baseline, in mm; converted to
#     unitless relative deviation by the fp64 trajectory RMSE
#     (X: 1017.8421 mm, Z: 8.26942 mm) -> values read as 1e-4, 1e-2, 1e2 ...
# -----------------------------
configs = ['fp64', 'fp32', 'io-fp16', 'dot-tf32', 'io-bf16', 'state-fp16']
labels = ['fp64', 'fp32', 'I/O\nfp16', 'dot\nTF32', 'I/O\nbf16', 'state\nfp16']
throughput = np.array([17090, 249619, 249785, 208076, 250142, 252808])  # steps/s

RMSE_X_FP64 = 1017.8421   # mm
RMSE_Z_FP64 = 8.26942     # mm
dX_mm = np.array([np.nan, 0.0, 0.11, 4.82, 11.01, 280753.72])
dZ_mm = np.array([np.nan, 0.0, 2.19, 0.31, 1005.84, 0.30])
dX = np.abs(dX_mm) / RMSE_X_FP64
dZ = np.abs(dZ_mm) / RMSE_Z_FP64

BLUE, BLUE_LIGHT = '#4C78A8', '#B9C7D6'
RED, AMBER, GREEN, GREY = '#E45756', '#F58518', '#54A24B', '#888888'

pos = np.array([0.0, 1.0, 3.0, 4.0, 5.0, 6.0])
SEP = 2.0
FLOOR = 1e-6


def _v(v):
    return FLOOR if (np.isnan(v) or v == 0.0) else v


def sci(v):
    s = f'{v:.1e}'
    m, e = s.split('e')
    return f'{float(m):g}e{int(e)}'


# -----------------------------
# Single-panel combo figure:
#   bars  (left  log, lower band) = throughput
#   lines (right log, upper band) = relative deviation dX / dZ
# The bars are anchored to the lower part of the panel (headroom up to 1e7),
# and the deviation lines are drawn in a separate upper band via a blended
# transform (x = data, y = axes fraction).  The two occupy disjoint vertical
# regions of the same axes, so the line sits above the bars and never
# overlaps them.  All text is black.
# -----------------------------
INK = 'black'

# upper band (axes fraction) reserved for the deviation lines
BAND_BOTTOM, BAND_TOP = 0.56, 0.965
DEV_LO, DEV_HI = 1e-6, 1e3      # deviation range mapped into the band


def dev_to_screen(v):
    """Map a deviation value (floored) to an axes-fraction y in the band."""
    vv = _v(v)
    t = (np.log10(vv) - np.log10(DEV_LO)) / (np.log10(DEV_HI) - np.log10(DEV_LO))
    return BAND_BOTTOM + np.clip(t, 0.0, 1.0) * (BAND_TOP - BAND_BOTTOM)


fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=180)
tfm_dev = blended_transform_factory(ax.transData, ax.transAxes)

# ---------- bars (lower region) ----------
bar_colors = [BLUE_LIGHT, BLUE, BLUE, BLUE, BLUE, BLUE]
ax.bar(pos, throughput, width=0.62, color=bar_colors,
       edgecolor=INK, linewidth=0.5, zorder=2, label='throughput (bar)')
ax.set_yscale('log')
ax.set_ylim(1e4, 1e7)           # headroom keeps bar tops below the line band
ax.set_ylabel('Throughput (filter steps/s, log)', fontsize=11, color=INK)
ax.tick_params(axis='y', colors=INK)
ax.axhline(throughput[1], color=BLUE, linestyle='--', linewidth=1.1,
           alpha=0.6, zorder=1)
for i in (0, 1, 3):
    v = throughput[i]
    ax.text(pos[i], v * 1.12, f'{v/1000:.0f}k', ha='center', va='bottom',
            fontsize=8.8, color=INK, zorder=3)
ax.text(5.0, throughput[1] * 1.12, 'plateau ≈ 250k', ha='center', va='bottom',
        fontsize=8.6, color=INK, zorder=3)
ratio = throughput[1] / throughput[0]
ax.annotate(f'≈{ratio:.1f}×', xy=(0.5, np.sqrt(throughput[0] * throughput[1])),
            ha='center', va='center', fontsize=13, fontweight='bold',
            color=INK, zorder=5)

# ---------- deviation lines (upper band, blended transform) ----------
p_dev = pos[1:]
sy_dX = [dev_to_screen(v) for v in dX[1:]]
sy_dZ = [dev_to_screen(v) for v in dZ[1:]]
line_dX, = ax.plot(p_dev, sy_dX, '-o', transform=tfm_dev, color=RED,
                   linewidth=2.2, markersize=8,
                   label=r'rel. $\Delta X$ (line)', zorder=5)
line_dZ, = ax.plot(p_dev, sy_dZ, '--s', transform=tfm_dev, color=AMBER,
                   linewidth=2.0, markersize=7,
                   label=r'rel. $\Delta Z$ (line)', zorder=5)

# right-axis tick gridlines + labels confined to the band (black)
for v in [1e-6, 1e-3, 1e0, 1e2]:
    y = dev_to_screen(v)
    is_thr = abs(v - 1e-3) < 1e-12
    ax.plot([0, 1], [y, y], transform=ax.transAxes,
            color=(GREEN if is_thr else '#CCCCCC'),
            linestyle=(':' if is_thr else '--'),
            linewidth=(1.5 if is_thr else 0.8),
            alpha=(0.9 if is_thr else 0.7),
            zorder=(3 if is_thr else 1), clip_on=False)
    lbl = r'$10^{-3}$ (fp16-$\varepsilon$)' if is_thr \
        else f'$10^{{{int(round(np.log10(v)))}}}$'
    ax.text(1.012, y, lbl, transform=ax.transAxes, ha='left', va='center',
            fontsize=8.2, color=INK)
ax.text(1.012, BAND_TOP + 0.006, 'rel. dev.',
        transform=ax.transAxes, ha='left', va='bottom',
        fontsize=8.6, color=INK)

# deviation value labels (black) on the outer side of each marker
for i in range(1, len(configs)):
    dx_ok = dX[i] > 1e-3
    dz_ok = dZ[i] > 1e-3
    if dx_ok:
        va = 'bottom' if (not dz_ok or dX[i] >= dZ[i]) else 'top'
        off = +0.028 if va == 'bottom' else -0.028
        ax.text(pos[i], dev_to_screen(dX[i]) + off, sci(dX[i]),
                transform=tfm_dev, ha='center', va=va, fontsize=8,
                color=INK, zorder=7)
    if dz_ok:
        va = 'bottom' if (not dx_ok or dZ[i] >= dX[i]) else 'top'
        off = +0.028 if va == 'bottom' else -0.028
        ax.text(pos[i], dev_to_screen(dZ[i]) + off, sci(dZ[i]),
                transform=tfm_dev, ha='center', va=va, fontsize=8,
                color=INK, zorder=7)

# ---------- shared x axis + region divider ----------
ax.axvline(SEP, color=GREY, linestyle='-', linewidth=1.0, alpha=0.7, zorder=1)
ax.set_xticks(pos)
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_xlim(-0.6, pos[-1] + 1.1)
ax.grid(axis='y', which='major', linestyle='--', alpha=0.25, zorder=0)

axt = blended_transform_factory(ax.transData, ax.transAxes)
ax.text(0.5, -0.105, 'Global compute precision', ha='center', va='top',
        fontsize=10.5, fontweight='bold', color=INK, transform=axt)
ax.text(4.5, -0.105, 'Component lowered below fp32', ha='center', va='top',
        fontsize=10.5, fontweight='bold', color=INK, transform=axt)

# ---------- legend ----------
bar_handle = plt.Rectangle((0, 0), 1, 1, facecolor=BLUE, edgecolor=INK,
                           linewidth=0.5)
ax.legend([bar_handle, line_dX, line_dZ],
          ['throughput (bar)', r'rel. $\Delta X$ (line)', r'rel. $\Delta Z$ (line)'],
          frameon=True, fancybox=False, edgecolor='#BBBBBB',
          loc='upper left', fontsize=9)

fig.suptitle('Precision trade-off in the fused SINS/EKF scan',
             fontsize=14, y=0.98, color=INK)
fig.tight_layout(rect=[0, 0.02, 0.91, 0.95])
fig.savefig(FIGURE_DIR / 'precision_tradeoff.pdf', bbox_inches='tight')
fig.savefig(FIGURE_DIR / 'precision_tradeoff.png', bbox_inches='tight', dpi=180)
plt.close(fig)
print(f'Saved: {FIGURE_DIR / "precision_tradeoff.pdf"}')
print(f'Saved: {FIGURE_DIR / "precision_tradeoff.png"}')