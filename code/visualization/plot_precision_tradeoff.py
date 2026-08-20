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
# Single combo figure:
#   bars  (left  log) = throughput
#   lines (right log) = relative deviation dX / dZ (unitless)
# -----------------------------
fig, ax = plt.subplots(figsize=(11.4, 6.1), dpi=180)

bar_colors = [BLUE_LIGHT, BLUE, BLUE, BLUE, BLUE, BLUE]
ax.bar(pos, throughput, width=0.62, color=bar_colors,
       edgecolor='black', linewidth=0.5, zorder=2,
       label='throughput (bars, left)')
ax.set_yscale('log')
ax.set_ylim(1e4, 4e5)
ax.set_ylabel('Throughput (filter steps/s, log)', fontsize=11, color=BLUE)
ax.tick_params(axis='y', colors=BLUE)
ax.axhline(throughput[1], color=BLUE, linestyle='--', linewidth=1.1,
           alpha=0.55, zorder=1)
# label only the bars that carry information (fp64 slow, fp32 ref, tf32 slow);
# the plateau trio (~250k) is annotated once to avoid top clutter
for i in (0, 1, 3):
    v = throughput[i]
    ax.text(pos[i], v * 1.12, f'{v/1000:.0f}k', ha='center', fontsize=8.8,
            color=BLUE, zorder=3)
ax.text(5.0, throughput[1] * 1.12, 'plateau ≈ 250k', ha='center', fontsize=8.5,
        color=BLUE, zorder=3)
ratio = throughput[1] / throughput[0]
ax.annotate(f'≈{ratio:.1f}×', xy=(0.5, np.sqrt(throughput[0] * throughput[1])),
            ha='center', va='center', fontsize=13, fontweight='bold',
            color=RED, zorder=5)

# --- relative deviation lines (right axis) ---
ax2 = ax.twinx()
p_dev = pos[1:]
dXv = np.array([_v(v) for v in dX[1:]])
dZv = np.array([_v(v) for v in dZ[1:]])
ax2.plot(p_dev, dXv, '-o', color=RED, linewidth=2.2, markersize=8,
         label=r'rel. $\Delta X$ (line, right)', zorder=6)
ax2.plot(p_dev, dZv, '--s', color=AMBER, linewidth=2.0, markersize=7,
         label=r'rel. $\Delta Z$ (line, right)', zorder=6)
ax2.set_yscale('log')
ax2.set_ylim(1e-6, 1e3)
ax2.set_ylabel('Relative deviation vs fp32  (unitless, log)', fontsize=11)
ax2.axhline(1e-3, color=GREEN, linestyle=':', linewidth=1.4, zorder=4)
ax2.text(pos[-1] + 0.30, 1e-3, r'$10^{-3}$ (fp16-$\varepsilon$ grade)',
         ha='left', va='center', fontsize=8.5, color=GREEN)
for i in range(1, len(configs)):
    if dX[i] > 1e-3:
        ax2.text(pos[i] - 0.16, dX[i] * 1.8, sci(dX[i]), ha='center', fontsize=8,
                 color=RED, zorder=7)
    if dZ[i] > 1e-3:
        ax2.text(pos[i] + 0.18, dZ[i] * 1.8, sci(dZ[i]), ha='center', fontsize=8,
                 color=AMBER, zorder=7)

ax2.set_zorder(ax.get_zorder() + 1)
ax.patch.set_visible(False)

# --- region divider + bottom brackets (free the top for the legend) ---
ax.axvline(SEP, color=GREY, linestyle='-', linewidth=1.0, alpha=0.6, zorder=1)
axt = blended_transform_factory(ax.transData, ax.transAxes)
ax.text(0.5, -0.06, 'Global compute precision', ha='center', va='top',
        fontsize=10.5, fontweight='bold', color='#444444', transform=axt)
ax.text(4.5, -0.06, 'Component lowered below fp32', ha='center', va='top',
        fontsize=10.5, fontweight='bold', color='#444444', transform=axt)

ax.set_xticks(pos)
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_xlim(-0.6, pos[-1] + 0.7)
ax.grid(axis='y', which='major', linestyle='--', alpha=0.25, zorder=0)

# --- legend, upper left ---
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=True, fancybox=False, edgecolor='#BBBBBB',
          loc='upper left', fontsize=9)

fig.suptitle('Precision trade-off in the fused SINS/EKF scan', fontsize=14, y=0.99)
fig.tight_layout(rect=[0, 0.12, 1, 0.96])
fig.savefig(FIGURE_DIR / 'precision_tradeoff.pdf', bbox_inches='tight')
fig.savefig(FIGURE_DIR / 'precision_tradeoff.png', bbox_inches='tight', dpi=180)
plt.close(fig)
print(f'Saved: {FIGURE_DIR / "precision_tradeoff.pdf"}')
print(f'Saved: {FIGURE_DIR / "precision_tradeoff.png"}')