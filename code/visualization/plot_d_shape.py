import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = REPO_ROOT / 'data' / 'figures'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# 已验证的 CVD-safe 分类色（dataviz reference palette slots）
BLUE, AQUA, YELLOW, GREEN = '#2a78d6', '#1baf7a', '#eda100', '#008300'
INK, MUTED = '#0b0b0b', '#898781'

SERIES = [
    ('Trident (ours)', GREEN, 'o', 2.4, 'triton'),
    ('torch eager', BLUE, 's', 1.6, 'eager'),
    ('torch.compile', AQUA, '^', 1.6, 'compile'),
    ('torch-kf', YELLOW, 'D', 1.6, 'torch_kf'),
]

def _read(name):
    return json.load(open(REPO_ROOT / f'data/results/{name}.json'))

djson = _read('a100_scaling_d')
drows = djson['d_sweep']
mrows = _read('a100_scaling_m')['m_sweep']
_gpu = djson.get('hardware', {}).get('gpu', '')
dx = [r['d'] for r in drows]
mx = [r['m'] for r in mrows]

fig, (axd, axm) = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=200, sharey=True)

for name, color, mkr, lw, key in SERIES:
    axd.plot(dx, [r[key] for r in drows], '-', marker=mkr, color=color,
             linewidth=lw, markersize=6, label=name, markeredgecolor='white',
             markeredgewidth=0.8, zorder=3)
    axm.plot(mx, [r[key] for r in mrows], '-', marker=mkr, color=color,
             linewidth=lw, markersize=6, markeredgecolor='white',
             markeredgewidth=0.8, zorder=3)

for ax, xvals, xlabel in [(axd, dx, 'State dimension d (m=3)'),
                          (axm, mx, 'Observation dimension m (d=15)')]:
    ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontsize=10.5, color=INK)
    ax.set_xticks(xvals)
    ax.set_yticks([2e3, 1e4, 1e5, 3e5])
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.grid(axis='y', which='major', linestyle='--', linewidth=0.7, color='#e1e0d9', zorder=0)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for sp in ax.spines.values():
        sp.set_color('#c3c2b7')

axd.set_ylabel('Throughput (filter steps/s, log)', fontsize=10.5, color=INK)
axd.set_title('(a) vs state dimension d', fontsize=10.5, color=INK)
axm.set_title('(b) vs observation dimension m', fontsize=10.5, color=INK)

# 直接标注 Trident 的领先量级
axd.annotate('60–90×', xy=(6, 189337), xytext=(6, 350000), fontsize=11,
             fontweight='bold', color=GREEN)
axm.annotate('30–100×', xy=(2, 314867), xytext=(2, 500000), fontsize=11,
             fontweight='bold', color=GREEN)
axm.text(12, 1600, 'baselines ≈ 2–3.5k', fontsize=8, color=MUTED, ha='right')

handles, labels = axd.get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, -0.06), fontsize=9, handlelength=1.4)

fig.suptitle(f'Throughput vs input shape ({_gpu}, N=2000, fp32, batch=1)', fontsize=12.5, color=INK, y=1.02)
fig.tight_layout(rect=[0, 0.02, 1, 0.99])
fig.savefig(FIGURE_DIR / 'd_shape_throughput.pdf', bbox_inches='tight')
fig.savefig(FIGURE_DIR / 'd_shape_throughput.png', bbox_inches='tight', dpi=200)
plt.close(fig)
print('Saved: data/figures/d_shape_throughput.pdf / .png')