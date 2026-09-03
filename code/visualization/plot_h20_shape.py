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
BLUE, AQUA, YELLOW, GREEN, VERMILLION = '#2a78d6', '#1baf7a', '#eda100', '#008300', '#d55e00'
INK, MUTED = '#0b0b0b', '#898781'

SERIES = [
    ('Trident (ours)', GREEN, 'o', 2.4, 'triton'),
    ('Särkkä prefix-sum', VERMILLION, 'v', 1.6, 'prefix'),
    ('torch eager', BLUE, 's', 1.6, 'eager'),
    ('torch.compile', AQUA, '^', 1.6, 'compile'),
    ('torch-kf', YELLOW, 'D', 1.6, 'torch_kf'),
]

def _read(name):
    return json.load(open(REPO_ROOT / f'data/results/{name}.json'))

djson = _read('h20_scaling_d')
mjson = _read('h20_scaling_m')

drows = djson['d_sweep']
mrows = mjson['m_sweep']
_gpu = djson.get('hardware', {}).get('gpu', '')

# 合并 (d,m) 组合轴：d组 (m=3) + m组 (d=15)，共享锚点 (15,3)
x_labels = []
data_map = {}

# 1. d 组: (6,3) ... (30,3)
for r in drows:
    label = f"({r['d']},3)"
    x_labels.append(label)
    data_map[label] = r

# 2. m 组 (只加入 m=1, m=2; m=3 已在 d 组的 (15,3))
for r in mrows:
    if r['m'] == 3:
        continue
    label = f"(15,{r['m']})"
    x_labels.append(label)
    data_map[label] = r

x = np.arange(len(x_labels))

fig, ax = plt.subplots(figsize=(9.2, 4.2), dpi=200)

for name, color, mkr, lw, key in SERIES:
    y = [data_map[label][key] for label in x_labels]
    ax.plot(x, y, '-', marker=mkr, color=color, linewidth=lw,
            markersize=6, label=name, markeredgecolor='white',
            markeredgewidth=0.8, zorder=3)

# 绘制分隔线和组标签
split_idx = len(drows) - 0.5
ax.axvline(split_idx, color='#c3c2b7', linestyle='--', linewidth=1.0, zorder=1)
ax.text((len(drows)-1)/2.0, 1.4e6, 'State dimension sweep (m=3)', fontsize=9.5, color=INK, ha='center', weight='semibold')
ax.text(len(drows) + 0.5, 1.4e6, 'Observation dim (d=15)', fontsize=9.5, color=INK, ha='center', weight='semibold')

# 坐标轴设置
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=9.5, rotation=0)
ax.set_xlabel('Input shape $(d, m)$ [state dim $d$, observation dim $m$]', fontsize=10.5, color=INK)
ax.set_ylabel('Throughput (filter steps/s, log)', fontsize=10.5, color=INK)

# y 轴刻度
yticks = [3e3, 1e4, 1e5, 3e5, 1e6]
ax.set_yticks(yticks)
ax.set_ylim(2e3, 2e6)
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
ax.grid(axis='y', which='major', linestyle='--', linewidth=0.7, color='#e1e0d9', zorder=0)
ax.tick_params(colors=MUTED, labelsize=8.5)
for sp in ax.spines.values():
    sp.set_color('#c3c2b7')

# 标注领先倍数
triton_y = [data_map[label]['triton'] for label in x_labels]
ax.annotate('Trident up to 2.4× vs Särkkä\n& 45–160× vs tensor baselines',
            xy=(8, triton_y[8]), xytext=(3.5, 650000),
            fontsize=9.5, color=GREEN, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.2))

ax.text(x_labels.index('(6,3)'), 1800, 'Tensor baselines ≈ 3–4.5k steps/s', fontsize=8.5, color=MUTED, ha='left')

ax.set_title(f'End-to-End Filter Throughput across $(d, m)$ Configurations ({_gpu}, N=2000, fp32, B=1)',
             fontsize=11.5, color=INK, pad=12)

fig.legend(frameon=False, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.06), fontsize=9, handlelength=1.4)
fig.tight_layout(rect=[0, 0.02, 1, 0.99])

fig.savefig(FIGURE_DIR / 'h20_shape_throughput.pdf', bbox_inches='tight')
fig.savefig(FIGURE_DIR / 'h20_shape_throughput.png', bbox_inches='tight', dpi=200)
plt.close(fig)
print('Saved: data/figures/h20_shape_throughput.pdf / .png')
