"""
「吞吐 vs 输入 shape (d, m)」论文终版插图 —— 每个硬件一张单轴图：
  横轴 = (d, m) 组合配置点，按规模（先 d 后 m 的字典序）从左到右递增排列：
  d 扫描（m=3，代表性子集）与 m 扫描（d=15）按其 d 的位置穿插合并，
  例如 (6,3),(9,3),(15,1),(15,2),(15,3),(21,3),(30,3)。
纵轴 log 吞吐（throughput, steps/s）。无图内标题、无解释性文字、无领先倍数标注；
5 条方法线的图例统一放在绘图区上方。论文 caption 承载全部说明。

5 条方法线（与 benchmark_h20_shape.py 输出字段一一对应）：
  Trident (ours) / PrefixScan / torch eager / torch.compile / torch-kf
  （PrefixScan 即 Särkkä & García-Fernández 时间并行前缀扫描，图例与正文同名）

调色板为 dataviz 校验通过的分类色（light surface，worst adjacent CVD ΔE = 24.2 ≥ 12，
marker 形状作为 secondary encoding）。

输出：data/figures/{a100,h20}_shape_throughput.{pdf,png}
（对 data/results/ 下每个存在 {tag}_scaling_d.json 的硬件各生成一张；
 生成后需复制到 paper/latex/figures/5-eval-shape-{tag}.pdf 供论文引用。）

用法：python code/visualization/plot_h20_shape.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = REPO_ROOT / 'data' / 'figures'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# 代表性 (d, m) 配置：从左到右严格递增。
# d: 6=δp+δv 退化 / 9=+姿态 / 15=本文 SINS/EKF / 21=扩展 INS / 30=实际上限
D_SELECT = [6, 9, 15, 21, 30]
M_SELECT = [1, 2, 3]

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# dataviz 校验通过的分类色槽位（勿改顺序换色；如需替换先跑 validate_palette.py）
BLUE, AQUA, YELLOW, GREEN, VIOLET = '#2a78d6', '#1baf7a', '#eda100', '#008300', '#4a3aa7'
INK, MUTED = '#0b0b0b', '#898781'

SERIES = [
    ('Trident (ours)',     GREEN,  'o', 2.4, 'triton'),
    ('PrefixScan',         VIOLET, 'v', 1.6, 'prefix'),
    ('torch eager',        BLUE,   's', 1.6, 'eager'),
    ('torch.compile',      AQUA,   '^', 1.6, 'compile'),
    ('torch-kf',           YELLOW, 'D', 1.6, 'torch_kf'),
]

Y_TICKS = [3e3, 1e4, 3e4, 1e5, 3e5]
Y_LIM = (1.5e3, 9.0e5)   # 上界容纳 H20 m=1 峰值 (~7.3e5)


def _read(name):
    return json.load(open(REPO_ROOT / f'data/results/{name}.json'))


def _tags():
    """所有存在结果文件的硬件标签（a100 在前，保持论文上图/下图顺序）。"""
    return [t for t in ('a100', 'h20')
            if (REPO_ROOT / f'data/results/{t}_scaling_d.json').exists()]


def plot_one(tag):
    drows = {r['d']: r for r in _read(f'{tag}_scaling_d')['d_sweep']}
    mrows = {r['m']: r for r in _read(f'{tag}_scaling_m')['m_sweep']}

    # (d,m) 配置点按 (d, m) 字典序递增穿插：d 扫描(m=3) 与 m 扫描(d=15)
    # 在 d=15 锚点处汇合成 (15,1),(15,2),(15,3) 的局部递增段。
    configs = sorted([(d, 3) for d in D_SELECT if d in drows] +
                     [(15, m) for m in M_SELECT if m in mrows and m != 3])
    rows = [drows[d] if m == 3 else mrows[m] for d, m in configs]

    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=200)

    x = np.arange(len(configs))
    for name, color, mkr, lw, key in SERIES:
        ax.plot(x, [r[key] for r in rows], '-', marker=mkr, color=color,
                linewidth=lw, markersize=5.5, markeredgecolor='white',
                markeredgewidth=0.7, label=name, zorder=3)

    ax.set_yscale('log')
    ax.set_ylim(Y_LIM)
    ax.set_xticks(x)
    ax.set_xticklabels([f'$({d},{m})$' for d, m in configs], fontsize=9)
    ax.set_yticks(Y_TICKS)
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.set_xlabel('Input shape $(d, m)$  [state dim $d$, observation dim $m$]',
                  fontsize=9.5, color=INK, labelpad=4)
    ax.set_ylabel('Throughput (filter steps/s)', fontsize=9.5, color=INK)
    ax.grid(axis='y', which='major', linestyle='--', linewidth=0.6,
            color='#e1e0d9', zorder=0)
    ax.minorticks_off()
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.tick_params(axis='x', colors=INK)
    for sp in ax.spines.values():
        sp.set_color('#c3c2b7')

    # 图例：绘图区上方，单行 5 项，无边框
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='upper center',
               bbox_to_anchor=(0.5, 1.035), ncol=5, fontsize=8.5,
               handlelength=1.5, columnspacing=1.4, handletextpad=0.5)

    # 显式边距（tight_layout 与 fig.legend 不兼容）：顶部留图例行
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.17, top=0.87)
    fig.savefig(FIGURE_DIR / f'{tag}_shape_throughput.pdf', bbox_inches='tight')
    fig.savefig(FIGURE_DIR / f'{tag}_shape_throughput.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'Saved: data/figures/{tag}_shape_throughput.pdf / .png  '
          f'configs={configs}')


for tag in _tags():
    plot_one(tag)
