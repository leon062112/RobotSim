import numpy as np
import matplotlib.pyplot as plt

# Use fonts that work in most Linux environments.
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------
# Figure 1: trajectory benchmark
# -----------------------------
lengths = np.array([500, 1000, 5000])
torch_cpu = np.array([0.054153, 0.102441, 0.502842])
torch_gpu = np.array([0.004722, 0.010405, 0.057971])
speedup = torch_cpu / torch_gpu

x = np.arange(len(lengths))
width = 0.32

fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=180)
ax.bar(x - width / 2, torch_cpu, width, label='PyTorch CPU', color='#F58518')
ax.bar(x + width / 2, torch_gpu, width, label='PyTorch GPU', color='#54A24B')

ax.set_xlabel('Trajectory length (m)', fontsize=11)
ax.set_ylabel('Average runtime (s)', fontsize=11)
ax.set_title('PyTorch CPU vs GPU Trajectory Generation Runtime', fontsize=13, pad=12)
ax.set_xticks(x)
ax.set_xticklabels([f'{v} m' for v in lengths])
ax.grid(axis='y', linestyle='--', alpha=0.35)
ax.legend(frameon=False)

for values, offset in [(torch_cpu, -width / 2), (torch_gpu, width / 2)]:
    for i, v in enumerate(values):
        ax.text(i + offset, v + max(torch_cpu) * 0.015, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

for i, s in enumerate(speedup):
    ax.text(i, max(torch_cpu[i], torch_gpu[i]) + max(torch_cpu) * 0.08,
            f'{s:.1f}x faster', ha='center', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#BBBBBB', alpha=0.9))

fig.tight_layout()
fig.savefig('/workspace/denghaodong/RobotSim/定位/trajectory_runtime_comparison.png', bbox_inches='tight')
plt.close(fig)

# ----------------------------------
# Figure 2: complete workflow runtime
# ----------------------------------
modules = ['Trajectory\n+ CSV', 'EKF Fusion\nPositioning']
times = np.array([1.626583, 90.105915])
colors = ['#72B7B2', '#E45756']

fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=180)
bars = ax.bar(modules, times, color=colors, width=0.55)
ax.set_ylabel('Runtime (s)', fontsize=11)
ax.set_title('Runtime Breakdown of Complete PyTorch Workflow', fontsize=13, pad=12)
ax.grid(axis='y', linestyle='--', alpha=0.35)

for bar, v in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width() / 2, v + max(times) * 0.015, f'{v:.2f} s', ha='center', va='bottom', fontsize=10)

ratio = times[1] / times[0]
ax.text(0.5, max(times) * 0.72, f'EKF is about {ratio:.1f}x slower\nthan trajectory generation',
        ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#999999', alpha=0.9))

fig.tight_layout()
fig.savefig('/workspace/denghaodong/RobotSim/定位/workflow_runtime_breakdown.png', bbox_inches='tight')
plt.close(fig)

print('Saved: /workspace/denghaodong/RobotSim/定位/trajectory_runtime_comparison.png')
print('Saved: /workspace/denghaodong/RobotSim/定位/workflow_runtime_breakdown.png')
