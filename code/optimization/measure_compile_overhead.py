"""精度特化的一次性编译开销测量 (cold JIT per precision plan).

用法（有 GPU 环境，从仓库根目录执行）:
    python code/optimization/measure_compile_overhead.py

方法: 每个 plan 在全新的 TRITON_CACHE_DIR 中首次运行全量轨迹 -> cold (含 JIT 编译);
      同进程内第二次运行 -> hot (磁盘缓存命中, 仅执行)。两次的数据加载与 kernel
      执行相同, 相减即可隔离出 JIT 编译时间:
          compile_s_est = cold - hot
产出: data/results/compile_overhead_{tag}.json   ({tag} = a100 / h20, 按 GPU 名自适应)
对应论文 §5 Overhead Analysis 的 Table (tab:overhead-compile)。
"""
import os
import sys
import json
import subprocess
import tempfile

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
os.chdir(os.path.join(_here, '..', '..'))

import torch  # noqa: E402
from benchmark_a100 import get_hardware  # noqa: E402

CSV = 'data/trajectory/PipeRobot_Trajectory.csv'
PLANS = ['fp64', 'fp32', 'tf32']
N_STEPS = 166667
N_REPS = 3

TEMPLATE = """
import time, sys
sys.path.insert(0, '@OPT@')
from ekf_v5 import run_ekf_v5
CSV = '@CSV@'
t0 = time.time()
run_ekf_v5(CSV, n_steps=@N@, batch=1, precision='@P@', verbose=False)
t1 = time.time()
run_ekf_v5(CSV, n_steps=@N@, batch=1, precision='@P@', verbose=False)
t2 = time.time()
print('COLD %.3f' % (t1 - t0))
print('HOT %.3f' % (t2 - t1))
"""


def main():
    name = torch.cuda.get_device_name(0)
    tag = 'a100' if 'A100' in name else ('h20' if 'H20' in name else 'gpu')
    out = {
        'hardware': get_hardware(),
        'n_steps': N_STEPS,
        'batch': 1,
        'reps_warm': N_REPS,
        'note': ('cold = first full run in a fresh TRITON_CACHE_DIR (JIT compile + exec); '
                 'hot = second run, disk-cache hit; compile_s_est = cold - hot '
                 '(data loading and kernel exec identical in both).'),
        'plans': {},
    }
    total = 0.0
    for plan in PLANS:
        cache = tempfile.mkdtemp(prefix='triton_cold_')
        env = dict(os.environ, TRITON_CACHE_DIR=cache)
        code = (TEMPLATE.replace('@OPT@', _here).replace('@CSV@', CSV)
                .replace('@N@', str(N_STEPS)).replace('@P@', plan))
        r = subprocess.run([sys.executable, '-c', code], env=env,
                           capture_output=True, text=True)
        vals = {}
        for line in r.stdout.splitlines():
            if line.startswith(('COLD', 'HOT')):
                k, v = line.split()
                vals[k.lower()] = float(v)
        if r.returncode != 0 or len(vals) < 2:
            print(f'{plan}: FAILED\n{r.stderr[-800:]}')
            out['plans'][plan] = {'error': r.stderr[-500:]}
            continue
        vals['compile_s_est'] = round(vals['cold'] - vals['hot'], 3)
        total += vals['compile_s_est']
        out['plans'][plan] = vals
        print(f'{plan}: {vals}')
    out['total_compile_s_est'] = round(total, 3)
    path = f'data/results/compile_overhead_{tag}.json'
    json.dump(out, open(path, 'w'), indent=2)
    print(f'total compile {total:.1f}s -> {path}')


if __name__ == '__main__':
    main()
