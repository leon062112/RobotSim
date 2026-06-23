"""SM 利用率 profiling 辅助脚本: 单独跑一次指定 kernel, 供 ncu/nsys 采样.
用法:
  python opt/profile_sm.py v4 <N>      # 单 block (v4 fp32)
  python opt/profile_sm.py v5 <N> <B>  # 多 block batch (v5 fp32)
"""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
os.chdir(here.rsplit('/opt', 1)[0])
import torch

mode = sys.argv[1] if len(sys.argv) > 1 else 'v4'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
B = int(sys.argv[3]) if len(sys.argv) > 3 else 78

if mode == 'v4':
    from ekf_v4 import run_ekf_v4
    run_ekf_v4(n_steps=N, precision='fp32', verbose=True)
else:
    from ekf_v5 import run_ekf_v5
    run_ekf_v5(n_steps=N, batch=B, precision='fp32', verbose=True)
torch.cuda.synchronize()
