"""
SM Profiling Script — v7 (扩展版)
用于在 ncu/nsys 下采集 v5/v7 batch kernel 的 SM 级性能指标。

用法:
  # 单轨迹 profiling
  python profile_sm_v7.py v4 N=20000

  # batch profiling (指定 B)
  python profile_sm_v7.py v5 N=20000 B=78

  # 多 stream profiling
  python profile_sm_v7.py v7 N=20000 B=128 streams=4

  # 在 ncu 下运行:
  ncu --set full --kernel-name ekf_mega_batch_kernel \
      python opt/profile_sm_v7.py v5 N=20000 B=78

  # 在 nsys 下运行 (timeline):
  nsys profile --trace=cuda,nvtx \
      python opt/profile_sm_v7.py v5 N=20000 B=78

  # 批量采集 (多个 B 值):
  for B in 1 8 78 128 256 312; do
    ncu --set full --kernel-name ekf_mega_batch_kernel \
        -o profile_B${B} \
        python opt/profile_sm_v7.py v5 N=20000 B=${B}
  done
"""
import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import torch


def parse_args():
    """Parse command line: mode=v4|v5|v7 N=20000 B=78 streams=1"""
    args = {'mode': 'v4', 'N': 20000, 'B': 78, 'streams': 1}
    for arg in sys.argv[1:]:
        if '=' in arg:
            k, v = arg.split('=', 1)
            if k in ('N', 'B', 'streams'):
                args[k] = int(v)
            else:
                args[k] = v
        elif arg in ('v4', 'v5', 'v7'):
            args['mode'] = arg
    return args


def main():
    args = parse_args()
    mode = args['mode']
    N = args['N']
    B = args['B']
    n_streams = args['streams']

    print(f"[profile_sm_v7] mode={mode} N={N} B={B} streams={n_streams}")
    print(f"[profile_sm_v7] GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"[profile_sm_v7] SMs: {props.multi_processor_count} "
          f"max_threads/SM: {props.max_threads_per_multi_processor} "
          f"regs/SM: {props.registers_per_multiprocessor}")

    if mode == 'v4':
        from ekf_v4 import run_ekf_v4
        _, _, _, _, m = run_ekf_v4(n_steps=N, precision='fp32', verbose=False)
        print(f"[profile_sm_v7] v4 done: {m['throughput_steps_per_s']:.0f} steps/s")
        # Force a dummy warmup + real run for ncu capture (ncu captures last kernel)
        from ekf_v4 import ekf_mega_kernel_p as kernel_fn
        from ekf_v4 import _PREC
        DT, IP, torch_dt = _PREC['fp32']
        # Run one more time for ncu to capture
        import numpy as np
        data = np.loadtxt('PipeRobot_Trajectory.csv', delimiter=',', skiprows=1)
        gyro = torch.from_numpy(data[:N, 7:10]).to('cuda', torch_dt).contiguous()
        accel = torch.from_numpy(data[:N, 10:13]).to('cuda', torch_dt).contiguous()
        pos_out = torch.zeros(N, 3, dtype=torch_dt, device='cuda').contiguous()
        vel_out = torch.zeros(N, 3, dtype=torch_dt, device='cuda').contiguous()
        odom1 = torch.from_numpy(data[:N, 13]).to('cuda', torch_dt).contiguous()
        odom2 = torch.from_numpy(data[:N, 14]).to('cuda', torch_dt).contiguous()
        qinit = torch.tensor([1.,0.,0.,0.], dtype=torch_dt, device='cuda')
        qdiag = torch.tensor([1e-6]*3+[1e-5]*3+[1e-4]*3+[1e-8]*3+[1e-7]*3, dtype=torch_dt, device='cuda')
        kernel_fn[(1,)](gyro, accel, odom1, odom2, qinit, qdiag,
                        pos_out, vel_out, N, 0.01, 9.81, 1e-4, 1e-3, 0.01, 1e12, DT, IP)
        torch.cuda.synchronize()
    elif mode == 'v5':
        from ekf_v5 import run_ekf_v5
        _, _, _, _, m = run_ekf_v5(n_steps=N, batch=B, precision='fp32', verbose=False)
        print(f"[profile_sm_v7] v5 B={B} done: {m['throughput_steps_per_s']:.0f} steps/s")
    elif mode == 'v7':
        from ekf_v7_concurrency import run_ekf_v7_events
        _, _, _, _, m = run_ekf_v7_events(n_steps=N, batch=B, precision='fp32',
                                            n_streams=n_streams, verbose=False)
        print(f"[profile_sm_v7] v7 B={B} streams={n_streams} done: "
              f"{m['throughput_steps_per_s']:.0f} steps/s")

    torch.cuda.synchronize()
    print("[profile_sm_v7] complete.")


if __name__ == '__main__':
    main()