"""
EKF eager-mode profiling (TODO #1, #5)
使用 torch.profiler 分析基线主循环，统计：
- 各算子耗时占比 (self CPU/CUDA time)
- 算子调用次数 (kernel/op count proxy)
- 导出 chrome trace 供 Nsight/chrome://tracing 查看

用法: python opt/profile_baseline.py --device cpu --n 2000
"""
import os
import argparse
import torch
from torch.profiler import profile, ProfilerActivity, schedule
from ekf_baseline import run_ekf_baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--csv', default='PipeRobot_Trajectory.csv')
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()

    tag = args.tag or f'v0_{args.device}_n{args.n}'
    activities = [ProfilerActivity.CPU]
    if args.device == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    # warm up (esp. for cuda context / allocator)
    run_ekf_baseline(args.csv, device=args.device, n_steps=64, verbose=False)

    with profile(activities=activities, record_shapes=False,
                 profile_memory=True, with_stack=False) as prof:
        run_ekf_baseline(args.csv, device=args.device, n_steps=args.n, verbose=False)

    sort_key = 'self_cuda_time_total' if args.device == 'cuda' else 'self_cpu_time_total'
    table = prof.key_averages().table(sort_by=sort_key, row_limit=30)
    print(table)

    os.makedirs('opt/profiles', exist_ok=True)
    trace_path = f'opt/profiles/{tag}_trace.json'
    prof.export_chrome_trace(trace_path)

    txt_path = f'opt/profiles/{tag}_summary.txt'
    with open(txt_path, 'w') as f:
        f.write(table)

    # Aggregate op/kernel statistics
    evts = prof.key_averages()
    total_calls = sum(e.count for e in evts)
    print(f"\n[profile] device={args.device} n={args.n}")
    print(f"  distinct ops: {len(evts)} | total op invocations: {total_calls}")
    print(f"  per-step op invocations: {total_calls/(args.n-1):.1f}")
    print(f"  trace -> {trace_path}")
    print(f"  summary -> {txt_path}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)).rsplit('/opt', 1)[0])
    main()
