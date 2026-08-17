import torch
import triton
import triton.language as tl


@triton.jit
def probe_kernel(out_ptr):
    # atan2 / sin / cos / sqrt availability via libdevice
    x = tl.full((1,), 0.5, tl.float64)
    y = tl.full((1,), 0.3, tl.float64)
    a = tl.extra.cuda.libdevice.atan2(x, y)
    s = tl.extra.cuda.libdevice.sin(a)
    c = tl.extra.cuda.libdevice.cos(a)
    r = tl.extra.cuda.libdevice.sqrt(x * x + y * y)
    tl.store(out_ptr + tl.arange(0, 1), a + s + c + r)


@triton.jit
def scan_kernel(inp_ptr, out_ptr, N, B: tl.constexpr):
    # single-program runtime loop carrying a BxB fp64 state, using tl.dot
    offs = tl.arange(0, B)
    P = tl.zeros((B, B), dtype=tl.float64) + (offs[:, None] == offs[None, :]).to(tl.float64)
    acc = tl.zeros((1,), dtype=tl.float64)
    for k in range(0, N):
        v = tl.load(inp_ptr + k)
        F = (offs[:, None] == offs[None, :]).to(tl.float64) * v
        P = tl.dot(F, P)            # 16x16 fp64 matmul in loop
        acc += tl.sum(P)
    tl.store(out_ptr + tl.arange(0, 1), acc)


def main():
    out = torch.zeros(1, dtype=torch.float64, device='cuda')
    try:
        probe_kernel[(1,)](out)
        torch.cuda.synchronize()
        print('atan2/sin/cos/sqrt fp64 OK ->', out.item())
    except Exception as e:
        print('math probe FAILED:', repr(e)[:250])

    N = 2000
    inp = torch.ones(N, dtype=torch.float64, device='cuda') * 1.0001
    out2 = torch.zeros(1, dtype=torch.float64, device='cuda')
    try:
        scan_kernel[(1,)](inp, out2, N, 16)
        torch.cuda.synchronize()
        print('runtime-loop scan w/ 16x16 tl.dot OK ->', out2.item())
    except Exception as e:
        print('scan FAILED:', repr(e)[:250])


if __name__ == '__main__':
    main()
