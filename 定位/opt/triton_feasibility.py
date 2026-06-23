import torch
import triton
import triton.language as tl


@triton.jit
def dot_fp64_kernel(a_ptr, b_ptr, c_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    a = tl.load(a_ptr + offs[:, None] * N + offs[None, :])
    b = tl.load(b_ptr + offs[:, None] * N + offs[None, :])
    c = tl.dot(a, b)
    tl.store(c_ptr + offs[:, None] * N + offs[None, :], c)


def test_dot():
    N = 16
    a = torch.randn(N, N, dtype=torch.float64, device='cuda')
    b = torch.randn(N, N, dtype=torch.float64, device='cuda')
    c = torch.empty(N, N, dtype=torch.float64, device='cuda')
    try:
        dot_fp64_kernel[(1,)](a, b, c, N)
        torch.cuda.synchronize()
        ref = a @ b
        print('fp64 tl.dot OK, max_err=', (c - ref).abs().max().item())
    except Exception as e:
        print('fp64 tl.dot FAILED:', repr(e)[:300])


@triton.jit
def manual_matmul_kernel(a_ptr, b_ptr, c_ptr, N: tl.constexpr):
    # 单 program 手工三重循环 matmul (无 tensor core)，验证 fp64 标量算路径
    i = tl.arange(0, N)
    j = tl.arange(0, N)
    acc = tl.zeros((N, N), dtype=tl.float64)
    for kk in range(N):
        a_col = tl.load(a_ptr + i * N + kk)          # (N,)
        b_row = tl.load(b_ptr + kk * N + j)          # (N,)
        acc += a_col[:, None] * b_row[None, :]
    tl.store(c_ptr + i[:, None] * N + j[None, :], acc)


def test_manual():
    N = 15
    a = torch.randn(N, N, dtype=torch.float64, device='cuda')
    b = torch.randn(N, N, dtype=torch.float64, device='cuda')
    c = torch.empty(N, N, dtype=torch.float64, device='cuda')
    try:
        manual_matmul_kernel[(1,)](a, b, c, N)
        torch.cuda.synchronize()
        ref = a @ b
        print('manual fp64 matmul OK, max_err=', (c - ref).abs().max().item())
    except Exception as e:
        print('manual fp64 matmul FAILED:', repr(e)[:300])


if __name__ == '__main__':
    test_dot()
    test_manual()
