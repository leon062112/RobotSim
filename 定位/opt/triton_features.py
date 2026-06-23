import torch
import triton
import triton.language as tl


@triton.jit
def feat_kernel(out_ptr):
    i = tl.arange(0, 16)
    j = tl.arange(0, 16)
    A = (i[:, None] * 16 + j[None, :]).to(tl.float64)
    At = tl.trans(A)                       # transpose
    rowsum = tl.sum(A, axis=1)             # reduce axis=1 -> (16,)
    tl.store(out_ptr + i, rowsum)
    tl.store(out_ptr + 16 + i[:, None] * 16 + j[None, :], At)


def main():
    out = torch.zeros(16 + 256, dtype=torch.float64, device='cuda')
    try:
        feat_kernel[(1,)](out)
        torch.cuda.synchronize()
        A = (torch.arange(16, device='cuda')[:, None] * 16 +
             torch.arange(16, device='cuda')[None, :]).double()
        rs_ref = A.sum(1)
        at_ref = A.T
        rs = out[:16]
        at = out[16:].reshape(16, 16)
        print('tl.trans OK err=', (at - at_ref).abs().max().item())
        print('tl.sum(axis=1) OK err=', (rs - rs_ref).abs().max().item())
    except Exception as e:
        print('FEATURE FAILED:', repr(e)[:300])


if __name__ == '__main__':
    main()
