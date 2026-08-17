"""
Triton building-block 验证: 用 padded-16 tl.dot 表示所有矩阵, 通过 mask 构造/抽取标量。
目标: 确认能在单 kernel 内完成 EKF 单步所需的全部原语。
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def blocks_kernel(gyro_ptr, out_ptr, dt):
    i = tl.arange(0, 16)
    j = tl.arange(0, 16)
    eye = (i[:, None] == j[None, :]).to(tl.float64)

    # load 3 scalars (gyro x,y,z) via masked load
    wx = tl.load(gyro_ptr + 0)
    wy = tl.load(gyro_ptr + 1)
    wz = tl.load(gyro_ptr + 2)
    tx = wx * dt
    ty = wy * dt
    tz = wz * dt

    # build 4x4 theta_mat inside 16x16 via index masks
    r = i[:, None]
    c = j[None, :]
    tm = tl.zeros((16, 16), dtype=tl.float64)
    # row0: [0,-tx,-ty,-tz]; row1:[-tx,0,-tz,-ty]; row2:[-ty,-tz,0,-tx]; row3:[-tz,-ty,-tx,0]
    tm += tl.where((r == 0) & (c == 1), -tx, 0.0)
    tm += tl.where((r == 0) & (c == 2), -ty, 0.0)
    tm += tl.where((r == 0) & (c == 3), -tz, 0.0)
    tm += tl.where((r == 1) & (c == 0), -tx, 0.0)
    tm += tl.where((r == 1) & (c == 2), -tz, 0.0)
    tm += tl.where((r == 1) & (c == 3), -ty, 0.0)
    tm += tl.where((r == 2) & (c == 0), -ty, 0.0)
    tm += tl.where((r == 2) & (c == 1), -tz, 0.0)
    tm += tl.where((r == 2) & (c == 3), -tx, 0.0)
    tm += tl.where((r == 3) & (c == 0), -tz, 0.0)
    tm += tl.where((r == 3) & (c == 1), -ty, 0.0)
    tm += tl.where((r == 3) & (c == 2), -tx, 0.0)

    theta_norm = libdevice.sqrt(tx * tx + ty * ty + tz * tz)
    coef = libdevice.sin(theta_norm / 2) / theta_norm
    cosv = libdevice.cos(theta_norm / 2)
    # q_update only valid in top-left 4x4
    qmask = ((r < 4) & (c < 4)).to(tl.float64)
    q_update = (cosv * eye + coef * tm) * qmask

    # matmul test: q_update @ q_update  (16x16 fp64 tl.dot)
    prod = tl.dot(q_update, q_update)

    # extract a scalar via masked sum (e.g. element [0,1])
    s01 = tl.sum(tl.where((r == 0) & (c == 1), prod, 0.0))

    tl.store(out_ptr + i[:, None] * 16 + j[None, :], prod)
    tl.store(out_ptr + 256, s01)


def main():
    dev = 'cuda'
    gyro = torch.tensor([0.01, -0.005, 0.008], dtype=torch.float64, device=dev)
    out = torch.zeros(257, dtype=torch.float64, device=dev)
    dt = 0.01
    blocks_kernel[(1,)](gyro, out, dt)
    torch.cuda.synchronize()

    # reference in torch
    tx, ty, tz = (gyro * dt).tolist()
    tm = torch.tensor([
        [0, -tx, -ty, -tz], [-tx, 0, -tz, -ty],
        [-ty, -tz, 0, -tx], [-tz, -ty, -tx, 0]], dtype=torch.float64)
    import math
    tn = math.sqrt(tx*tx+ty*ty+tz*tz)
    qu = math.cos(tn/2)*torch.eye(4, dtype=torch.float64) + (math.sin(tn/2)/tn)*tm
    ref = qu @ qu
    got = out[:256].reshape(16, 16)[:4, :4].cpu()
    print('mega-block max_err (4x4 q_update@q_update):', (got - ref).abs().max().item())
    print('scalar extract [0,1]:', out[256].item(), 'ref:', ref[0, 1].item())


if __name__ == '__main__':
    main()
