"""
EKF 精度隔离实验 — 用以论证 §3.2 "分层混合精度必要性"

复用 v6 mega-kernel 主体, 额外加入 4 个独立 constexpr 精度门, 每个 combo
只单独降其中一个组件的精度, 其余保持 fp32/ieee, 跑全轨迹测 RMSE 增量:

  门 1  DT_IO    : 传感器 (陀螺/加计/里程) load 精度
  门 2  IP       : 8 个 tl.dot 的 matmul input_precision (ieee / tf32)
  门 3  DT       : 内部算术 + 输出精度 (始终 fp32)
  门 4  DT_STATE : 递推状态 (pos/vel/q) 跨步存储精度 — 反映"被反复传播"量

combo:
  baseline    全 fp32/ieee (参考)
  io-fp16     仅传感器 I/O 降 fp16        → 预期不敏感 (消费一次的噪声输入)
  io-bf16     仅传感器 I/O 降 bf16        → 预期 Z 敏感 (bf16 尾数太粗)
  dot-tf32    仅 matmul 用 TF32           → 预期不敏感 (≤16×16 微矩阵)
  state-fp16  仅递推状态降 fp16           → 预期高度敏感 (累积 round-off)
"""
import torch, triton, triton.language as tl, numpy as np, os, time, json
from triton.language.extra.cuda import libdevice


# =====================================================================
# KERNEL: v4 body preserved EXACTLY; 2 changes: (1) load dtype (2) IP
# =====================================================================
@triton.jit
def ekf_kernel_v6(
    gyro_ptr, accel_ptr, odom1_ptr, odom2_ptr,
    qinit_ptr, qdiag_ptr, pos_out_ptr, vel_out_ptr,
    N, dt, g, R_odo, R_vcon, delta_thresh, R_big,
    DT_IO: tl.constexpr,    # sensor load dtype (fp16 / bf16 / fp32)
    DT: tl.constexpr,       # compute/output dtype (always tl.float32)
    IP: tl.constexpr,       # tl.dot input_precision (ieee / tf32)
    DT_STATE: tl.constexpr, # recurrent state (pos/vel/q) storage dtype between steps
):
    i = tl.arange(0, 16); j = tl.arange(0, 16)
    r = i[:, None]; c = j[None, :]
    eye = (r == c).to(DT)

    q0v = tl.load(qinit_ptr + i, mask=i < 4, other=0.0).to(DT)
    qdiag = tl.load(qdiag_ptr + i, mask=i < 15, other=0.0).to(DT)
    # --- GATE 4: recurrent state (pos/vel/q) held in DT_STATE between steps ---
    pos_s = tl.zeros((16,), dtype=DT_STATE); vel_s = tl.zeros((16,), dtype=DT_STATE)
    q_s = q0v.to(DT_STATE)
    P = eye * 0.1 * ((r < 15) & (c < 15)).to(DT)
    Qmat = tl.where(r == c, qdiag[:, None], tl.zeros((16, 16), dtype=DT))

    for k in range(1, N):
        # --- GATE 1: sensor load as DT_IO, then cast to DT ---
        wx = tl.load(gyro_ptr + (k-1)*3 + 0).to(DT_IO).to(DT)
        wy = tl.load(gyro_ptr + (k-1)*3 + 1).to(DT_IO).to(DT)
        wz = tl.load(gyro_ptr + (k-1)*3 + 2).to(DT_IO).to(DT)
        fx = tl.load(accel_ptr + (k-1)*3 + 0).to(DT_IO).to(DT)
        fy = tl.load(accel_ptr + (k-1)*3 + 1).to(DT_IO).to(DT)
        fz = tl.load(accel_ptr + (k-1)*3 + 2).to(DT_IO).to(DT)
        o1 = tl.load(odom1_ptr + k).to(DT_IO).to(DT)
        o2 = tl.load(odom2_ptr + k).to(DT_IO).to(DT)

        # upcast recurrent state to compute dtype for this step
        pos = pos_s.to(DT); vel = vel_s.to(DT); q = q_s.to(DT)
        pos_prev = pos; vel_prev = vel

        tx = wx*dt; ty = wy*dt; tz = wz*dt
        theta_norm = libdevice.sqrt(tx*tx+ty*ty+tz*tz)
        small = theta_norm > 1e-10
        denom = tl.where(small, theta_norm, 1.0)
        coef = tl.where(small, libdevice.sin(theta_norm/2)/denom, 0.5)
        cosv = libdevice.cos(theta_norm/2)

        tm = tl.zeros((16,16), dtype=DT)
        tm += tl.where((r==0)&(c==1), -tx, 0.0); tm += tl.where((r==0)&(c==2), -ty, 0.0); tm += tl.where((r==0)&(c==3), -tz, 0.0)
        tm += tl.where((r==1)&(c==0), -tx, 0.0); tm += tl.where((r==1)&(c==2), -tz, 0.0); tm += tl.where((r==1)&(c==3), -ty, 0.0)
        tm += tl.where((r==2)&(c==0), -ty, 0.0); tm += tl.where((r==2)&(c==1), -tz, 0.0); tm += tl.where((r==2)&(c==3), -tx, 0.0)
        tm += tl.where((r==3)&(c==0), -tz, 0.0); tm += tl.where((r==3)&(c==1), -ty, 0.0); tm += tl.where((r==3)&(c==2), -tx, 0.0)
        q4mask = ((r<4)&(c<4)).to(DT)
        q_update = (cosv*eye+coef*tm)*q4mask
        q = tl.sum(q_update*q[None,:], axis=1)
        qn = libdevice.sqrt(tl.sum(q*q)); q = q/qn

        qa=tl.sum(tl.where(i==0,q,0.0)); qb=tl.sum(tl.where(i==1,q,0.0))
        qc=tl.sum(tl.where(i==2,q,0.0)); qd=tl.sum(tl.where(i==3,q,0.0))

        C00=qa*qa+qb*qb-qc*qc-qd*qd; C01=2*(qb*qc+qa*qd); C02=2*(qb*qd-qa*qc)
        C10=2*(qb*qc-qa*qd); C11=qa*qa-qb*qb+qc*qc-qd*qd; C12=2*(qc*qd+qa*qb)
        C20=2*(qb*qd+qa*qc); C21=2*(qc*qd-qa*qb); C22=qa*qa-qb*qb-qc*qc+qd*qd

        fbx,fby,fbz=fx,fy,fz-g
        fn0=C00*fbx+C01*fby+C02*fbz; fn1=C10*fbx+C11*fby+C12*fbz; fn2=C20*fbx+C21*fby+C22*fbz
        velinc=tl.zeros((16,),dtype=DT); velinc+=tl.where(i==0,fn0*dt,0.0); velinc+=tl.where(i==1,fn1*dt,0.0); velinc+=tl.where(i==2,fn2*dt,0.0)
        vel=vel_prev+velinc; pos=pos_prev+vel_prev*dt

        rfn0=C00*fx+C01*fy+C02*fz; rfn1=C10*fx+C11*fy+C12*fz; rfn2=C20*fx+C21*fy+C22*fz
        dpx=tl.sum(tl.where(i==0,pos-pos_prev,0.0)); dpy=tl.sum(tl.where(i==1,pos-pos_prev,0.0)); dpz=tl.sum(tl.where(i==2,pos-pos_prev,0.0))
        delta_S=libdevice.sqrt(dpx*dpx+dpy*dpy+dpz*dpz); delta_D=(o1+o2)/2*dt
        normal=libdevice.abs(delta_D-delta_S)<delta_thresh; r_odo_eff=tl.where(normal,R_odo,R_big)
        vy=tl.sum(tl.where(i==1,vel,0.0)); vz=tl.sum(tl.where(i==2,vel,0.0))
        psi=libdevice.atan2(C01,C00); spsi=libdevice.sin(psi); cpsi=libdevice.cos(psi)

        zvec=tl.zeros((16,),dtype=DT); zvec+=tl.where(i==0,delta_S-delta_D,0.0); zvec+=tl.where(i==1,vy,0.0); zvec+=tl.where(i==2,vz,0.0)

        H=tl.zeros((16,16),dtype=DT)
        H+=tl.where((r==0)&(c==3),C00,0.0); H+=tl.where((r==0)&(c==4),C01,0.0); H+=tl.where((r==0)&(c==5),C02,0.0)
        H+=tl.where((r==1)&(c==0),-spsi,0.0); H+=tl.where((r==1)&(c==1),cpsi,0.0); H+=tl.where((r==2)&(c==2),1.0,0.0)

        F=eye*((r<15)&(c<15)).to(DT)
        F+=tl.where((r<3)&(c==r+3),dt,0.0); F+=tl.where((r==3)&(c==7),rfn2*dt,0.0); F+=tl.where((r==3)&(c==8),-rfn1*dt,0.0)
        F+=tl.where((r==4)&(c==6),-rfn2*dt,0.0); F+=tl.where((r==4)&(c==8),rfn0*dt,0.0); F+=tl.where((r==5)&(c==6),rfn1*dt,0.0)
        F+=tl.where((r==5)&(c==7),-rfn0*dt,0.0)
        F+=tl.where((r==3)&(c==12),-C00*dt,0.0); F+=tl.where((r==3)&(c==13),-C01*dt,0.0); F+=tl.where((r==3)&(c==14),-C02*dt,0.0)
        F+=tl.where((r==4)&(c==12),-C10*dt,0.0); F+=tl.where((r==4)&(c==13),-C11*dt,0.0); F+=tl.where((r==4)&(c==14),-C12*dt,0.0)
        F+=tl.where((r==5)&(c==12),-C20*dt,0.0); F+=tl.where((r==5)&(c==13),-C21*dt,0.0); F+=tl.where((r==5)&(c==14),-C22*dt,0.0)
        F+=tl.where((r==c)&(r>=6)&(r<9),-1.0,0.0)
        F+=tl.where((r==6)&(c==7),wz*dt,0.0); F+=tl.where((r==6)&(c==8),-wy*dt,0.0)
        F+=tl.where((r==7)&(c==6),-wz*dt,0.0); F+=tl.where((r==7)&(c==8),wx*dt,0.0)
        F+=tl.where((r==8)&(c==6),wy*dt,0.0); F+=tl.where((r==8)&(c==7),-wx*dt,0.0)
        F+=tl.where((r>=6)&(r<9)&(c==r+3),-dt,0.0)

        # --- GATE 2: all 8 tl.dot use IP ---
        FP=tl.dot(F,P,input_precision=IP); P_pred=tl.dot(FP,tl.trans(F),input_precision=IP)+Qmat
        HP=tl.dot(H,P_pred,input_precision=IP); S=tl.dot(HP,tl.trans(H),input_precision=IP)
        S+=tl.where((r==0)&(c==0),r_odo_eff,0.0); S+=tl.where((r==1)&(c==1),R_vcon,0.0); S+=tl.where((r==2)&(c==2),R_vcon,0.0)

        s00=tl.sum(tl.where((r==0)&(c==0),S,0.0)); s01=tl.sum(tl.where((r==0)&(c==1),S,0.0)); s02=tl.sum(tl.where((r==0)&(c==2),S,0.0))
        s10=tl.sum(tl.where((r==1)&(c==0),S,0.0)); s11=tl.sum(tl.where((r==1)&(c==1),S,0.0)); s12=tl.sum(tl.where((r==1)&(c==2),S,0.0))
        s20=tl.sum(tl.where((r==2)&(c==0),S,0.0)); s21=tl.sum(tl.where((r==2)&(c==1),S,0.0)); s22=tl.sum(tl.where((r==2)&(c==2),S,0.0))
        A=s11*s22-s12*s21; B=-(s10*s22-s12*s20); Cc=s10*s21-s11*s20
        D=-(s01*s22-s02*s21); E=s00*s22-s02*s20; Ff=-(s00*s21-s01*s20)
        G=s01*s12-s02*s11; Hh=-(s00*s12-s02*s10); Ii=s00*s11-s01*s10
        det=s00*A+s01*B+s02*Cc; idet=1.0/det
        Si=tl.zeros((16,16),dtype=DT)
        Si+=tl.where((r==0)&(c==0),A*idet,0.0); Si+=tl.where((r==0)&(c==1),D*idet,0.0); Si+=tl.where((r==0)&(c==2),G*idet,0.0)
        Si+=tl.where((r==1)&(c==0),B*idet,0.0); Si+=tl.where((r==1)&(c==1),E*idet,0.0); Si+=tl.where((r==1)&(c==2),Hh*idet,0.0)
        Si+=tl.where((r==2)&(c==0),Cc*idet,0.0); Si+=tl.where((r==2)&(c==1),Ff*idet,0.0); Si+=tl.where((r==2)&(c==2),Ii*idet,0.0)
        PHt=tl.dot(P_pred,tl.trans(H),input_precision=IP); K=tl.dot(PHt,Si,input_precision=IP)
        x=tl.sum(K*zvec[None,:],axis=1); KH=tl.dot(K,H,input_precision=IP); P=tl.dot(eye-KH,P_pred,input_precision=IP)

        x0=tl.sum(tl.where(i==0,x,0.0)); x1=tl.sum(tl.where(i==1,x,0.0)); x2=tl.sum(tl.where(i==2,x,0.0))
        x3=tl.sum(tl.where(i==3,x,0.0)); x4=tl.sum(tl.where(i==4,x,0.0)); x5=tl.sum(tl.where(i==5,x,0.0))
        p6=tl.sum(tl.where(i==6,x,0.0)); p7=tl.sum(tl.where(i==7,x,0.0)); p8=tl.sum(tl.where(i==8,x,0.0))
        poscorr=tl.zeros((16,),dtype=DT); poscorr+=tl.where(i==0,x0,0.0); poscorr+=tl.where(i==1,x1,0.0); poscorr+=tl.where(i==2,x2,0.0)
        velcorr=tl.zeros((16,),dtype=DT); velcorr+=tl.where(i==0,x3,0.0); velcorr+=tl.where(i==1,x4,0.0); velcorr+=tl.where(i==2,x5,0.0)
        pos=pos-poscorr; vel=vel-velcorr

        d0,d1,d2,d3=1.0,0.5*p6,0.5*p7,0.5*p8; dn=libdevice.sqrt(d0*d0+d1*d1+d2*d2+d3*d3); d0/=dn; d1/=dn; d2/=dn; d3/=dn
        nqa=qa*d0-qb*d1-qc*d2-qd*d3; nqb=qa*d1+qb*d0+qc*d3-qd*d2; nqc=qa*d2-qb*d3+qc*d0+qd*d1; nqd=qa*d3+qb*d2-qc*d1+qd*d0
        nn=libdevice.sqrt(nqa*nqa+nqb*nqb+nqc*nqc+nqd*nqd); nqa/=nn; nqb/=nn; nqc/=nn; nqd/=nn
        q=tl.zeros((16,),dtype=DT); q+=tl.where(i==0,nqa,0.0); q+=tl.where(i==1,nqb,0.0); q+=tl.where(i==2,nqc,0.0); q+=tl.where(i==3,nqd,0.0)

        # --- GATE 3: output stores (always fp32 for accurate RMSE) ---
        tl.store(pos_out_ptr+k*3+0,tl.sum(tl.where(i==0,pos,0.0)))
        tl.store(pos_out_ptr+k*3+1,tl.sum(tl.where(i==1,pos,0.0)))
        tl.store(pos_out_ptr+k*3+2,tl.sum(tl.where(i==2,pos,0.0)))
        tl.store(vel_out_ptr+k*3+0,tl.sum(tl.where(i==0,vel,0.0)))
        tl.store(vel_out_ptr+k*3+1,tl.sum(tl.where(i==1,vel,0.0)))
        tl.store(vel_out_ptr+k*3+2,tl.sum(tl.where(i==2,vel,0.0)))

        # --- round recurrent state back down to DT_STATE for the next step ---
        pos_s = pos.to(DT_STATE); vel_s = vel.to(DT_STATE); q_s = q.to(DT_STATE)


# =====================================================================
# 精度隔离实验: 每行只单独降一个组件的精度, 其余保持 fp32/ieee.
#   DT_IO    : 传感器 load 精度
#   IP       : 8 个 tl.dot 的 matmul input_precision
#   DT_STATE : 递推状态 (pos/vel/q) 跨步存储精度
# tuple = (DT_IO, IP, DT_STATE, torch_io, state_kind, desc)
# =====================================================================
COMBO = {
    'baseline':   (tl.float32,   'ieee', tl.float32, torch.float32,   tl.float32, '基线 (全部 fp32 / ieee)'),
    'io-fp16':    (tl.float16,   'ieee', tl.float32, torch.float16,   tl.float32, '仅传感器 I/O 降 fp16'),
    'io-bf16':    (tl.bfloat16,  'ieee', tl.float32, torch.bfloat16,  tl.float32, '仅传感器 I/O 降 bf16'),
    'dot-tf32':   (tl.float32,   'tf32', tl.float32, torch.float32,   tl.float32, '仅 matmul dot 用 TF32'),
    'state-fp16': (tl.float32,   'ieee', tl.float16, torch.float32,   tl.float16, '仅递推状态 pos/vel/q 降 fp16'),
}


def run_one(csv_path='data/trajectory/PipeRobot_Trajectory.csv', n_steps=None, combo='baseline', verbose=True):
    DT_IO, IP, DT_STATE, torch_io, tl_state, desc = COMBO[combo]
    DT = tl.float32  # compute always fp32
    dev = torch.device('cuda')

    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    t = torch.from_numpy(data[:,0]).to(dev)
    dt_val = float((t[1:]-t[:-1]).mean().item())
    n = len(t); n = min(n, n_steps) if n_steps else n

    gyro = torch.from_numpy(data[:,7:10]).to(dev, torch_io).contiguous()
    accel = torch.from_numpy(data[:,10:13]).to(dev, torch_io).contiguous()
    odom1 = torch.from_numpy(data[:,13]).to(dev, torch_io).contiguous()
    odom2 = torch.from_numpy(data[:,14]).to(dev, torch_io).contiguous()
    pos_true = torch.from_numpy(data[:,1:4]).to(dev)

    g=9.81; ax0=accel[:10,0].float().mean(); ay0=accel[:10,1].float().mean(); az0=accel[:10,2].float().mean()
    pitch0=torch.atan(ay0/torch.sqrt(ax0**2+az0**2)); roll0=torch.atan(-ax0/az0)
    yaw0=torch.tensor(0.0,dtype=torch.float32,device=dev)
    cy,sy=torch.cos(yaw0/2),torch.sin(yaw0/2); cp,sp=torch.cos(pitch0/2),torch.sin(pitch0/2); cr,sr=torch.cos(roll0/2),torch.sin(roll0/2)
    qinit=torch.stack([cy*cp*cr+sy*sp*sr,cy*cp*sr-sy*sp*cr,cy*sp*cr+sy*cp*sr,sy*cp*cr-cy*sp*sr]).to(torch.float32).contiguous()
    qdiag=torch.tensor([1e-6]*3+[1e-5]*3+[1e-4]*3+[1e-8]*3+[1e-7]*3,dtype=torch.float32,device=dev).contiguous()
    pos_out=torch.zeros(n,3,dtype=torch.float32,device=dev).contiguous()
    vel_out=torch.zeros(n,3,dtype=torch.float32,device=dev).contiguous()

    ekf_kernel_v6[(1,)](gyro,accel,odom1,odom2,qinit,qdiag,pos_out,vel_out,
                        min(n,64),dt_val,g,1e-4,1e-3,0.01,1e12,DT_IO,DT,IP,DT_STATE)
    torch.cuda.synchronize(); pos_out.zero_(); vel_out.zero_()

    t0=time.time()
    ekf_kernel_v6[(1,)](gyro,accel,odom1,odom2,qinit,qdiag,pos_out,vel_out,
                        n,dt_val,g,1e-4,1e-3,0.01,1e12,DT_IO,DT,IP,DT_STATE)
    torch.cuda.synchronize(); elapsed=time.time()-t0

    pos64=pos_out.double(); err=pos64-pos_true[:n]
    rx=(torch.sqrt((err[:,0]**2).mean())*1000).item()
    ry=(torch.sqrt((err[:,1]**2).mean())*1000).item()
    rz=(torch.sqrt((err[:,2]**2).mean())*1000).item()
    l=(pos_true[:n,0].max()-pos_true[:n,0].min()).item()
    xd=torch.sqrt(((pos64[-1]-pos64[0])**2).sum()).item()
    rho=(l-xd)/l if l!=0 else 0.0
    m={'combo':combo,'desc':desc,'n':n,'elapsed':elapsed,'tput':(n-1)/elapsed,
       'rmse_x':rx,'rmse_y':ry,'rmse_z':rz,'rho':rho*100}
    if verbose: print(f"[{combo}] t={elapsed:.3f}s ({m['tput']:.0f} st/s) X={rx:.3f} Y={ry:.3f} Z={rz:.3f} rho={m['rho']:.4f}%")
    return pos64,vel_out,pos_true[:n],t[:n],m


def sweep(n_steps=None):
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    print("="*112); print("EKF 精度隔离实验 — 每行只单独降一个组件精度 (其余 fp32/ieee)"); print("="*112)
    results=[]
    for name in COMBO:  # keep baseline first
        desc = COMBO[name][5]
        print(f"\n[{name}] {desc}")
        try:
            _,_,_,_,m=run_one(n_steps=n_steps,combo=name)
            results.append(m)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAILED: {e}"); results.append({'combo':name,'error':str(e)})

    base=next((r for r in results if r.get('combo')=='baseline'),None)
    bx,bz= (base['rmse_x'],base['rmse_z']) if base and 'rmse_x' in base else (float('nan'),float('nan'))
    for r in results:
        if 'error' not in r:
            r['dX']=r['rmse_x']-bx; r['dZ']=r['rmse_z']-bz
            r['rel_dX_pct']=(r['dX']/bx*100) if bx else float('nan')

    print("\n"+"="*128)
    print(f"{'Combo':<12} {'Desc':<34} {'RMSE_X(mm)':>11} {'dX(mm)':>9} {'rel%':>7} {'RMSE_Z(mm)':>11} {'dZ(mm)':>9} {'rho%':>8}")
    print("-"*128)
    for r in results:
        if 'error' in r: print(f"{r['combo']:<12} FAILED: {r['error'][:60]}"); continue
        print(f"{r['combo']:<12} {r['desc']:<34} {r['rmse_x']:>11.2f} {r.get('dX',0):>+9.2f} {r.get('rel_dX_pct',0):>+7.2f} {r['rmse_z']:>11.2f} {r.get('dZ',0):>+9.2f} {r['rho']:>8.4f}")

    print("\n结论: 不敏感组件 (可降精度以省带宽/寄存器) vs 敏感组件 (必须保 fp32) 的差异 → 论证分层混合精度必要性")
    os.makedirs('data/results',exist_ok=True)
    out='data/results/precision_isolation.json'
    with open(out,'w') as f: json.dump(results,f,indent=2,ensure_ascii=False)
    print(f"\nSaved → {out}")
    return results

if __name__=='__main__': sweep(n_steps=None)  # full trajectory (166667 steps)
