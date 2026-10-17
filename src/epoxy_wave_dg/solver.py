"""2-D tensor-product nodal DG spectral element method (DGSEM), P2 GLL.

State q=[vx,vy,sxx,syy,sxy]; q_t=Ax q_x+Ay q_y+source.
Linear isotropic plane-strain elasticity; upwind characteristic fluxes.
"""
from dataclasses import dataclass
import math
import time
import numpy as np
import torch

GLL_D = np.array([[-1.5, 2., -.5],[-.5, 0., .5],[.5,-2.,1.5]])
GLL_W = np.array([1./3, 4./3, 1./3])

@dataclass
class Config:
    length: float = 1.0
    width: float = 0.02
    nx: int = 100
    ny: int = 4
    E: float = 20e9
    nu: float = 0.3
    rho: float = 1900.
    frequency: float = 100000.
    decay_us: float = 100.
    source_velocity_scale: float = 0.001
    source_x: float = 0.25
    source_y: float = 0.01
    source_sigma_x: float = 0.008
    source_sigma_y: float = 0.008
    receiver_x: float = 0.75
    receiver_y: float = 0.02
    progress_every: int = 100
    duration_us: float = 60.
    cfl: float = 0.12
    snapshot_every: int = 20
    record_every: int = 5
    periodic_x: bool = False
    damping_per_s: float = 0.
    device: str = 'auto'

class DGElastic:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if cfg.device == 'auto' else cfg.device
        self.dtype = torch.float64
        lam = cfg.E*cfg.nu/((1+cfg.nu)*(1-2*cfg.nu))
        mu = cfg.E/(2*(1+cfg.nu))
        self.cp = math.sqrt((lam+2*mu)/cfg.rho)
        self.cs = math.sqrt(mu/cfg.rho)
        Ax = np.zeros((5,5)); Ay=np.zeros((5,5))
        Ax[0,2]=1/cfg.rho; Ax[1,4]=1/cfg.rho
        Ax[2,0]=lam+2*mu; Ax[3,0]=lam; Ax[4,1]=mu
        Ay[0,4]=1/cfg.rho; Ay[1,3]=1/cfg.rho
        Ay[2,1]=lam; Ay[3,1]=lam+2*mu; Ay[4,0]=mu
        self.Ax=torch.tensor(Ax,device=self.device,dtype=self.dtype)
        self.Ay=torch.tensor(Ay,device=self.device,dtype=self.dtype)
        def abs_matrix(A):
            eig,vec=np.linalg.eig(A)
            return np.real(vec@np.diag(np.abs(eig))@np.linalg.inv(vec))
        self.absAx=torch.tensor(abs_matrix(Ax),device=self.device,dtype=self.dtype)
        self.absAy=torch.tensor(abs_matrix(Ay),device=self.device,dtype=self.dtype)
        self.D=torch.tensor(GLL_D,device=self.device,dtype=self.dtype)
        self.dx=cfg.length/cfg.nx; self.dy=cfg.width/cfg.ny
        self.q=torch.zeros((cfg.nx,cfg.ny,3,3,5),device=self.device,dtype=self.dtype)
        self.u=torch.zeros_like(self.q[...,0:2])
        self.dt=cfg.cfl*min(self.dx,self.dy)/(5*self.cp)
        # P2 nodal coordinates, shape (nx,ny,3,3)
        xx=(torch.arange(cfg.nx,device=self.device,dtype=self.dtype)[:,None,None,None]
            +(torch.tensor([0.,.5,1.],device=self.device)[None,None,:,None]))*self.dx
        yy=(torch.arange(cfg.ny,device=self.device,dtype=self.dtype)[None,:,None,None]
            +(torch.tensor([0.,.5,1.],device=self.device)[None,None,None,:]))*self.dy
        self.X=xx+torch.zeros_like(yy); self.Y=yy+torch.zeros_like(xx)
        gaussian=torch.exp(-.5*((self.X-cfg.source_x)/cfg.source_sigma_x)**2
                             -.5*((self.Y-cfg.source_y)/cfg.source_sigma_y)**2)
        self.source_shape=gaussian/(gaussian.max()+1e-30)  # retained for diagnostics
        # Internal localized acceleration source (normal/y direction), not a boundary condition.
        # v_scale is a characteristic velocity scale; forcing is scaled by 2*pi*f.
        self.source_acceleration_shape=self.source_shape
        self.source_acceleration_scale=2*math.pi*cfg.frequency*cfg.source_velocity_scale

        # receiver and source nodal samples: choose nearest DOF
        self.probe_idx={}
        for label,x,y in [('source',cfg.source_x,cfg.source_y),('middle',cfg.length/2,cfg.width/2),('receiver',cfg.receiver_x,cfg.receiver_y)]:
            dist=(self.X-x)**2+(self.Y-y)**2
            flat=int(torch.argmin(dist).item())
            self.probe_idx[label]=np.unravel_index(flat,dist.shape)

    @staticmethod
    def matvec(A,q):
        return torch.einsum('ab,...b->...a',A,q)

    def rhs(self,q,t):
        gx=self.matvec(self.Ax,q); gy=self.matvec(self.Ay,q)
        rhs=(2/self.dx)*torch.einsum('ik,abkjc->abijc',self.D,gx)
        rhs+=(2/self.dy)*torch.einsum('jk,abikc->abijc',self.D,gy)
        # x internal faces: left element right face / right element left face
        l=q[:-1,:,-1,:,:]; r=q[1:,:,0,:,:]
        f=.5*(self.matvec(self.Ax,l)+self.matvec(self.Ax,r))+ .5*self.matvec(self.absAx,r-l)
        rhs[:-1,:,-1,:,:]+=(2/self.dx)*3*(f-gx[:-1,:,-1,:,:])
        rhs[1:,:,0,:,:]+=(2/self.dx)*3*(gx[1:,:,0,:,:]-f)
        if self.cfg.periodic_x:
            l=q[-1,:,-1,:,:]; r=q[0,:,0,:,:]
            f=.5*(self.matvec(self.Ax,l)+self.matvec(self.Ax,r))+.5*self.matvec(self.absAx,r-l)
            rhs[-1,:,-1,:,:]+=(2/self.dx)*3*(f-gx[-1,:,-1,:,:])
            rhs[0,:,0,:,:]+=(2/self.dx)*3*(gx[0,:,0,:,:]-f)
        else:
            # free ends: reflected normal/shear traction; velocities continuous
            for idx,face,sgn in [(0,0,-1),(-1,-1,1)]:
                qi=q[idx,:,face,:,:]
                ghost=qi.clone(); ghost[...,2]*=-1; ghost[...,4]*=-1
                if sgn==-1:
                    f=.5*(self.matvec(self.Ax,ghost)+self.matvec(self.Ax,qi))+.5*self.matvec(self.absAx,qi-ghost)
                    rhs[idx,:,face,:,:]+=(2/self.dx)*3*(gx[idx,:,face,:,:]-f)
                else:
                    f=.5*(self.matvec(self.Ax,qi)+self.matvec(self.Ax,ghost))+.5*self.matvec(self.absAx,ghost-qi)
                    rhs[idx,:,face,:,:]+=(2/self.dx)*3*(f-gx[idx,:,face,:,:])
        # y internal faces
        l=q[:,:-1,:,-1,:]; r=q[:,1:,:,0,:]
        f=.5*(self.matvec(self.Ay,l)+self.matvec(self.Ay,r))+.5*self.matvec(self.absAy,r-l)
        rhs[:,:-1,:,-1,:]+=(2/self.dy)*3*(f-gy[:,:-1,:,-1,:])
        rhs[:,1:,:,0,:]+=(2/self.dy)*3*(gy[:,1:,:,0,:]-f)
        # traction-free long sides: normal stress syy and shear sxy vanish
        for idx,face,sgn in [(0,0,-1),(-1,-1,1)]:
            qi=q[:,idx,:,face,:]
            ghost=qi.clone(); ghost[...,3]*=-1; ghost[...,4]*=-1
            if sgn==-1:
                f=.5*(self.matvec(self.Ay,ghost)+self.matvec(self.Ay,qi))+.5*self.matvec(self.absAy,qi-ghost)
                rhs[:,idx,:,face,:]+=(2/self.dy)*3*(gy[:,idx,:,face,:]-f)
            else:
                f=.5*(self.matvec(self.Ay,qi)+self.matvec(self.Ay,ghost))+.5*self.matvec(self.absAy,ghost-qi)
                rhs[:,idx,:,face,:]+=(2/self.dy)*3*(f-gy[:,idx,:,face,:])
        source_wave=(math.exp(-t/(self.cfg.decay_us*1e-6))
                     *math.sin(2*math.pi*self.cfg.frequency*t))
        rhs[...,1]+=self.source_acceleration_scale*self.source_acceleration_shape*source_wave
        if self.cfg.damping_per_s:
            rhs[...,0:2]-=self.cfg.damping_per_s*q[...,0:2]
        return rhs

    def run(self):
        cfg=self.cfg
        nsteps=math.ceil(cfg.duration_us*1e-6/self.dt)
        dt=cfg.duration_us*1e-6/nsteps
        q=self.q; u=self.u
        wall_start=time.perf_counter()
        print(f'[DG] device={self.device} | mesh={cfg.nx}x{cfg.ny} P2 | '
              f'dt={dt*1e9:.2f} ns | steps={nsteps} | duration={cfg.duration_us:g} us',flush=True)
        print(f'[DG] internal PD source=({cfg.source_x:.3f}, {cfg.source_y:.3f}) m | '
              f'surface receiver=({cfg.receiver_x:.3f}, {cfg.receiver_y:.3f}) m',flush=True)
        records={name:[] for name in self.probe_idx}
        velocity_records={name:[] for name in self.probe_idx}
        times=[]; snapshots=[]; snapshot_times=[]
        for n in range(nsteps+1):
            t=n*dt
            if cfg.progress_every > 0 and (n % cfg.progress_every == 0 or n == nsteps):
                elapsed=time.perf_counter()-wall_start
                eta=elapsed*(nsteps-n)/n if n else float('nan')
                print(f'[DG] {n:>7}/{nsteps} ({100*n/nsteps:5.1f}%) | '
                      f't={t*1e6:7.2f} us | elapsed={elapsed:6.1f}s | '
                      f'ETA={eta:6.1f}s' if n else
                      f'[DG] {n:>7}/{nsteps} (  0.0%) | starting',flush=True)
            if n%cfg.record_every==0 or n==nsteps:
                times.append(t)
                for name,idx in self.probe_idx.items():
                    records[name].append(float(u[idx+(1,)].item()))
                    velocity_records[name].append(float(q[idx+(1,)].item()))
            if n%cfg.snapshot_every==0 or n==nsteps:
                # average displacement magnitude over local polynomial nodes
                snapshots.append(torch.linalg.vector_norm(u,dim=-1).mean(dim=(2,3)).detach().cpu().numpy())
                snapshot_times.append(t)
            if n==nsteps: break
            # Classical RK4 for q and u_t=v
            k1=self.rhs(q,t); v1=q[...,0:2]
            q2=q+.5*dt*k1
            k2=self.rhs(q2,t+.5*dt); v2=q2[...,0:2]
            q3=q+.5*dt*k2
            k3=self.rhs(q3,t+.5*dt); v3=q3[...,0:2]
            q4=q+dt*k3
            k4=self.rhs(q4,t+dt); v4=q4[...,0:2]
            q=q+(dt/6)*(k1+2*k2+2*k3+k4)
            u=u+(dt/6)*(v1+2*v2+2*v3+v4)
            if not torch.isfinite(q).all(): raise FloatingPointError(f'Nonfinite DG state at step {n}')
        return {'times':np.asarray(times),'records':{k:np.asarray(v) for k,v in records.items()},
                'velocity_records':{k:np.asarray(v) for k,v in velocity_records.items()},
                'snapshots':np.asarray(snapshots),'snapshot_times':np.asarray(snapshot_times),
                'dt':dt,'steps':nsteps,'device':self.device,'cp':self.cp,'cs':self.cs}
