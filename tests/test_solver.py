import numpy as np
from epoxy_wave_dg import Config,DGElastic

def test_wave_speeds():
    s=DGElastic(Config(nx=3,ny=2,device='cpu'))
    assert np.isclose(s.cp,3763.6,rtol=.002)
    assert np.isclose(s.cs,2011.7,rtol=.002)

def test_characteristic_speeds():
    s=DGElastic(Config(nx=3,ny=2,device='cpu'))
    for A in (s.Ax,s.Ay):
        eig=np.sort(np.linalg.eigvals(A.cpu().numpy()).real)
        assert np.allclose(eig,[-s.cp,-s.cs,0,s.cs,s.cp],rtol=1e-7,atol=1e-5)

def test_zero_state_stays_zero():
    s=DGElastic(Config(nx=3,ny=2,source_velocity_scale=0, duration_us=.5,device='cpu'))
    r=s.run()
    assert np.max(np.abs(r['records']['receiver']))==0

def test_forced_run_finite():
    s=DGElastic(Config(nx=4,ny=2,duration_us=1,device='cpu'))
    r=s.run()
    assert np.isfinite(r['snapshots']).all()


def test_internal_source_and_surface_receiver():
    cfg=Config(nx=4,ny=2,source_y=.01,receiver_y=.02,duration_us=1,device='cpu',progress_every=0)
    model=DGElastic(cfg)
    src=model.probe_idx['source']; rec=model.probe_idx['receiver']
    assert abs(float(model.Y[src])-.01)<1e-12
    assert abs(float(model.Y[rec])-.02)<1e-12
    assert float(model.source_shape[src])>0.9
