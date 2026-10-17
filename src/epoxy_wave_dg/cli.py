import argparse, json
from pathlib import Path
import numpy as np
import yaml
from .solver import Config, DGElastic

def main():
    p=argparse.ArgumentParser(description='GPU 2D P2 DG elastic waves')
    p.add_argument('--config',default='configs/demo.yaml')
    p.add_argument('--out',default='results/demo')
    p.add_argument('--video',action='store_true',help='Save MP4 (also enabled by default if ffmpeg is installed)')
    p.add_argument('--no-video',action='store_true',help='Disable MP4 generation')
    args=p.parse_args()
    cfg=Config(**yaml.safe_load(Path(args.config).read_text()))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    print('[DG] initializing solver...',flush=True)
    solver=DGElastic(cfg); result=solver.run()
    print('[DG] saving numerical results...',flush=True)
    np.savez_compressed(out/'simulation.npz',
        **{k:v for k,v in result.items() if k not in ('records','velocity_records')},
        **{'displacement_'+k:v for k,v in result['records'].items()},
        **{'velocity_'+k:v for k,v in result['velocity_records'].items()})
    (out/'metadata.json').write_text(json.dumps({**vars(cfg),'cp':result['cp'],'cs':result['cs'],
        'dt':result['dt'],'steps':result['steps'],'device':result['device']},indent=2))
    print('[DG] rendering plots...',flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times=result['times']*1e6
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(10,6),sharex=True,gridspec_kw={'height_ratios':[1,1]})
    t=result['times']; target=cfg.source_velocity_scale*np.exp(-t/(cfg.decay_us*1e-6))*np.sin(2*np.pi*cfg.frequency*t)
    ax.plot(times,target*1e3,'k--',lw=1.5,label='PD driving waveform (scaled velocity reference)')
    ax.plot(times,result['velocity_records']['source']*1e3,label='internal source particle velocity',alpha=.85)
    ax.set(ylabel='Source velocity (mm/s)',title='Internal 100 kHz PD-like forcing and surface AE response')
    ax.legend(fontsize=8);ax.grid(alpha=.25)
    ax2.plot(times,result['velocity_records']['receiver']*1e3,color='tab:green',label='surface receiver (0.75 m)')
    ax2.axvline((cfg.receiver_x-cfg.source_x)/result['cp']*1e6,color='gray',ls=':',label='bulk P-wave travel time (reference)')
    ax2.set(xlabel='Time (µs)',ylabel='Receiver velocity (mm/s)')
    ax2.legend(fontsize=8);ax2.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'waveforms.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3.5))
    for name,y in result['records'].items():ax.plot(times,y*1e9,label=name)
    ax.set(xlabel='Time (µs)',ylabel='Normal displacement (nm)',title='Integrated normal displacement (diagnostic)')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'displacements.png',dpi=180);plt.close(fig)
    snap=result['snapshots']; vlim=max(float(np.percentile(snap,99.7)),1e-20)
    def markers(ax):
        ax.plot(cfg.source_x,cfg.source_y,'wo',ms=7,mec='black',clip_on=False)
        ax.plot(cfg.receiver_x,cfg.receiver_y,'w^',ms=8,mec='black',clip_on=False)
        ax.annotate('PD source', (cfg.source_x,cfg.source_y),xytext=(0,-18),textcoords='offset points',ha='center',color='black',fontsize=8,
                    bbox=dict(fc='white',alpha=.85,ec='none',pad=1))
        ax.annotate('AE receiver', (cfg.receiver_x,cfg.receiver_y),xytext=(0,-18),textcoords='offset points',ha='center',color='black',fontsize=8,
                    bbox=dict(fc='white',alpha=.85,ec='none',pad=1))
    fig,ax=plt.subplots(figsize=(10,2.7))
    im=ax.imshow(snap[-1].T,origin='lower',extent=(0,cfg.length,0,cfg.width),aspect='auto',cmap='turbo',vmin=0,vmax=vlim)
    markers(ax);fig.colorbar(im,ax=ax,label='|u| (m)');ax.set(xlabel='Along winding (m)',ylabel='Thickness (m)')
    fig.tight_layout();fig.savefig(out/'field.png',dpi=180);plt.close(fig)
    if not args.no_video:
        from matplotlib.animation import FuncAnimation, FFMpegWriter
        import shutil
        if shutil.which('ffmpeg'):
            print(f'[DG] encoding MP4 ({len(snap)} frames)...',flush=True)
            fig,ax=plt.subplots(figsize=(10,2.7))
            im=ax.imshow(snap[0].T,origin='lower',extent=(0,cfg.length,0,cfg.width),aspect='auto',cmap='turbo',vmin=0,vmax=vlim)
            markers(ax)
            ax.set(xlabel='Along winding (m)',ylabel='Thickness (m)')
            fig.colorbar(im,ax=ax,label='|u| (m)')
            title=ax.set_title('')
            def update(i):
                im.set_data(snap[i].T)
                title.set_text(f'Elastic wave | t={result["snapshot_times"][i]*1e6:.1f} µs')
                return im,title
            ani=FuncAnimation(fig,update,frames=len(snap),interval=40,blit=False)
            ani.save(out/'propagation.mp4',writer=FFMpegWriter(fps=20))
            plt.close(fig)
        else:
            print('MP4 skipped: ffmpeg is not installed. Install ffmpeg and rerun.')
    print(f'Done: {out}; device={result["device"]}; steps={result["steps"]}; dt={result["dt"]:.3g} s')

if __name__=='__main__':main()
