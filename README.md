# Epoxy Wave DG

**GPU-capable 2D elastic-wave simulation for partial-discharge (PD) acoustic propagation in an epoxy-equivalent winding.**

Epoxy Wave DG models the transient mechanical response of a thin epoxy strip to a localized internal, PD-like excitation. It uses a nodal discontinuous Galerkin spectral-element method (DGSEM), explicit time integration, and PyTorch for CPU or NVIDIA CUDA execution. The package produces wave-field images, source/receiver time histories, numerical output, and an optional MP4 animation.

## Model at a glance

| Item | Default setup |
| --- | --- |
| Domain | 1.0 m long × 0.020 m thick, 2D plane strain |
| Material | Homogeneous isotropic linear elastic epoxy-equivalent |
| Young's modulus | 20 GPa |
| Poisson's ratio | 0.30 |
| Density | 1900 kg/m³ |
| Bulk P-wave / S-wave speeds | Approximately 3764 / 2012 m/s |
| Excitation | Localized internal, exponentially decaying 100 kHz sinusoidal body acceleration |
| PD source | (x, y) = (0.25, 0.010) m (mid-thickness) |
| Surface AE receiver | (x, y) = (0.75, 0.020) m |
| Along-strip source–receiver separation | 0.50 m |
| Boundaries | Traction-free long sides and short ends by default; optional periodic short ends |
| Solver | Degree-2 nodal DGSEM on structured quadrilateral elements; explicit RK4 |
| Compute | PyTorch CUDA when available, otherwise CPU |

The source is a localized mechanical proxy for an internal discharge, not an electrical PD model. Its amplitude is not calibrated to apparent charge (pC). The receiver reports computed surface particle velocity and displacement; sensor electronics and transfer functions are not included.

## Repository layout

```text
epoxy-wave-dg/
├── .github/workflows/tests.yml    # Automated tests
├── configs/
│   ├── demo.yaml                  # Short installation smoke test
│   ├── preview_cpu.yaml           # Coarse 200 µs preview
│   └── production_template.yaml   # Finer 350 µs simulation
├── src/epoxy_wave_dg/
│   ├── __init__.py
│   ├── cli.py                     # Command-line interface and outputs
│   └── solver.py                  # DGSEM elastodynamics solver
├── tests/test_solver.py
├── .gitignore
├── pyproject.toml
└── README.md
```

## Installation

Requires **Python 3.10+**. Install from the repository root:

```bash
python -m pip install -e ".[test]"
```

PyTorch is a dependency. For NVIDIA GPU use, install a CUDA-enabled PyTorch build appropriate for your driver and operating system; see the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

Check CUDA availability:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

To produce MP4 files, install **FFmpeg** and ensure `ffmpeg` is on your `PATH`:

```powershell
# Windows (PowerShell)
winget install --id Gyan.FFmpeg -e
```

```bash
# Conda alternative
conda install -c conda-forge ffmpeg
```

Restart your terminal after installing FFmpeg, then verify with `ffmpeg -version`.

## Run a simulation

From the repository root:

```bash
# Quick functionality check
epoxy-wave-dg --config configs/demo.yaml --out results/demo

# Coarse preview, including a longer time window
epoxy-wave-dg --config configs/preview_cpu.yaml --out results/preview_cpu

# Finer model (GPU recommended)
epoxy-wave-dg --config configs/production_template.yaml --out results/production
```

MP4 generation is attempted by default when FFmpeg is installed. Use `--no-video` to skip animation generation. `--video` is also accepted.

### Output files

| File | Contents |
| --- | --- |
| `field.png` | Final displacement-magnitude field with source and receiver markers |
| `propagation.mp4` | Time-dependent displacement-magnitude contours (when FFmpeg is available) |
| `waveforms.png` | Internal excitation reference, source particle velocity, and surface receiver velocity |
| `displacements.png` | Integrated normal-displacement probe histories |
| `simulation.npz` | Numerical time histories and saved field snapshots |
| `metadata.json` | Configuration, material wave speeds, time step, steps, and device |

The excitation reference shown in `waveforms.png` is a scaled sinusoidal *forcing reference*. Because the source is internal body acceleration, it is not a prescribed particle velocity. The computed surface receiver waveform can contain guided modes, dispersion, and reflections.

## Configuration

Edit a YAML file under `configs/` to change the model. The principal settings are:

```yaml
length: 1.0
width: 0.02
nx: 400
ny: 12
E: 20000000000.0
nu: 0.3
rho: 1900.0
frequency: 100000.0
decay_us: 100.0
source_velocity_scale: 0.001
source_x: 0.25
source_y: 0.01
source_sigma_x: 0.005
source_sigma_y: 0.005
receiver_x: 0.75
receiver_y: 0.02
duration_us: 350.0
cfl: 0.12
snapshot_every: 100
record_every: 10
progress_every: 100
periodic_x: false
damping_per_s: 0.0
device: auto
```

- `nx`, `ny`: element counts along the strip and through its thickness.
- `source_sigma_x`, `source_sigma_y`: spatial extent of the localized internal excitation, in metres.
- `source_velocity_scale`: legacy parameter name; in the internal-source model it sets the **forcing scale**, with acceleration amplitude proportional to `2π × frequency × source_velocity_scale`. It is **not** a prescribed source velocity.
- `duration_us`: simulated time in microseconds.
- `snapshot_every`: time-step interval between stored wave-field frames; affects MP4 size and memory use.
- `record_every`: time-step interval between recorded probe samples.
- `progress_every`: print progress every N time steps; set to `0` to disable periodic progress output.
- `periodic_x`: periodic short-end boundaries if `true`; default is traction-free ends.
- `damping_per_s`: optional phenomenological velocity drag, in s⁻¹; default `0` means no material damping.
- `device`: `auto`, `cuda`, or `cpu` (subject to local PyTorch/CUDA availability).

The time step is selected by the solver using the configured CFL setting.

## Numerical formulation

The 2D plane-strain solver advances particle velocity and Cauchy stress:

\[
\rho\,\partial_t\mathbf v = \nabla\cdot\boldsymbol\sigma + \mathbf f,
\qquad
\partial_t\boldsymbol\sigma = \mathsf C : \operatorname{sym}(\nabla\mathbf v).
\]

The implementation uses tensor-product degree-2 Gauss–Lobatto–Legendre nodal elements, characteristic numerical fluxes, and classical explicit fourth-order Runge–Kutta time stepping. A localized Gaussian body-acceleration source drives the internal PD-like disturbance. The long faces represent traction-free epoxy–air surfaces.

The 1 m strip is an unwrapped winding section. Airborne transmission, winding curvature, duct-stick connections, and detailed composite construction are outside this simplified domain.

## Progress logging

During execution, the terminal prints the selected compute device, mesh and source/receiver locations, step count, percentage completed, simulated time, elapsed wall time, estimated time remaining, and post-processing stages.

For example:

```text
[DG] initializing solver...
[DG] device=cuda | ...
[DG] internal PD source=(0.250, 0.010) m
[DG] surface receiver=(0.750, 0.020) m
[DG] ... 50.0% | t=... us | elapsed=... | ETA=...
[DG] saving numerical results...
[DG] rendering plots...
[DG] encoding MP4 (... frames)...
```

If FFmpeg is not installed, numerical output and PNG figures are still generated, and the MP4 step is skipped.

## Testing

```bash
pytest -q
```

The repository includes automated solver checks and a GitHub Actions test workflow. For quantitative engineering use, perform mesh/time-step convergence checks and validate wave arrival times, boundary response, source scaling, and attenuation against suitable reference data. The default damping is zero, so reductions in measured probe amplitude should not be interpreted as calibrated epoxy material attenuation.

## License

No license file is included. Add a license appropriate for your intended distribution before publishing the repository for reuse.
