# WaveformInversionUST

[English](README.md) | [简体中文](README.zh-CN.md)

**A MATLAB runtime for two-dimensional sound-speed full-waveform inversion in ultrasound computed tomography (USCT).**

WUST reconstructs a sound-speed map from complex pressure measurements, acquisition geometry and an initial model. It combines frequency continuation, a nine-point Helmholtz solver and nonlinear conjugate-gradient updates. CPU and GPU backends share the same physical model and discretization coefficients.

The repository serves two integration paths: MATLAB functions for numerical development, and a Python JSON/HDF5 interface for scripts, benchmark systems and agents. [usct-benchlab](https://github.com/lucian-dw/usct-benchlab) connects to this runtime through its `fwi_wust` algorithm; WUST also runs as a standalone project.

**Start here:** [Repository map](#repository-map) · [Run a reconstruction](#run-a-reconstruction) · [Python and MATLAB interfaces](#python-and-matlab-interfaces) · [Data and results](#data-and-results) · [Development](#development)

## Numerical model and capabilities

The model variable is slowness $s=1/c$, where $c$ is sound speed in m/s. Reconstruction uses a declared update region and sound-speed bounds, with attenuation fixed at zero.

At each frequency and transmitter, $p(s)$ is the predicted pressure and $d$ is the observed pressure on the selected receivers. WUST fits a complex source coefficient and minimizes the resulting data misfit:

$$
a^\star=\frac{p^H d}{p^H p},
\qquad
\Phi_f(s)=\frac12\sum_{\mathrm{TX}}\|a^\star p(s)-d\|_2^2.
$$

| Component | Implementation |
| --- | --- |
| Wave propagation | Optimized nine-point Helmholtz discretization with a perfectly matched layer (PML) |
| CPU solver | Double-precision sparse LU, reused for forward, adjoint and curvature solves at the current model and frequency |
| GPU solver | Complex-single CUDA Block LU with batched transmitters |
| Derivatives | Discrete slowness gradient and directional curvature |
| Optimization | Frequency continuation, ROI gradient filtering, preconditioned nonlinear CG and projected slowness updates |
| Wavenumber | `continuum`, or `kwave-ldr9` with square cells and explicit simulator time step and reference speed |
| Data preparation | Frequency-pressure ingestion, RF resampling and DTFT, and k-Wave simulation |

The objective, stencil derivative and update formulas are documented in [Numerical contracts](docs/CONTRACTS.md).

## Repository map

| Location | Responsibility | Start with |
| --- | --- | --- |
| `Runtime/python/` | Request models, JSON/HDF5 transport and MATLAB process execution | [wust_runtime.py](Runtime/python/wust_runtime.py), [contracts.py](Runtime/python/contracts.py), [wust_io.py](Runtime/python/wust_io.py) |
| `Runtime/matlab/` | Observation preparation, FWI iteration and artifact I/O | [wust_run.m](Runtime/matlab/wust_run.m), [wust_prepare_observations.m](Runtime/matlab/wust_prepare_observations.m), [wust_reconstruct.m](Runtime/matlab/wust_reconstruct.m) |
| `Runtime/solver/` | Helmholtz assembly, CPU/GPU solves and CUDA MEX | [HelmholtzSolver.m](Runtime/solver/HelmholtzSolver.m), [ninePointCoefficients.m](Runtime/solver/ninePointCoefficients.m) |
| `Runtime/schemas/` | JSON schemas for requests and array artifacts | [wust.request.json](Runtime/schemas/wust.request.json) |
| `Runtime/native/` | Native k-Wave simulation launcher and source batching | [Deployment guide](Runtime/README.md) |
| `Runtime/third_party/` | Bundled k-Wave CUDA source and its license | [k-Wave CUDA](Runtime/third_party/kspaceFirstOrder-CUDA/) |
| `Runtime/tests/` | Python protocol tests and MATLAB numerical checks | [Validation guide](docs/validation.md) |
| `examples/` | Simulation, RF preparation and reconstruction workflow | [simulate_and_reconstruct.m](examples/simulate_and_reconstruct.m) |
| `docs/` | Numerical definitions, protocol and development references | [Runtime API](docs/RUNTIME_API.md), [中文流程说明](docs/PIPELINE_ZH.md) |

For a numerical reading path, start with `wust_reconstruct.m`, follow its loss and gradient call into `wust_oracle.m`, then read `HelmholtzSolver.m` for the discrete wave equation. For an integration reading path, start with `wust_runtime.py` and the request fields in `contracts.py`.

## Run a reconstruction

### 1. Set up the environment

Run the following commands from the repository root on Linux or macOS, using **Python 3.12+**:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy h5py
python Runtime/python/wust_runtime.py describe
```

`describe` returns the available operations, protocol schemas, backends and budget controls as JSON.

| Capability | Environment |
| --- | --- |
| CPU reconstruction | MATLAB |
| GPU reconstruction | MATLAB Parallel Computing Toolbox, a supported CUDA compiler and built CUDA MEX |
| FIR RF decimation | MATLAB Signal Processing Toolbox |
| Acoustic simulation | k-Wave MATLAB toolbox |
| Native batched simulation | Linux with the CUDA/HDF5 build described in [Runtime/README.md](Runtime/README.md) |

For GPU reconstruction, build the MEX functions from MATLAB:

```matlab
addpath('Runtime/matlab');
wust_build_mex;
```

Select the reconstruction backend with `config.backend`. The request below uses `cpu`.

### 2. Supply the observations and initial model

The runtime exposes four operations:

| Operation | Input | Output |
| --- | --- | --- |
| `reconstruct` | `wust.frequency_input` or `wust.measurements`, plus `wust.initial_model` | `wust.reconstruction` |
| `ingest_frequency` | `wust.frequency_input` | Reusable `wust.measurements` |
| `prepare` | `wust.rf` and an RF preparation configuration | `wust.measurements` |
| `simulate` | `wust.simulation_input` and a simulation configuration | `wust.rf` |

A reconstruction takes two input artifacts: frequency-domain pressure observations and an initial sound-speed map with its update mask. Each artifact consists of a JSON manifest and an HDF5 array file. `reconstruct` prepares the supplied observations and executes the frequency plan in one MATLAB process.

Use `ingest_frequency` to save prepared frequency observations for reuse, `prepare` to convert RF measurements, and `simulate` to generate RF data. The [runtime API](docs/RUNTIME_API.md) defines the fields for all four operations.

### 3. Create and execute a request

Save this as `request.json`. Set the absolute paths and physical values for your acquisition. This example schedules two updates at 100 kHz and one at 120 kHz:

```json
{
  "schema": "wust.request",
  "schema_version": 2,
  "operation": "reconstruct",
  "input_manifest": "/absolute/input/frequency.json",
  "initial_manifest": "/absolute/input/initial.json",
  "output_manifest": "/absolute/output/reconstruction.json",
  "frequency_schedule_hz": [100000, 100000, 120000],
  "max_updates": 3,
  "save_history": false,
  "config": {
    "backend": "cpu",
    "bounds_mps": [1300, 1800],
    "stencil_bounds": [1400, 1700],
    "pml_m": 0.004,
    "pml_strength": 10,
    "max_update_mps": 12,
    "step_damping": 0.25,
    "source_batch_size": 2,
    "wavenumber": "continuum",
    "filter_cutoff": 0,
    "filter_order": 4
  }
}
```

`bounds_mps` sets the sound-speed range, and `stencil_bounds` sets the fixed speed interval used to construct the nine-point stencil. `pml_m` is the PML thickness in meters; `max_update_mps` sets the per-step sound-speed change in m/s. Schedule entries select frequencies present in the input observations.

Create the output directory and choose a new output manifest for each run:

```bash
mkdir -p /absolute/output
python Runtime/python/wust_runtime.py run /absolute/request.json \
  --matlab /path/to/matlab --timeout-s 300
```

The launcher returns one JSON result on stdout and writes diagnostic messages to stderr. `--timeout-s` sets the elapsed deadline for the invocation. The saved manifest links the completed HDF5 arrays and contains the configuration and per-update records.

## Python and MATLAB interfaces

### Python scripts and agents

Choose the CLI command above or the Python call below to execute a request. Set `PYTHONPATH=Runtime/python` when importing the runtime modules; `capabilities()` describes the interface and `run()` executes the request file:

```bash
PYTHONPATH=Runtime/python python - <<'PY'
from wust_runtime import capabilities, run

spec = capabilities()
print(spec["operations"])
result = run(
    "/absolute/request.json",
    matlab="/path/to/matlab",
    timeout_s=300,
)
print(result["metadata"]["completion"])
PY
```

For agent integration, the host provides paths, backend and elapsed budget; the request specifies the observations, initial model and numerical plan. [usct-benchlab's fwi_wust adapter](https://github.com/lucian-dw/usct-benchlab) supplies this protocol through its shared algorithm API.

Read a saved reconstruction with the Python codec:

```bash
PYTHONPATH=Runtime/python python - <<'PY'
from wust_io import read_artifact

manifest, arrays = read_artifact(
    "/absolute/output/reconstruction.json", "wust.reconstruction"
)
sound_speed_mps = arrays["c_mps"]
print(sound_speed_mps.shape)
print(manifest["metadata"]["completed_updates"])
PY
```

The same module provides `write_artifact(path, name, metadata, arrays, axes, units)` for producing input artifacts. Array fields and units are defined in [contracts.py](Runtime/python/contracts.py); [Runtime API](docs/RUNTIME_API.md) explains their physical metadata.

### Direct MATLAB calls

MATLAB callers can use the observation preparation and reconstruction functions directly. The example reuses the request above:

```matlab
addpath('Runtime/matlab');
request = jsondecode(fileread('/absolute/request.json'));
obs = wust_prepare_observations( ...
    wust_read_artifact(request.input_manifest, ...
        {'wust.frequency_input', 'wust.measurements'}));
initial = wust_read_artifact(request.initial_manifest, 'wust.initial_model');

cfg = request.config;
cfg.update_mask = initial.update_mask;
cfg.frequency_schedule_hz = request.frequency_schedule_hz;
cfg.max_updates = request.max_updates;
cfg.save_history = request.save_history;

result = wust_reconstruct(obs, initial.initial_mps, cfg);
sound_speed_mps = result.c_mps;
```

`wust_prepare_observations` prepares geometry, frequency order, receiver counts and observation energy. Reuse this prepared object for reconstruction; after editing pressure or mask, call the preparation function to refresh the derived values. `wust_prepare` supplies the corresponding RF-to-observations path.

## Data and results

### Array conventions

| Quantity | Layout and units |
| --- | --- |
| Sound speed / update mask | `[y,x]`, m/s / boolean |
| Grid | `shape_yx`, `spacing_yx_m`, `origin_yx_m`, `origin_kind: pixel_edge` |
| TX/RX coordinates | `[element,xy]`, meters |
| Frequency input | Complex total pressure in `[frequency,tx,rx]` or `[tx,rx,frequency]` order |
| Canonical observations | `Y[tx,rx,frequency]`, matching mask, strictly increasing Hz |
| RF | `pressure[time,rx,tx]` and the actual `time_s` samples |
| Canonical Fourier transform | $\Delta t\sum_n p(t_n)e^{-2\pi i f t_n}$ |
| Sampling indices | MATLAB one-based, column-major |

Pressure metadata declares `fourier_sign`, `pressure_type`, `data_units` and `spectrum_normalization`. Preparation conjugates positive-sign spectra when `real_pressure: true`; `Pa*s` uses `dtft_dt` normalization. Complex arrays are stored as real and imaginary HDF5 datasets.

TX and fitted RX sample grid points inside the outer Dirichlet ring. The update mask lies inside the PML-free region. Each evaluated transmitter/frequency uses at least two valid receivers for the complex source fit. Valid zero measurements participate in that fit.

BenchLab uses `[y,x]` position columns and `[frequency,tx,rx]` pressure. Its adapter converts positions to WUST's `[x,y]` convention at the protocol boundary.

### Frequency plan and saved outputs

| Setting or output | Meaning |
| --- | --- |
| `frequency_schedule_hz` omitted or `null` | One update per canonical frequency |
| Repeated Hz entries | Repeated updates at that frequency |
| Empty schedule or `max_updates: 0` | Return the initial model with zero updates |
| `max_updates` | Cap the number of scheduled updates |
| `save_history: true` | Save `history_mps[y,x,model_step]`, including the initial model |
| `c_mps[y,x]` | Final sound-speed array, float64 |
| `records` | Per-update loss, residual, step, work counts and timings |
| `completion` | Progress through the requested schedule and update budget |

Each update record reports the loss and residual at the model **before that update**, for its scheduled frequency. Set `config.evaluate_final_residual: true` to evaluate the final model over all canonical frequencies; the result includes the associated work and time. The default `final_data_residual` is `null`.

## Development

Use the following entry points for focused changes:

| Development task | Files to read and extend |
| --- | --- |
| Add or change observation preparation | `wust_prepare.m`, `wust_prepare_observations.m`, `wust_geometry.m` |
| Change the objective or gradient | `wust_oracle.m`, `directional_curvature.m` |
| Change model updates | `wust_reconstruct.m`, `wust_ncg_direction.m`, `wust_roi_gradient.m`, `wust_project_slowness_update.m` |
| Change wave discretization or backend | `HelmholtzSolver.m`, `ninePointCoefficients.m`, CUDA MEX sources |
| Extend protocol operations or fields | `Runtime/python/contracts.py`, Python/MATLAB consumers and generated schemas |

Read [AGENTS.md](AGENTS.md) for project development conventions. Match each numerical change with its relevant gradient, solver or end-to-end reference. Generate schemas after protocol changes:

```bash
python Runtime/python/export_schemas.py
python -m unittest discover -s Runtime/tests -p 'test_*.py'
```

Run MATLAB CPU numerical checks with:

```matlab
addpath('Runtime/tests');
test_runtime('cpu');
test_ingest_frequency;
test_resampling;
```

After building CUDA MEX, `test_runtime('gpu')` exercises the GPU path. The [validation guide](docs/validation.md) provides the public file-based integration command. The [simulation example](examples/simulate_and_reconstruct.m) covers simulation, RF preparation and reconstruction.

## Documentation, license and citation

- [Runtime API](docs/RUNTIME_API.md): requests, arrays, configuration and result fields.
- [Numerical contracts](docs/CONTRACTS.md): objective, derivatives, projection and scheduling.
- [Deployment](Runtime/README.md): MATLAB, CUDA MEX and native simulation.
- [中文流程说明](docs/PIPELINE_ZH.md): data flow and integration.
- [Attribution](ATTRIBUTION.md): source attribution and third-party terms.

The runtime is released under the [MIT License](LICENSE.txt). The bundled [k-Wave CUDA source](Runtime/third_party/kspaceFirstOrder-CUDA/License.md) uses LGPL-3.0-or-later. MATLAB and the k-Wave MATLAB toolbox retain their respective licenses.

This project derives from [Rehman Ali's WaveformInversionUST](https://github.com/rehmanali1994/WaveformInversionUST). Please cite:

> Rehman Ali et al., *2-D Slicewise Waveform Inversion of Sound Speed and Acoustic Attenuation for Ring Array Ultrasound Tomography Based on a Block LU Solver*, IEEE Transactions on Medical Imaging, 2024. [doi:10.1109/TMI.2024.3383816](https://doi.org/10.1109/TMI.2024.3383816).
