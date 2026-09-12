# WUST runtime guide

Standalone runtime for a thin external adapter. No dependency on
`openbreastus_diffusion`, `kwave_dps`, PyTorch, or usct-benchlab.
Original upstream scripts are unchanged. Add only `Runtime/matlab` to MATLAB;
do NOT use `addpath(genpath(repo))`, which mixes reference and runtime solvers.

The algorithm is frequency-continuation **FWI**; Block-LU is its Helmholtz solver.
Supported: 2D, sound-speed inversion, fixed zero attenuation.

## Entry points

| Operation | MATLAB | External input schema |
|---|---|---|
| Generate RF | `rf=wust_simulate(model,cfg)` | `wust.simulation_input` |
| RF → frequencies | `obs=wust_prepare(rf,cfg)` | `wust.rf` |
| Frequency ingestion | `obs=wust_ingest_frequency(input)` | `wust.frequency_input` |
| FWI | `result=wust_reconstruct(obs,initial,cfg)` | `wust.measurements` + `wust.initial_model` |
| Loss/gradient | `[loss,g,state]=wust_oracle(c,obs,fi,cfg)` | direct MATLAB; g is slowness gradient |

Python (static discovery: standard library; batch arrays: NumPy/h5py):
```sh
python Runtime/python/wust_runtime.py describe --json
python Runtime/python/wust_runtime.py run /absolute/request.json --matlab matlab --timeout-s 300
```
The [runtime contract](../docs/RUNTIME_API.md) defines JSON manifests and explicit
HDF5 real/imag arrays, source identity, input snapshots and atomic finalization.
The launcher supervises a bounded MATLAB process group. Logs/errors propagate;
existing outputs are refused; no best-GT checkpoint selection.

Executable examples: `tests/smoke_pipeline.m` for direct MATLAB simulation and
`tests/integration_runtime.py` for the actual new external launcher.

## Install and verify

MATLAB; Signal Processing Toolbox for FIR; Parallel Computing Toolbox and a
supported CUDA MEX compiler for GPU FWI. External k-Wave is needed only for simulation.
No administrator installation or environment mutation is performed.
```matlab
addpath('Runtime/matlab');
addpath('Runtime/tests');
test_runtime('cpu');
test_resampling;
test_path_isolation;
```

Optional GPU and simulation checks (install their dependencies first):

```matlab
wust_build_mex;
test_runtime('gpu');
smoke_pipeline;
```
Native HDF5 wrapper tests:
```sh
python -m pip install numpy h5py
python Runtime/tests/test_python.py
```

## Native128 generation (Linux GPU)

Modified **LGPL-3.0-or-later** source is included under
`third_party/kspaceFirstOrder-CUDA`; preserve its notices.
This is 128 independent TX wavefields with rank-2 cuFFT planMany, NOT simultaneous
physical excitation in one wavefield.

From repository root:
```sh
make -C Runtime/third_party/kspaceFirstOrder-CUDA -j4 \
  CUDA_DIR=/path/to/cuda HDF5_DIR=/path/to/hdf5 \
  CUDA_ARCH='--generate-code arch=compute_80,code=sm_80'
python Runtime/native/install_launcher.py \
  --binary /absolute/path/to/kspaceFirstOrder-CUDA \
  --directory /absolute/path/to/new-wrapper-directory \
  --library-dir /path/to/hdf5/lib --library-dir /path/to/cuda/lib64
```
Set simulation `backend='native128'`, explicit `binary_path` (wrapper directory)
and zero-based `device_num`. Ordered TX and RX must be the same 128 grid indices.
Supported physics: scalar density, linear propagation, zero absorption, 2D.
Unsupported HDF5 flags fail closed. PML is outside the caller grid.
The library directories are important: k-Wave clears LD_LIBRARY_PATH when it
launches external binaries. The wrapper restores only the explicitly configured
directories. The Makefile also passes CUDA_ARCH at device-link time, not just
compilation; omitting it can yield a binary without the target GPU kernels.
RF and HDF5 intermediates are retained. Use an explicit disk work directory.

A100 80 GB is the intended full native128 target. Check RAM/VRAM before full jobs:
the generation output cube is still resident; FIR processes one TX at a time.
FWI source_batch_size controls RHS batching independently, but virtual sources
remain cached for the linearized step. No OOM fallback, GPU scheduler or automatic
/dev/shm allocation is hidden in this runtime.

- [Contracts](../docs/CONTRACTS.md)
- [中文管线总结](../docs/PIPELINE_ZH.md)
- [Provenance and licenses](../provenance/README.md)
- [Historical validation](../provenance/historical-validation/VALIDATION.md)
- [Stage 4A1 validation](../docs/STAGE4A1_VALIDATION.md)

Pin the runtime commit SHA, not a moving branch. Runtime/VERSION is the version
authority. New schema versions are separate integers. BenchLab integration is
deferred to Stage 4B; the old MAT request parser is not maintained in parallel.
