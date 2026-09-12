# k-Wave → frequency-domain FWI runtime (0.1.0)

Standalone runtime for a thin external adapter. No dependency on
`openbreastus_diffusion`, `kwave_dps`, PyTorch, or usct-benchlab.
Original upstream scripts are unchanged. Add only `Runtime/matlab` to MATLAB;
do NOT use `addpath(genpath(repo))`, which mixes reference and runtime solvers.

The algorithm is frequency-continuation **FWI**; Block-LU is its Helmholtz solver.
Supported: 2D, sound-speed inversion, fixed zero attenuation.

## Entry points

| Operation | MATLAB | Batch MAT variables |
|---|---|---|
| Generate RF | `rf=wfi_simulate(model,cfg)` | `model` |
| RF → frequencies | `obs=wfi_prepare(rf,cfg)` | `rf, prepare_config` |
| FWI | `result=wfi_reconstruct(obs,initial,cfg)` | `obs, initial_mps, update_mask` |
| Loss/gradient | `[loss,g,state]=wfi_oracle(c,obs,fi,cfg)` | direct MATLAB; g is slowness gradient |

Python (standard library only):
```sh
python Runtime/python/wfi_runtime.py /absolute/request.json --matlab matlab
```
JSON: `schema="wfi.request.v1"`, `operation="simulate"|"prepare"|"reconstruct"`,
absolute `input_mat/output_mat`, and `config`. Optional `kwave_toolbox_path`.
MAT output contains `result`: pass it as `rf` or `obs` to the next step.
Direct MATLAB/Engine calls avoid startup overhead. Logs/errors propagate; existing
outputs are refused; no best-GT checkpoint selection.

Executable complete example: **tests/smoke_pipeline.m**, which constructs every
required configuration field and writes an adapter request.

## Install and verify

MATLAB; Signal Processing Toolbox for FIR; Parallel Computing Toolbox and a
supported CUDA MEX compiler for GPU FWI. External k-Wave toolbox must be on path.
No administrator installation or environment mutation is performed.
```matlab
addpath('Runtime/matlab');
wfi_build_mex;
addpath('Runtime/tests');
test_runtime('cpu');
test_runtime('gpu');
test_resampling;
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
  --directory /absolute/path/to/new-wrapper-directory
```
Set simulation `backend='native128'`, explicit `binary_path` (wrapper directory)
and zero-based `device_num`. Ordered TX and RX must be the same 128 grid indices.
Supported physics: scalar density, linear propagation, zero absorption, 2D.
Unsupported HDF5 flags fail closed. PML is outside the caller grid.
RF and HDF5 intermediates are retained. Use an explicit disk work directory.

A100 80 GB is the intended full native128 target. Check RAM/VRAM before full jobs:
the generation output cube is still resident; FIR processes one TX at a time.
FWI source_batch_size controls RHS batching independently, but virtual sources
remain cached for the linearized step. No OOM fallback, GPU scheduler or automatic
/dev/shm allocation is hidden in this runtime.

- [Contracts](docs/CONTRACTS.md)
- [中文管线总结](docs/PIPELINE_ZH.md)
- [Provenance and licenses](docs/PROVENANCE.md)
- [Validation status](docs/VALIDATION.md)

Pin the compatibility commit SHA, not moving main/branch. Version 0.1.0 describes
the schema; it does not replace a Git pin. Benchlab integration is out of scope.
