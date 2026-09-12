# WUST runtime guide

Standalone runtime for a thin external adapter. No dependency on
`openbreastus_diffusion`, `kwave_dps`, PyTorch, or usct-benchlab.
Original upstream scripts are unchanged. Add only `Runtime/matlab` to MATLAB;
do NOT use `addpath(genpath(repo))`, which mixes reference and runtime solvers.

The algorithm is frequency-continuation **FWI**; Block-LU is its Helmholtz solver.
Supported: 2D, sound-speed inversion, fixed zero attenuation.

## Entry points

| Operation | MATLAB | Batch MAT variables |
|---|---|---|
| Generate RF | `rf=wust_simulate(model,cfg)` | `model` |
| RF → frequencies | `obs=wust_prepare(rf,cfg)` | `rf, prepare_config` |
| FWI | `result=wust_reconstruct(obs,initial,cfg)` | `obs, initial_mps, update_mask` |
| Loss/gradient | `[loss,g,state]=wust_oracle(c,obs,fi,cfg)` | direct MATLAB; g is slowness gradient |

Python (standard library only):
```sh
python Runtime/python/wust_runtime.py /absolute/request.json --matlab matlab
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
authority; serialized schema identifiers remain unchanged during Stage 4A1.
BenchLab integration and the new external protocol are deferred to later stages.
