# Current runtime validation

Executed on 2026-10-30 for R01-R20 on `codex/round3-r01-r20`. This record
replaces the prior F01-F22 snapshot's validation numbers; history remains in Git.

## Local Python and static checks

macOS arm64, Python 3.12.2, NumPy 2.5.4, h5py 3.16.0:

- Complete pytest -q Runtime/tests: **18 passed, 12 subtests passed**, 1.17 s.
- Ruff/Black: all maintained Python files passed.
- MISS_HIT 0.9.44, MATLAB 2021b: 10 changed MATLAB files, no findings.
- git diff --check: passed.

Python tests exercise real temporary JSON/HDF5, launcher/process boundaries and
native source expansion without requiring MATLAB or GPU. They do not stand in
for the hardware gates below. JSON null survives the first codec read; a failed
publication leaves no success manifest. No reader compatibility rewrite remains.

## A100 hardware gates

Fresh mirror on host im2; MATLAB R2021b, NVIDIA A100-SXM4-80GB, CUDA 11.4
compiler, Python 3.13.9. No old remote checkout, data or result was replaced.

| Entry point | Executed result |
| --- | --- |
| test_runtime('cpu') | continuum FD 1.23141e-11; LDR9 FD 3.81486e-10 |
| CPU update objective | 3.88345e-17 -> 8.95907e-18 |
| test_ingest_frequency | canonical wire/prepared roundtrip, sign/order, Y/mask refresh passed |
| test_resampling | passband ratio 0.999878; alias rejection -72.39 dB |
| Rebuild decompBlockLU / applyBlockLU MEX | passed; deprecated GPU-target compiler warnings only |
| test_runtime('gpu') | continuum FD 1.11200e-3; LDR9 FD 4.30214e-4 |
| GPU update objective | 3.88327e-17 -> 8.95982e-18 |
| Rebuilt native binary, test_native_pipeline | RF relative L2 1.77488e-6; spectrum relative L2 8.05494e-7 |
| Public RF preparation vs independent absolute-time DTFT | relative L2 2.51010e-8 |

CPU/GPU tests also exercise outer-Dirichlet rejection, permitted internal PML
sampling, all-frequency final-evaluation admission (including zero updates),
loss-only equality, explicit solver arguments and MATLAB-produced JSON null.
Final loss uses zero adjoint RHS. State is cleared after its last per-update
consumer; the resource test scope is the existing numerical/MEX entry points,
not a long-running memory profiler.

The native fixture is asymmetric 23x27, 3 unordered independent TX and 4 RX.
Actual native/serial times: 3.503/1.344 s; this small correctness gate is not a
speedup claim. The launcher uses batch size 3 and the rebuilt current native
binary. The earlier shell quoting failure during the make invocation was corrected
without code changes; the final build and native comparison exited 0.

## Shared public fixture with BenchLab

integration_runtime.py generated one coherent canonical fixture: 23x27,
2 TX/4 RX, 2 model updates from a 3-entry schedule. WUST's file reconstruction
passed. BenchLab then reused the same fixture on both explicit backends:

| Backend | Reconstruction time | Public chain |
| --- | ---: | --- |
| CPU | 11.185 s | run -> numerical save -> evaluate -> first read passed |
| GPU | 16.273 s | run -> numerical save -> evaluate -> first read passed |

Each BenchLab run launched MATLAB reconstruction once and generated zero new
fixtures. Its positive-DTFT/reversed-frequency input tests WUST-owned conversion
and sorting. Missing GT remains unavailable. Disabled final residual is null;
schedule truncation reports budget completion, not convergence. All final MATLAB
gates and their outer SSH commands returned exit 0, not merely printed success.

## Commands

From the runtime repository (use a fresh output directory for each public run):

~~~bash
python -m pytest -q Runtime/tests
python -m ruff check Runtime/python Runtime/native Runtime/tests
python -m black --check Runtime/python Runtime/native Runtime/tests
python Runtime/tests/integration_runtime.py --matlab /path/to/matlab \
  --backend cpu --out /absolute/new/coherent-fixture
python Runtime/tests/integration_runtime.py --matlab /path/to/matlab \
  --prepare-only --out /absolute/new/rf-check
~~~

MATLAB, with Runtime/tests on the path:

~~~matlab
test_runtime('cpu');
test_ingest_frequency;
test_resampling;
wust_build_mex;
test_runtime('gpu');
test_native_pipeline('/path/to/native-launcher', '/absolute/new/native-check', 0);
~~~

See [deployment](../Runtime/README.md) for CUDA/HDF5 build settings and the
native launcher, and BenchLab scripts/validate_wust_integration.py for the
shared public chain. Compact gate JSON/JUnit is retained outside source in the
sibling evidence directory. No new simulation/reconstruction arrays or binaries
are tracked.

## Limits

GPU uses complex single precision; CPU-double FD accuracy is not claimed for
GPU. These small fixtures validate the stated discretization, source projection,
mask/domain handling, lifetime paths and transport. No full-size dataset,
clinical validity, imaging-quality or throughput campaign was run. Completion
of a frequency schedule is completion, not numerical convergence.
