# Final runtime validation

Executed 2026-10-30 on the isolated F01-F22 branch. MATLAB R2021b, CUDA 11.4,
NVIDIA A100-SXM4-80GB; Python 3.13.9, NumPy 2.3.5, SciPy 1.17.1, h5py 3.16.0.

## Results

- Complete Python suite: **17 passed, 10 subtests, no skips** (1.43 s).
  Process-group checks now inspect only successful ps queries.
- Ruff, Black and Python compileall pass. Maintained MATLAB files are multiline.
- MEX rebuilt successfully. CPU and GPU test_runtime, frequency ingestion and
  FIR resampling all pass with final remote MATLAB shell exit code 0.
- CPU FD errors: continuum **3.46637e-10**, LDR9 **1.97022e-10**.
- GPU FD errors: continuum **1.112e-3**, LDR9 **4.30214e-4**.
- Four-update loss: CPU 3.88345e-17 -> 8.95907e-18;
  GPU 3.88327e-17 -> 8.95982e-18. This is one small fixture, not a general
  monotonicity claim.
- FIR pass ratio 0.999878; alias rejection -72.39 dB.
- Public RF prepare: independent absolute-time DTFT relative error **2.51010e-8**.
  Includes nonzero time origin, valid zero and masked NaN channels.
- Native default make -B compiled and linked. Eight-plan shutdown is exercised
  by the real native-batch run. Native vs MATLAB relative errors:
  RF **1.7749e-6**, spectrum **8.0549e-7**; final process exit 0.
- Public WUST reconstruction: passed, one generated coherent fixture, one
  reconstruct invocation (10.838 s). BenchLab reused that same fixture for CPU
  and GPU run/save/evaluate/read: **10.734 s / 13.043 s**, both passed.

Earlier direct SSH calls returned 255 after printing passed; these were not
accepted as process-success evidence. Final remote-shell checks captured
MATLAB_SHELL_EXIT=0. nvcc warns about deprecated default architecture targets;
no attempt was made to change unrelated toolchain policy.

## Maintained commands

~~~bash
python -m pytest -q Runtime/tests
ruff check Runtime/python Runtime/tests Runtime/native
black --check Runtime/python Runtime/tests Runtime/native
python -m compileall -q Runtime/python Runtime/tests Runtime/native
python Runtime/tests/integration_runtime.py --matlab /path/to/matlab \
  --backend cpu --out /absolute/new/coherent-fixture
python Runtime/tests/integration_runtime.py --matlab /path/to/matlab \
  --prepare-only --out /absolute/new/rf-check
~~~

~~~matlab
addpath('Runtime/matlab');
addpath('Runtime/tests');
wust_build_mex;
test_runtime('cpu');
test_ingest_frequency;
test_resampling;
test_runtime('gpu');
~~~

See [deployment](../Runtime/README.md) for native build/launcher setup and
test_native_pipeline. Numeric checks and process exit must both pass.
The shared fixture is 23x27, two TX/four RX; the native comparison is 23x27,
three independent TX/four RX. No full-size imaging/performance campaign was run.

## Scope

The validation covers explicit model/discretization, source elimination,
preconditioned ROI direction, slowness projection, curvature, batched
forward/adjoint reuse, protocol axes/masks, RF tooling and resource lifetimes.
It does not establish clinical validity or equal CPU/GPU precision.
Schedule completion is not numerical convergence.

Per-unit progress is preserved in the sibling research archive; compact JUnit
and gate reports are outside source in evidence/final_f01_f22. No binaries or
raw results were added to Git. No push, PR or merge was performed.

Final artifact inspection found MATLAB's unrequested final residual encoded as
an empty array. The first Python manifest read now maps that sentinel to null,
without discarding computed residual objects. After the complete suites, the
11-test transport subset (8 subtests) and both public CPU/GPU integration gates
were rerun successfully with an explicit null assertion. No numerical formula
changed in this final serialization correction.
