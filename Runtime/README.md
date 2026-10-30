# Runtime deployment

## Dependencies and backends

- Python 3.12+, NumPy and h5py: protocol transport and process ownership.
- MATLAB: CPU sparse Helmholtz reference solver.
- Parallel Computing Toolbox and supported CUDA MEX compiler: GPU Block LU.
- Signal Processing Toolbox: anti-aliased RF decimation.
- External k-Wave toolbox: simulation only.

Backend selection is explicit; GPU failures do not silently fall back to CPU.
Each HelmholtzSolver owns its backend. wust_setup changes paths, not global backend state.
Solver construction requires explicit `backend` and fixed, ordered
`stencil_bounds=[min,max]`; neither is inferred from the current model.

~~~matlab
addpath('Runtime/matlab');
wust_build_mex;
~~~

Normal Python checks do not start MATLAB:

~~~bash
python -m unittest discover -s Runtime/tests -p 'test_*.py'
~~~

Explicit numerical gates:

~~~matlab
addpath('Runtime/tests');
test_runtime('cpu');
test_ingest_frequency;
test_resampling;
% After building CUDA MEX:
test_runtime('gpu');
~~~

The same CPU/GPU test checks continuum/LDR9 gradients, source batching, a small
FWI update and schedule/zero/truncation semantics. See [API](../docs/RUNTIME_API.md)
for the one-process reconstruction protocol.

## Optional native simulation (Linux CUDA)

The maintained native-batch backend propagates independent TX wavefields with
shared medium and rank-2 batched FFTs. TX and RX may have different ordered
positions. It is not simultaneous physical excitation. Supported medium is 2-D,
linear, scalar-density and zero absorption. Unsupported flags fail explicitly.

~~~bash
make -C Runtime/third_party/kspaceFirstOrder-CUDA -j4 all \
  CXX=/path/to/cuda/bin/nvcc CUDA_DIR=/path/to/cuda HDF5_DIR=/path/to/hdf5 \
  CUDA_ARCH='--generate-code arch=compute_80,code=sm_80'
python Runtime/native/install_launcher.py --batch-size 3 \
  --binary /absolute/path/to/kspaceFirstOrder-CUDA \
  --directory /absolute/new/launcher \
  --library-dir /path/to/hdf5/lib --library-dir /path/to/cuda/lib64
~~~

Use a CUDA toolkit compatible with the installed driver/GPU. The caller chooses
batch size, device_num and work_directory; no global CUDA installation, hidden
scheduler or automatic RAM-disk allocation is performed. k-Wave clears its
child library path, so the launcher restores only the declared library folders.

Simulation config uses backend='native-batch', binary_path pointing to the
launcher directory and an explicit zero-based device_num. Each TX source is
scaled with its own local speed using k-Wave's pressure-source convention.
PML is outside the supplied grid. RF intermediates are retained in the requested
work directory. Budget disk/RAM/VRAM before larger runs.

The maintained hardware correctness gate uses 3 unordered TX and 4 RX on a
non-square heterogeneous grid, comparing against MATLAB k-Wave:

~~~matlab
addpath('/path/to/k-Wave');
addpath('Runtime/tests');
test_native_pipeline('/absolute/new/launcher', '/absolute/new/native-check', 0);
~~~

Full-size performance campaigns are in [research archive](../docs/research_index.md).
When running this gate with MATLAB -batch, check both the report and process exit
status; numerical assertions alone do not certify successful process cleanup.
[examples](../examples/simulate_and_reconstruct.m) demonstrates explicit simulation,
RF preparation and reconstruction; it is not a required CI stage.
