# WaveformInversionUST (WUST)

WUST is an independent MATLAB runtime for **2D sound-speed full-waveform
inversion**. It uses frequency continuation, complex source-scale elimination,
nonlinear conjugate-gradient updates and a nine-point Helmholtz discretization.
CPU solves use sparse linear algebra; optional GPU solves use CUDA Block-LU MEX.
This is research software, not a clinically validated reconstruction system.

See the [runtime API](docs/RUNTIME_API.md) and
[Stage 4A2 validation evidence](docs/STAGE4A2_VALIDATION.md).

## Maintained runtime

`Runtime/` is the maintained implementation. It does not import BenchLab,
PyTorch, diffusion packages or any consumer repository. Reconstruction consumes
supplied observations; RF preparation and simulation are separate data tools.
Active reconstruction estimates sound speed with attenuation fixed to zero.

```text
Runtime/
  VERSION                 # authoritative runtime version
  python/wust_runtime.py  # external MATLAB batch launcher
  matlab/wust_*.m         # preparation, reconstruction, oracle, setup, simulation
  solver/                 # maintained Helmholtz and optional CUDA Block-LU
  tests/                  # CPU, GPU and optional simulation checks
  native/                 # optional native simulation tooling
  third_party/            # separately licensed k-Wave CUDA source
docs/                     # current contracts and usage
provenance/               # upstream identity, source map and manifests
reference/upstream-9fb31657/  # original upstream examples, not a production backend
```

## Requirements and CPU checks

MATLAB is required for FWI. Signal Processing Toolbox is needed for FIR
resampling. GPU FWI additionally needs Parallel Computing Toolbox and a supported
CUDA MEX compiler. The external k-Wave toolbox and native simulator are optional
simulation dependencies, not requirements for reconstructing existing data.

From the repository root:

```sh
python -m pip install numpy h5py
python -m unittest discover -s Runtime/tests -p 'test_*.py'
python Runtime/python/export_schemas.py --check
python provenance/check_manifest.py
matlab -batch "addpath('Runtime/tests'); test_runtime('cpu'); test_resampling; test_path_isolation; test_ingest_frequency; test_batch_contract;"
```

Add **only** `Runtime/matlab` to the production MATLAB path. `wust_setup` selects
the maintained solver and rejects path shadowing. Never recursively add this
repository to the path. Run historical examples in a separate MATLAB process.

## Run FWI

```matlab
addpath('Runtime/matlab');
result = wust_reconstruct(obs, initial_mps, config);
```

`obs` contains supplied complex pressure, frequencies, geometry and a channel
mask. `initial_mps` is a sound-speed image on that geometry. `config` explicitly
specifies the frequency-index schedule, numerical controls and update mask.
See [the external runtime contract](docs/RUNTIME_API.md) for JSON/HDF5 schemas,
capabilities, units, array axes and required policies;
[the runtime guide](Runtime/README.md) covers the batch MAT interface, optional
GPU build and simulation. [The CPU test](Runtime/tests/test_runtime.m) is a
complete finite-input reconstruction example with every required configuration
field. Its generated observations are a numerical fixture, not measured data.

```sh
python Runtime/python/wust_runtime.py describe --json
python Runtime/python/wust_runtime.py run /absolute/request.json --matlab matlab --timeout-s 300
```

The maintained protocol uses `wust.*` schema names plus integer `schema_version`.
Historical `wfi.*.v1`/MAT requests are rejected with migration guidance. Direct
frequency ingestion and RF preparation share WUST geometry and indexing rules.
Pin a commit SHA alongside `Runtime/VERSION`; schedule completion does not
establish numerical convergence. No attenuation reconstruction is supported.

## Attribution and licenses

Derived from [Rehman Ali's WaveformInversionUST](https://github.com/rehmanali1994/WaveformInversionUST)
at `9fb31657c4ba9141a2bdf5c2eeac644320d090e8`. The original README, examples,
solver and figures remain byte-preserved under [reference](reference/upstream-9fb31657/README.md).
See [provenance](provenance/README.md) for the maintained modifications and
[the migration record](docs/MAINLINE_MIGRATION.md) for archive and validation policy.

Please cite Rehman Ali et al., *2-D Slicewise Waveform Inversion of Sound Speed
and Acoustic Attenuation for Ring Array Ultrasound Tomography Based on a Block LU
Solver*, IEEE TMI, [doi:10.1109/TMI.2024.3383816](https://doi.org/10.1109/TMI.2024.3383816).
The paper's attenuation reconstruction is not an active feature of this runtime.

[LICENSE.txt](LICENSE.txt) is MIT. The vendored
[k-Wave CUDA source](Runtime/third_party/kspaceFirstOrder-CUDA/License.md) is
**LGPL-3.0-or-later** with its own copyright and redistribution requirements;
it is not relicensed as MIT. External MATLAB and k-Wave toolboxes are not included.
