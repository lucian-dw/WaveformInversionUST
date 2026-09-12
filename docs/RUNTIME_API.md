# WUST external runtime contract

Runtime semantic version is read from `Runtime/VERSION`. Serialized schema
names are independent of that version and carry integer `schema_version: 1`.
The historical `wfi.*.v1` and MAT-request protocol is rejected, not auto-converted.
Re-export inputs using this format, or use the historical Git tag to reproduce
old requests. This development version is not a Stage 4C release certification.

## Commands and dependencies

```sh
python Runtime/python/wust_runtime.py describe --json
python Runtime/python/wust_runtime.py describe --json --probe --matlab matlab --timeout-s 60
python Runtime/python/wust_runtime.py run /absolute/request.json --matlab matlab --timeout-s 300
```

Python 3.10+ is supported. Static discovery uses the standard library only and
does not launch MATLAB or allocate GPU memory. Batch transport uses NumPy and
h5py: `python -m pip install numpy h5py`. MATLAB is required for all four batch
operations. SciPy is needed only for before/after regression-file comparison.
Logs go to stderr; successful CLI stdout is one JSON value.

Static `supported` facts are not deployment `available` facts. `--probe` executes
a small CPU Helmholtz solve and, when possible, a small GPU solve; it reports
MATLAB/platform, GPU/CUDA properties, MEX paths/hashes and failure reasons.
An optional simulation binary is not discovered by guessing installation paths.

| Operation | Input schema | Output schema | Role |
|---|---|---|---|
| `ingest_frequency` | `wust.frequency_input` | `wust.measurements` | Production observation ingestion |
| `reconstruct` | `wust.measurements` + `wust.initial_model` | `wust.reconstruction` | SOS FWI |
| `prepare` | `wust.rf` | `wust.measurements` | RF data tooling |
| `simulate` | `wust.simulation_input` | `wust.rf` | Fixture/dataset tooling only |

`wust_oracle` remains a direct MATLAB numerical-validation function, not a
production reconstruction variant. Reconstruction never calls simulation and
never accepts a ground-truth model, an initialization strategy or artifact ID.

## Request envelope

```json
{
  "schema": "wust.request",
  "schema_version": 1,
  "operation": "ingest_frequency",
  "input_manifest": "/absolute/frequency.json",
  "output_manifest": "/absolute/measurements.json",
  "config": {}
}
```

Unknown request/config fields fail. Manifest paths must be absolute, the output
parent must exist, and existing outputs are never overwritten. Reconstruction
also requires `initial_manifest` and an explicit `schedule` of 1-based indices
into **post-ingestion** frequencies. `planned_schedule_length` is optional and
must be at least the supplied schedule length. It labels caller truncation;
WUST neither constructs nor extends the caller's schedule.

## JSON manifest and HDF5 arrays

Every array artifact has exactly these top-level fields:
`schema`, `schema_version`, `metadata`, `arrays_file`, `arrays_sha256`, `arrays`.
`arrays_file` names a sibling HDF5 file. SHA-256 is checked before execution.
External/virtual HDF5 datasets are not accepted. Each named array descriptor has
`axes` (comma-separated), `shape`, `dtype`, `units`, and either `dataset` or the
pair `real_dataset`/`imag_dataset`. Complex arrays always use explicit real/imag
datasets. Supported storage types are float32/64, complex64/128, int64, uint8 and
bool (stored as uint8 with values 0/1). HDF5 shape follows the declared axes;
MATLAB dimension reversal is handled exclusively in WUST transport helpers.

Use `Runtime/python/wust_io.py::write_artifact` and `read_artifact` rather than
MAT struct-layout guessing. Structural JSON Schemas in `Runtime/schemas/` are
generated from `contracts.py` and `export_schemas.py`; physical cross-field
constraints are enforced by the Python/MATLAB validators, not JSON Schema alone.

### Frequency input

Required arrays: `pressure`, `mask`, `frequencies_hz`, `tx_xy_m`, `rx_xy_m`.
Required metadata:

```json
{
  "grid": {
    "shape_yx": [23, 27],
    "origin_yx_m": [-0.0115, -0.02],
    "spacing_yx_m": [0.001, 0.0015],
    "origin_kind": "pixel_edge"
  },
  "fourier_sign": 1,
  "real_pressure": true,
  "pressure_type": "total_pressure",
  "data_units": "instrument_units",
  "spectrum_normalization": "dtft_dt",
  "measurement_provenance": "external_measurement"
}
```

- Models are `[y,x]`; coordinates are `[x,y]` metres. Pixel centres are
  `origin + (index + 0.5) * spacing`. No resize, transpose or domain expansion.
- Pressure axes are `frequency,tx,rx` or `tx,rx,frequency`. Mask may use either
  full layout or explicitly declared `tx,rx`, broadcast across frequencies.
- Frequencies are finite, positive and unique. Duplicates fail. WUST sorts
  ascending and applies the same permutation to pressure and mask; original
  frequencies and the canonical-to-input 1-based permutation are recorded.
- WUST uses the negative-sign DTFT and Helmholtz sign -1. Explicit sign +1 is
  conjugated **once**, only with `real_pressure: true`. Unknown signs fail.
- Input must be total pressure, not a ratio, scattered pressure or travel time.
  Units are `Pa*s`, `instrument_units` or explicitly `unknown`; normalization
  is `dtft_dt`, `discrete_sum` or explicitly `unknown`. `Pa*s` requires `dtft_dt`.
  No amplitude normalization is inferred or silently applied.
- Valid zeros remain valid. Valid NaN/Inf fails. Masked invalid values are
  zeroed for computation only after preserving the mask.
- Positions must be within the grid's sampling-coordinate extent. Nearest-node
  snapping uses the lowest index on an exact distance tie. Same-array elements
  collapsing to one node fail; TX/RX overlap across the two arrays is allowed.
- Actual and snapped positions, MATLAB 1-based column-major indices, source
  precision and preprocessing decisions are saved. Ingestion preserves
  complex64/128; compute casting is reported separately.

### Reconstruction

`wust.initial_model` contains only `initial_mps[y,x]` (m/s) and
`update_mask[y,x]` (1). The initial image must be finite, positive, exactly match
the measurement grid, and lie within the configured bounds. No scalar expansion
or initialization selection occurs in WUST. The update mask cannot include the
PML/Dirichlet boundary; it is never silently clipped.

All numerical settings are explicit; there are no calibrated quality presets:

| Config field | Type / legality | Meaning |
|---|---|---|
| `backend` | `cpu` / `gpu` | Sparse float64 / CUDA complex64 |
| `bounds_mps` | two increasing positive numbers | Sound-speed projection bounds |
| `stencil_bounds` | two increasing positive numbers | Frozen stencil optimization bounds, m/s |
| `pml_m`, `pml_strength` | positive finite numbers | Thickness in metres / dimensionless strength |
| `max_update_mps` | positive finite number | Velocity-step projection cap |
| `step_damping` | positive finite number | Linearized-step multiplier |
| `source_batch_size` | positive integer | TX RHS batch size |
| `filter_cutoff` | nonnegative finite number | Existing dimensionless radial cutoff; zero disables |
| `filter_order` | positive integer | Existing filter order |
| `wavenumber` | `continuum` / `kwave-ldr9` | Discrete wavenumber model |
| `dispersion` | required only for LDR9 | Positive `time_step_s`, `reference_speed_mps`, `model_reference_speed_mps` |

LDR9 time step means the original simulator step, not decimated RF dt. There are
no forward/adjoint caps, `update_rtol`, objective plateau or GT-based stopping.
One schedule entry is one model update; repeated indices repeat updates and
frequency changes preserve the existing NCG restart semantics.

Per-frequency/per-TX complex source-scale elimination uses only the supplied
mask. At least two fitting receivers per scheduled TX/frequency and a usable
finite denominator are required. No water reference or source spectrum is
required. This does not correct directionality/general model mismatch and does
not implement a separate hold-out fit mask or independent final evaluation.

### Preparation and simulation

RF arrays: `pressure[time,rx,tx]`, `time_s[time]`, TX/RX coordinates. Metadata
declares `data_units` (`Pa` or `instrument_units`) and measurement provenance.
Preparation config supplies `grid`, `frequencies_hz`, `c_geom_mps`, `window`,
`phase_correction`, and a 0/1 mask. Existing `none`/`legacy-nominal` window and
`none`/`homogeneous-tof` correction policies remain. Geometry and canonical
measurement construction are shared with direct ingestion, including the
recorded physical nonzero time origin. Time sampling must be finite and uniform.
RF traces used at any frequency must be finite; wholly masked receiver traces
are zeroed before the transform, retaining their explicit false mask.

Simulation input arrays: `c_mps[y,x]`, TX/RX coordinates and `source_pressure[time]`;
metadata supplies `grid` and `dt_s`. Simulation config is the existing explicit
backend/PML/density/reference-speed/decimation/time-offset/work-directory config.
`--kwave-toolbox-path` is an explicit simulation-only deployment option. See the
runtime guide for optional native binary setup; it never becomes a reconstruct
parameter. Simulated RF is labeled `self_simulated`.

## Completion, supervision and publication

The launcher requires an explicit timeout for each invocation. It deducts
admission/snapshot work before starting MATLAB; it does not renew a multi-stage
budget. On POSIX it terminates the entire MATLAB process group on timeout/error,
including descendants. Current supervised platforms are Linux and macOS; Windows
execution is rejected. Process-tree termination has a bounded cleanup grace.

| Outcome | Meaning |
|---|---|
| `schedule_complete` | Supplied complete plan executed; not convergence |
| `caller_truncated_schedule` | Supplied prefix executed; budget category |
| `zero_updates` | Validated initial model returned without solver updates |
| `time_budget` | Deadline expired; no successful final artifact |
| `process_termination` | Launcher received TERM/INT/HUP; supervised children terminated |
| `numerical_failure` | Degenerate fit or invalid numerical state; no success |
| `incompatible_request` / `incompatible_runtime` | Admission rejection |
| `runtime_failure` / `incomplete_output` | MATLAB/process/output failure |

`wust.reconstruction` includes authoritative float64 `c_mps`, float32 history,
executed schedule, complete-update count, records, resolved config, selection,
completion and provenance. Record losses/residuals are **before update** and
carry `loss_model_step`. They are not final-model residuals; `final_data_residual`
is null. Diagnostic relative update is L2 slowness change divided by old slowness
norm on the declared mask, in s/m; it does not activate a stopping threshold.

Inputs are hash-checked and copied into a private invocation snapshot. Arrays are
written to private staging, validated and hashed; the success JSON is atomically
published last with no-overwrite semantics. Failed output is never published as
success. Hard external kills may leave hidden staging, but no final success
manifest. These are trusted local runtime requests, not an arbitrary-code sandbox.

## Identity and release archives

Results include runtime/schema versions, upstream identity, source-manifest hash,
Git SHA/dirty status when detectable, MATLAB/platform, backend/precision, MEX
hashes where used, input/request/initial/array hashes and preprocessing metadata.
Dirty or hash-mismatched source trees are rejected by default. `--allow-dirty`
is an explicit research override and is recorded.

Without `.git`, observed Git fields are null, not guessed from a directory name.
Source-manifest identity still works. From a clean verified source checkout,
export a build record to the assembled release package:

```sh
python Runtime/python/export_build_manifest.py /absolute/package/provenance/build-manifest.json
```

The build record binds its declared revision to the source-manifest digest and
is validated when present. It is not itself a hashed source file (avoids a
self-referential digest). Consumers must separately pin an approved revision.

## Reproducible CPU integration

```sh
python -m unittest discover -s Runtime/tests -p 'test_*.py'
python Runtime/python/export_schemas.py --check
python provenance/check_manifest.py
matlab -batch "addpath('Runtime/tests'); test_runtime('cpu'); test_resampling; test_path_isolation; test_ingest_frequency; test_batch_contract;"
python Runtime/tests/integration_runtime.py --matlab matlab --out /absolute/new-artifacts
```

The integration tool uses explicit fixtures and a recorded dirty-tree override
so it can test a PR before committing. It is not a production orchestration API.
No MATLAB/GPU is needed for the ordinary Python unittest suite.
