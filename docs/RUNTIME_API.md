# WUST runtime protocol 2

WUST is independent of BenchLab. `Runtime/VERSION` supplies the software version;
`Runtime/python/contracts.py` supplies protocol version 2 and the array field
table. `export_schemas.py` generates seven JSON schemas consumed by tooling and
MATLAB serialization. There are no source/build identity or dirty-tree gates.

## Operations

| Operation | Input | Output |
| --- | --- | --- |
| `reconstruct` | `wust.frequency_input` or `wust.measurements`, plus `wust.initial_model` | `wust.reconstruction` |
| `ingest_frequency` | `wust.frequency_input` | `wust.measurements` |
| `prepare` | `wust.rf` and explicit preparation config | `wust.measurements` |
| `simulate` | `wust.simulation_input` and simulation config | `wust.rf` |

Production reconstruction uses supplied observations. RF preparation and physical
simulation are explicit tooling, not implicit reconstruction stages. Normal FWI
does not run separate ingestion or availability-probe MATLAB processes.

```bash
python Runtime/python/wust_runtime.py describe --json
python Runtime/python/wust_runtime.py run /absolute/request.json \
  --matlab /path/to/matlab --timeout-s 300
```

Optional `describe --probe --backend cpu` checks only the requested backend
explicitly; use `--backend gpu` for CUDA MEX. Logs go to stderr;
stdout is one JSON result. Each invocation has a separate run directory. Input
paths must remain stable during execution. Arrays close before the successful
manifest is published. Failed runs retain their request/partial artifacts and a
separate failure record; they do not publish a success index.

## Reconstruction request

```json
{
  "schema": "wust.request",
  "schema_version": 2,
  "operation": "reconstruct",
  "input_manifest": "/absolute/frequency.json",
  "initial_manifest": "/absolute/initial.json",
  "output_manifest": "/absolute/new-result.json",
  "frequency_schedule_hz": [100000, 100000, 120000],
  "max_updates": 2,
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

These are fixture parameters, not validated clinical defaults. Schedule null or
omission means each canonical frequency once; an empty list means no updates.
Explicit repetitions repeat updates. MATLAB maps exact Hz values after preparing
observations and then truncates by `max_updates`. Null/omitted budget leaves the
plan intact; zero performs no updates. `save_history` defaults to false.

For an already canonical input, use the same request with input_manifest pointing
to a wust.measurements artifact. No other reconstruction field changes. An
independent ingest_frequency request is needed only when saving reusable canonical
observations, not before every reconstruct call.

Speed bounds are positive and ordered. PML and maximum speed update are in m and
m/s. Batch size and filter order are positive integers. The dimensionless positive
filter cutoff multiplies `2*f/c0`; zero is identity. `kwave-ldr9` requires explicit
`dispersion` with `time_step_s`, `reference_speed_mps` and
`model_reference_speed_mps`. Continuum mode does not use these simulator facts.

## Artifacts

Each JSON document has `schema`, `schema_version`, `metadata`, `arrays_file` and
`arrays`. HDF5 arrays use fixed dataset names (`/name`, or `/name_real` and
`/name_imag`); descriptors declare axes, units, dtype and shape. `arrays_file` is
a sibling filename. No hashes or copied input snapshots are required.

Images are `[y,x]`. Grid metadata declares `shape_yx`, `spacing_yx_m`,
`origin_yx_m` and `origin_kind: pixel_edge`. Coordinates are physical `[x,y]`.
MATLAB storage dimension reversal is confined to artifact read/write functions.

Frequency input carries complex total `pressure`, explicit `mask`, frequencies
and TX/RX positions. Pressure axes are `frequency,tx,rx` or `tx,rx,frequency`;
mask can additionally be `tx,rx`. Required physical metadata declares
`pressure_type: total_pressure`, `fourier_sign`, `data_units` and
`spectrum_normalization`. Positive-sign DTFT conversion requires
`real_pressure: true`. `Pa*s` requires `dtft_dt` normalization; otherwise use
explicit instrument units or `unknown`. Provenance is acquisition information,
not a source-code identity. Invalid channels are zeroed before arithmetic;
valid zero pressure remains valid data.

`wust_prepare_observations` owns geometry, sorting, conjugation and masks.

RF `prepare` requests pass the explicit `grid` unchanged to `wust_prepare`.
Derived coordinates are not additional configuration fields. The public path
can be checked independently of reconstruction with
`python Runtime/tests/integration_runtime.py --matlab /path/to/matlab --out /tmp/wust-rf-check --prepare-only`.

Reconstruction masks the raw slowness gradient both before and after spatial
filtering (`M F M g`). The NCG direction uses preconditioned clipped PR/FR
inner products and restarts on a frequency change or non-descent direction.
Trials are projected in slowness to the joint global and per-step speed bounds
before taking the reciprocal. Each record reports the absolute pre-update
pressure residual. Its relative residual is unavailable (`null`, status
`zero_observation_energy`) when observations have zero energy, not divided by
an arbitrary epsilon floor. Schedule completion is not a convergence claim.
Canonical measurements have `Y[tx,rx,frequency]`, matching mask, strictly sorted
frequencies, physical positions and one-based sampling indices. Derived grid
coordinates are computed from grid, not stored as duplicate protocol arrays.
Nearest-grid ties use the lowest index; duplicate same-array snapped elements
are rejected. Source scale is eliminated per TX/frequency using at least two
valid receivers. No external source spectrum or water reference is required.

Initial model has `initial_mps[y,x]` and `update_mask[y,x]`. Shapes, positive speed
bounds and exclusion of PML/Dirichlet boundaries are checked in MATLAB; no silent
resizing, clipping or GT initialization is performed.

## Results and budgets

Final `c_mps[y,x]` is float64; optional `history_mps[y,x,model_step]` is float32.
Metadata includes actual version/backend/precision, complete planned length,
completed updates, canonical/executed frequencies, observation preparation and
per-update records. `loss_before_update` describes that frequency at the previous
model, never the final model. An uncomputed final residual is null.

One schedule entry is one full-slowness update, units s/m. Relative L2 change is
measured over the declared update mask and normalized by old slowness norm.
`schedule_complete` is completion, not convergence; `zero_updates` and
`caller_truncated_schedule` are budget outcomes. No online convergence,
forward/adjoint caps or `update_rtol` are advertised. Hard elapsed deadline
terminates the owned process group; structured failures distinguish time budget,
numerical failure and runtime failure.

## Validation

```bash
python -m unittest discover -s Runtime/tests -p 'test_*.py'
python Runtime/tests/integration_runtime.py --matlab /path/to/matlab \
  --backend cpu --out /absolute/new/runtime-acceptance
```

The manual integration shares a coherent non-square fixture with BenchLab's
`scripts/validate_wust_integration.py`. Fixture generation is separately recorded.
MATLAB numerical tests cover source-scale-projected gradients, batching,
resampling, axes, zero observations, budgets and failures. GPU/native acceptance
is explicit and separate from normal Python tests.

Fixture preparation and formal reconstruction each launch MATLAB once and have
separate wall-time fields. The latter includes preparation of the supplied
frequency input plus two model updates in that same process.
Pass --fixture /absolute/existing/fixture-directory to reuse coherent.json and
initial.json without generating observations again. Both this script and the
BenchLab integration command accept it and report zero fixture invocations.
