# Stage 4A2 validation

## Scope and identity

Base: `8d28f5baeb66e7f510a87a7704e1313f01000061`, merged Stage 4A1 PR #1.
Branch: `feat/wust-runtime-contract`. Runtime: `0.2.0-dev.1`; schema version: `1`.
Upstream base: `9fb31657c4ba9141a2bdf5c2eeac644320d090e8`.

This changes WUST's serialized protocol, ingestion, admission/failure and
process/artifact contracts. No BenchLab/consumer code or valid-input FWI
optimization formula is changed. Version, upstream metadata and source manifest
remain the identity authorities. Results record detectable Git facts and
explicit dirty-tree overrides. Pre-commit test artifacts truthfully identify the
dirty checkout, not the future PR commit. Hashes identify sources; they are not
approval signatures.

## Python and static checks

```sh
python -m unittest discover -s Runtime/tests -p 'test_*.py' -v
python -m ruff check Runtime/python Runtime/native Runtime/tests provenance
python -m black --check Runtime/python Runtime/native Runtime/tests provenance
python -m compileall -q Runtime/python Runtime/native Runtime/tests provenance
python Runtime/python/export_schemas.py --check
python provenance/check_manifest.py
git diff --check
```

**19 tests pass**, including retained native-runner tests. Coverage includes
schema/legacy/unknown rejection, axes/units/hashes, masked nonfinite data and
valid zeros, atomic no-overwrite publication, process errors, deadlines,
descendant cleanup, launcher termination, discovery and archive/build identity.
Ruff, Black, compileall, schema consistency, manifest and diff checks pass.
These checks require neither MATLAB nor GPU. No GitHub Actions workflow is
introduced; local checks are not remote CI.

## MATLAB CPU

MATLAB R2024b Update 4, macOS ARM64, CPU float64:

```sh
matlab -batch "addpath('Runtime/tests'); test_runtime('cpu'); test_resampling; test_path_isolation; test_ingest_frequency; test_batch_contract;"
python Runtime/tests/integration_runtime.py --matlab matlab --out /absolute/new-artifacts
python Runtime/python/wust_runtime.py describe --json --probe --matlab matlab --timeout-s 90
```

Five MATLAB test functions pass; batch-contract includes seven explicit
invalid/numerical failure cases. Source-projected gradient relative error:
`3.75286e-10`; LDR9 derivative error: `8.41586e-11`. Tiny loss:
`3.88345e-17 -> 8.95907e-18`. FIR pass ratio: `1.00006`; alias rejection:
`-75.17 dB`.

The actual JSON/HDF5 launcher passes six success invocations: asymmetric sentinel
ingestion, zero updates, coherent physical-data ingestion, two reconstruction
updates, RF preparation, and equivalent direct-frequency ingestion. One further
invocation deliberately fails with insufficient fitting receivers and leaves no
success manifest. The coherent fixture is rectangular 23 by 27, unequal spacing,
two TX/four RX. Pre-update residuals are approximately `0.0050622` and
`0.00423262`; neither is reported as a final-model residual.

Independent ingestion assertions check nonzero origin, exact MATLAB indices,
complex sentinels, frequency/mask permutation, conjugation exactly once,
duplicate snapping rejection and RF/direct-DTFT equivalence. Wholly masked RF
traces containing NaN are excluded; nonfinite used traces fail.

The real probe reports MATLAB/CPU available and GPU unavailable on the Mac
(Parallel Computing Toolbox function unavailable). Static discovery never starts
MATLAB. An unrelated user-startup missing-directory warning remains on stderr.
This gate validates transport and numerics, not clinical/image performance.

## Before/after preservation

Use separate checkout/processes for base and new source:

```sh
# At the base checkout:
matlab -batch "addpath('Runtime/tests'); capture_regression(pwd,'/absolute/before.mat');"
# At the new checkout:
matlab -batch "addpath('Runtime/tests'); capture_regression(pwd,'/absolute/after.mat');"
python Runtime/tests/compare_regression.py /absolute/before.mat /absolute/after.mat --protocol-migration
```

**73 numeric fields pass.** Objective, initial slowness gradient, authoritative
final sound speed, single-precision history, update alpha/beta, work counts and
all non-timing per-update numerical records have **zero difference**. Prepared
pressure, mask, frequencies and sampling indices also have zero difference.

Reconstructed grid/snapped coordinates differ by at most
`2.0816681711721685e-17 m`. Only these coordinate fields use `atol=1e-15 m`;
numerical comparisons remain `rtol=1e-12, atol=0`. Timing fields are excluded;
schema/new metadata are checked as intentional protocol differences. MAT files
are internal regression captures, not production transport, and are not tracked.

An initial manual capture omitted its required root argument; the corrected
two-argument invocation passed. An exploratory integration attempt used random
transport sentinels as a nonlinear target and failed candidate validation. The
accepted reconstruction fixture uses coherent Helmholtz observations; failure
checks and optimization formulas were not relaxed to accommodate random data.

## Tiny GPU gate

An isolated A100 80GB directory, MATLAB R2021b, one visible GPU/four CPU threads:

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 matlab -batch "addpath('Runtime/tests'); test_runtime('cpu'); test_runtime('gpu'); test_ingest_frequency; test_batch_contract;"
```

CPU gradient/LDR9 errors: `1.64326e-10`, `2.31723e-10`.
GPU gradient/LDR9 errors: `0.001112`, `0.000430214`, within the original `0.03`
single-precision tolerance. GPU tiny loss: `3.88327e-17 -> 8.95982e-18`.
Ingestion and completion/failure tests also pass. Existing MEX binaries matching
unchanged maintained solver sources were reused, not rebuilt in this stage.

The process exits zero, but its existing MATLAB exit hook emits an undefined
`Settings` warning. This is disclosed rather than called a clean full release
certification. Full native simulation/A100 release capability remains Stage 4C.

## Remaining boundaries

Stage 4B must add BenchLab's single `fwi_wust`, call WUST ingestion, resolve
initial maps/masks and schedule budgets, and map completion/failure semantics.
Do not require Born calibration or regenerate observations from GT. No consumer
migration, old BenchLab API retirement, production-size imaging run, online
numerical stopping, or clinical validation is included here.
