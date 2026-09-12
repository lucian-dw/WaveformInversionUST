# Stage 4A1 validation (2026-09-12)

This is a source/layout/naming regression gate, not a new GPU release or an
anatomical image-quality claim. Baseline is
`4273f29472fafd149d9ad23acf5748f639b06971` in a detached, unchanged worktree.

## Tests

| Check | Result |
|---|---|
| Baseline Python unittest | 3 passed |
| Maintained Python unittest | 7 passed (3 existing + 4 mainline contracts) |
| Local MATLAB R2024b CPU `test_runtime` | passed |
| Local MATLAB R2024b `test_resampling` | passed |
| Local MATLAB R2024b `test_path_isolation` | passed |
| CPU before/after capture | 73 numeric fields, max absolute difference 0 |
| A100 MATLAB R2021b CPU `test_runtime` | passed |
| A100 MATLAB R2021b GPU `test_runtime` | passed |
| A100 resampling | passed |
| A100 serial k-Wave/RF/DTFT/FWI smoke | passed |
| Python launcher with real MATLAB and smoke request | output saved; returned runtime version matches VERSION |
| Native 128-TX pipeline versus eight serial TX | passed; detached A100 run recorded exit status 0 |

Local CPU projected-gradient relative errors: continuum `3.75286e-10`, LDR9
`8.41586e-11`. These match the baseline. Tiny training loss changes from
`3.88345e-17` to `8.95907e-18`. FIR pass ratio `1.00006`, alias rejection
`-75.17 dB`. These are numerical checks on finite synthetic fixtures.

A100 CPU errors: `1.64326e-10`, `2.31723e-10`; GPU errors: `0.001112`,
`0.000430214` (existing complex-single tolerance `0.03`). The two MEX files
were copied into the isolated test checkout from the previous runtime build;
their corresponding CUDA source hashes match this tree. Actual tiny GPU solves,
not just `exist` checks, exercised both modules. No MEX or test output is committed.

The local MATLAB startup reports a missing user `topflow` directory. A100 MATLAB
reports an `exit`/`Settings` error even for `matlab -batch "disp(version)"` without
loading WUST. These are existing environment diagnostics, not introduced by the
rename. The launcher returned exit status zero and its output was present.

The optional native pipeline was run in fresh output directories. Its saved
reports both give RF relative L2 `3.7962280291354263e-6` and frequency relative L2
`3.6212664326281162e-6` (existing threshold `1e-4`), with `passed=true`.
The first two attached SSH/MATLAB invocations nevertheless ended with status 255
after printing the report, including a retry with an explicit forced MATLAB
exit. A subsequent detached A100 execution captured the MATLAB exit code on the
server: **0**, with identical numerical errors. Its wrapper had a 600-second
timeout and no test process remained after completion. Thus the native numerical
and process gate passed; the attached-session exit cause remains unestablished.
This reuses the previously built native binary and does not certify a fresh
native rebuild or the production-size performance benchmark.

## Reproduce CPU preservation

Install `scipy` in addition to the normal test dependencies. In isolated MATLAB
processes, use `Runtime/tests/capture_regression.m` with the respective runtime
root and an output path outside Git. For the historical run, mechanically replace
`wust_` with `wfi_` in a copy of that test harness outside the baseline checkout.
Do not change the historical source.

```matlab
addpath('Runtime/tests');
capture_regression(pwd, '/absolute/artifacts/after.mat');
```

```sh
python Runtime/tests/compare_regression.py /absolute/artifacts/before.mat /absolute/artifacts/after.mat
```

Comparison checks prepared frequency observations, geometry/indexing/masks,
initial loss and gradient, final image, image history, config and every record's
non-timing fields. It also checks record keys and serialized schema strings.
Tolerance is `rtol=1e-12`, `atol=0`; observed differences are exactly zero.
Measured wall, assembly/factor, forward, adjoint and linearized times are
intentionally excluded from numerical equality, not removed from records.

## Provenance and static checks

`provenance/source-map.json` covers all 133 files at the starting commit. The
upstream reference, license, vendor sources and archived validation files are
byte-preserved. The current manifest is checked with
`python provenance/check_manifest.py`; refresh it only after reviewed edits.
Ruff and Black pass for the new Python files and the renamed launcher. Existing
compressed native tooling is left unchanged in this numerical-preservation round.
The maintained-file `git diff --check` passes. The full staged diff flags seven
pre-existing trailing-space lines in the byte-preserved upstream README, now
shown as an added reference file alongside the rewritten root README. These
historical bytes are deliberately not reformatted. No new observation arrays, binaries or checkpoints are
tracked. This repository has no GitHub Actions workflow at the starting point;
local/A100 evidence is not presented as a remote CI run.
The full `git diff --cached --find-copies-harder --check` also passes, correctly
recognizing the original README as a byte-preserved copy instead of new prose.

The 1024-grid production performance benchmark is not repeated for a naming
cleanup. Full release-scale GPU validation belongs to Stage 4C.
