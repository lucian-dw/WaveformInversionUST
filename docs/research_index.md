# Research archive

Historical materials were exported to sibling
`research_archive/WaveformInversionUST/`, preserving their relative paths.
The source baseline is `79e347015be64cca88bacf591b4eed0952398800`; the pre-archive
maintained tree ends at `60b82fb`. Git history retains the original sources.

| Location inside archive | Contents |
|---|---|
| `reference/upstream-9fb31657/` | Original upstream Functions, Simulations, examples, sample download instructions and figures. |
| `provenance/historical-validation/` | Original environment and validation reports, including their historical attachments. |
| `provenance/source-map.json` | Historical extraction/migration record, not a runtime requirement. |
| `docs/MAINLINE_MIGRATION.md`, `docs/STAGE4A*_VALIDATION.md` | Stage-specific development records. |
| `Runtime/tests/benchmark_native_production.m` | Large 128-TX performance study; separate from maintenance tests. |
| `Runtime/tests/capture_regression.m`, `compare_regression.py` | Historical cross-revision comparisons and their original result conventions. |

Use the original checkout and recorded MATLAB/CUDA/A100 environment to replay a
historical experiment. Do not add archived Functions to the current MATLAB path.
Original run/data directories remain untouched; the archived reports retain their
locations. The external archive is not part of a normal Git clone.

See [attribution](../ATTRIBUTION.md) for licenses and upstream source, and
[the runtime API](RUNTIME_API.md) for current operation contracts.
Browse the [local archive](../../research_archive/WaveformInversionUST/).

The former experimental block-LU macro variants and free-b stencil fitting are
preserved at the same relative paths under Runtime/solver in that archive.
The maintained solver uses pivoted inverse pages and fixed b=5/6.
