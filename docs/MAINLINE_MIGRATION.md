# Stage 4A1 mainline migration

Upstream/main base: `9fb31657c4ba9141a2bdf5c2eeac644320d090e8`.
Runtime starting point: `4273f29472fafd149d9ad23acf5748f639b06971`.
The starting runtime is two commits ahead and zero behind the base. This branch
retains both commits through normal ancestry and targets `main` by PR, without
reset, force push or an automatic merge.

## Boundaries

Original upstream files move under `reference/upstream-9fb31657/` without content
changes; root LICENSE.txt is retained. Runtime entry points, the Python launcher,
the GPU-selection helper and its environment key use WUST naming. Production
code never adds reference paths. The numerical solver and native CUDA code are
unchanged except for the GPU-selection helper's mechanical rename.

`Runtime/VERSION` is shared by MATLAB and Python. Its value is retained in this
cleanup; no new runtime release is claimed. Serialized `wfi.*.v1` schemas stay
unchanged until Stage 4A2, independently of active function names.

## Archive strategy

Existing tag `runtime-v0.1.0` preserves the runtime starting point. The planned
additional archive tag is `archive/runtime-0.1.0`, pointing to that same full
SHA. Do not move existing tags. Do not delete `compat/kwave-fwi-runtime-v1` until
the maintained runtime is merged, the new protocol is verified and cross-repository
validation is complete. New release candidates and releases belong to subsequent
delivery gates, not this source cleanup.

Historical validation files retain their original bytes and commands. Their
old paths/version strings describe that historical run, not the current API.
`provenance/source-map.json` maps every starting tracked file to its new path,
and checks exact preservation of the upstream reference and third-party source.
Current numerical evidence and reproducible commands are in
[Stage 4A1 validation](STAGE4A1_VALIDATION.md).

## Deferred

Stage 4A2 owns capability discovery, direct frequency ingestion, explicit new
schemas, deadline enforcement and schedule-budget semantics. BenchLab integration,
runtime retirement in BenchLab and GPU release certification are not part of
Stage 4A1. No consumer repository is modified here.
