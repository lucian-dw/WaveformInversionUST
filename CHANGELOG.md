# Refactor changelog

## Post-review C04

RF preparation records phase policy, geometry reference speed when used,
sampling interval and mask shape/count, not the entire RF configuration.
The full mask remains an HDF5 observation array. The A100 MATLAB ingestion/RF
equivalence test passed with the compact metadata assertions (exit 0).

- Protocol **2** uses JSON field descriptions and HDF5 arrays. Field definitions
  are maintained once and exported to seven schemas. Arrays precede successful
  manifest publication; source/build hashes and dirty-tree admission are removed.
- One reconstruction process prepares physical observations once and reuses them
  through the frequency schedule. Null schedule means sorted frequencies once;
  repeated Hz values repeat updates. Update caps truncate the resolved full plan.
- Oracle, source-scale elimination, Helmholtz assembly and explicit instance-level
  CPU/GPU backends have separate responsibilities. No backend is auto-selected.
- Native dimensions, HDF5 resources and MEX array boundaries are explicit. A
  corrected TX-local pressure-source scale preserves independent heterogeneous
  transmitters in batch simulation.
- Historical kernels and research tools moved to the sibling archive. Current
  tests use a shared small fixture; Python transport tests are CPU-only, while
  MATLAB, CUDA, native simulation and BenchLab integration are manual gates.

Update BenchLab and WUST together. Old runtime manifests and scalar frequency
index schedules are not retained as compatibility APIs. Reconstruction still
uses slowness NCG, source-projected pressure loss and the exact mass derivative;
budget/schedule completion does not assert numerical convergence.

See [Runtime API](docs/RUNTIME_API.md), [numerics](docs/CONTRACTS.md),
[deployment](Runtime/README.md) and [execution record](docs/validation.md).

## Final numerical and lifecycle consolidation

- ROI filtering is MFMg. Clipped PR/FR uses preconditioned g*z inner products,
  restarts on frequency changes/non-descent, and projects in full slowness.
- Frozen-source curvature is a named mathematical step. One oracle starts each
  update; optional final residual evaluation records its own work.
- CPU reuses decomposition(H,'lu') for forward/adjoint source batches. CPU/GPU
  share nine-point coefficients. LDR9 returns dq/ds directly.
- Removed the unused CUDA factors branch and four unused kernels, retained the
  execution recurrences, and widened global address products to size_t.
- Native Makefile default/recipes and the eight-plan FFT shutdown chain are
  repaired. Third-party changes remain limited to those functional dependencies.
- Maintained MATLAB code is multiline. Array fields remain defined once in
  contracts.FIELDS; Python codecs publish arrays before the manifest.

## Round 3: prepared observations and bounded state lifetime

- File reconstruction decodes then prepares observations once in `wust_run`.
  Canonical artifacts consume Y directly; in-memory reconstruction consumes the
  prepared object. Changing Y/mask requires explicit re-preparation.
- TX and fitted RX exclude only outer Dirichlet degrees of freedom. The PML
  update-mask restriction is independent; optional final evaluation validates
  all canonical frequencies, including a zero-update run.
- `wust_oracle` computes only requested outputs. Final loss does not run an
  adjoint; per-update solver/virtual-source state is released after curvature
  and work records. Solver backend and stencil bounds are explicit arguments.
- General codecs expose `write_artifact`; `describe` always returns JSON.
  Unavailable metrics originate as JSON null. Current fixtures use canonical
  grid/positions, not coordinate/index conversion shims.
- Removed unused MEX resource accessors, duplicate storage checks, diagnostic
  printf, and the unconsumed native batch-size dataset. Native source layout,
  RAII ownership and residual guards remain validated.

The current [validation record](docs/validation.md) separates Python, MATLAB CPU,
MEX/GPU, native simulation and the shared BenchLab public fixture results.
