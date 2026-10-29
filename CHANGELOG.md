# P1-P6 refactor

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
[deployment](Runtime/README.md) and [execution record](docs/refactor_progress.md).
