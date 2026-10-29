# P1-P6 refactor execution record

Branch: `codex/full-refactor-p1-p6`, based on main
`79e347015be64cca88bacf591b4eed0952398800`. Original local/A100 checkouts are
preserved. Root `AGENTS.md` contains the approved common development rules.

## P1: data, layout and resource boundaries

- Native TX expansion validates physics, indices and layout before creating the
  output. The Python API requires explicit `source_indices` or `colocated=True`.
  The CLI has explicit flags and parses input/output using argparse. MATLAB's
  existing explicit-TX environment transport is decoded only at that boundary.
- Zero ringing cutoff returns the input unchanged. Positive values keep the
  existing dimensionless fraction of `2*f/c0`; that resulting spatial cutoff is
  in cycles/m. The plan's wording must not silently change the existing positive
  cutoff from a dimensionless multiplier into a dimensional parameter.
- Block-LU application checks rank/complex-single storage before pointers and
  moves synchronization barriers outside conditional thread participation.
- Native single-domain dynamic allocations retain physical z dimensions; TX-batch
  layout remains separate. Source-gradient correction uses the per-domain
  spectrum index rather than the flattened TX-batch index.
- Parameters uses a function-local static instance rather than deleting itself.
- Matrix error formatting now matches its actual arguments.
- Matrix/stream containers use unique owners; host/device/mapped buffers and FFT
  plans release on exceptions. A prepared SimulationLayout is shared by matrix
  allocation, ND FFT and CUDA device constants.
- MEX helper errors unwind to the entrypoint; GPU arrays, cuBLAS/cuSOLVER handles
  and events are released before reporting WUST errors. Complex operations are
  shared by both Block-LU translation units.
- HDF5 temporary creation handles have local ownership; shape queries allocate
  according to rank; strings use their actual length; destruction is nonthrowing
  while explicit close reports errors. Complex disk shapes are consistent across
  dataset creation, chunks and slab writes.

Validation so far:

- Python unittest: 19 passed, 1 failed on the old source-manifest gate because
  maintained sources changed. This obsolete identity gate is scheduled for P2;
  no manifest regeneration was used to conceal that transition.
- A100: CUDA MEX build succeeded. CPU projected-gradient FD error 1.64326e-10,
  CPU LDR9 error 2.31723e-10; GPU errors 0.001112 and 0.000430214 respectively.
- CPU/GPU four-update FWI fixtures reduced loss from approximately 3.883e-17 to
  8.960e-18. Resampling and zero/truncated-budget checks passed their assertions.
- Initial MATLAB processes returned 255 at exit with a `Settings` lookup error.
  Initializing `s=settings` before the checks resolved this host initialization
  issue; the repeated CPU/GPU/resampling/budget command exited 0. No server startup
  files or other jobs were changed.
- Native CUDA executable compiled on A100 with CUDA 11.4 and local HDF5.
  CUDA 11.6 compilation also succeeded, but execution rejected the older driver;
  actual native acceptance used 11.4.
- Manual C++ HDF5 checks passed on A100: rank 3/4/5, 1024-character attribute,
  slab round trip, explicit close twice. Runtime shared libraries were supplied
  through `LD_LIBRARY_PATH` from the existing local HDF5 installation.
- Three-TX native vs serial simulation passed after the ownership changes:
  RF relative L2=0 and DTFT relative L2=0 on the 24x24, 100-step fixture.
- Genuine 3D native simulation (20x22x18 expanded grid, 100 steps) exited 0;
  pressure was finite and nonzero (L2 norm 0.07117558).
- Expanded HDF5 checks passed: scalar-write handle counts, rejected rank-5 slab,
  and actual ComplexMatrix 3D write/read with doubled float-x disk storage.

Next: jointly migrate BenchLab/WUST protocol v2 in P2, including identity-chain
removal. No P2-P6 completion is claimed. Historical references and observations
are untouched. GPU acceptance covers representative fixtures, not clinical data
or a new imaging-quality campaign.
