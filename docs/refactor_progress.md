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

## P2: protocol v2 and single-process reconstruction

- Unified frequency/canonical preparation in wust_prepare_observations; removed
  the old ingestion/measurement/reconstruction validation wrappers. Oracle
  consumes prepared observations without rescanning them each update.
- Physical Hz schedule is mapped and capped once in MATLAB; optional history
  defaults off. Reconstruction numerical update order is unchanged.
- One field table exports seven schemas and drives MATLAB/Python array layout.
  Removed identity manifests, snapshot/hash passes and repeated full readback.
  Runtime records version/backend/precision; errors use a separate failure file.
- Python: 15 tests passed. Removed identity-only tests, preserving numerical,
  transport, valid-zero, process termination and timeout coverage.
- A100 BenchLab/WUST CPU and GPU integration passed. One MATLAB reconstruction
  invocation per run, plus separately recorded coherent fixture generation.
- A100 numerical tests passed: CPU FD 1.64326e-10, GPU FD 0.001112;
  CPU LDR9 2.31723e-10, GPU LDR9 0.000430214. Four-update losses remained
  approximately 3.883e-17 -> 8.960e-18. Asymmetric ingestion, masked invalid
  pressure, valid zero, RF DTFT, history opt-in and zero/truncated budgets passed.
- Logs/results: /tmp/usct_refactor_v2_math.log, /tmp/usct_refactor_v2_cpu,
  /tmp/usct_refactor_v2_gpu on A100. Default CI still needs no MATLAB/GPU.

Next: P3 removes historical runtime duplicates and relocates research material
to sibling research_archive with an index. Native display-only Git labels were
removed; the native backend will be rebuilt at final acceptance.

## P3: maintained runtime and research archive

- Moved 39 listed historical/reference/provenance files into sibling
  research_archive/WaveformInversionUST. Added an index with source commits.
- Consolidated upstream paper, license and source attribution in ATTRIBUTION.md;
  kept vendor licenses intact. Removed reference-path certification tests.
- Moved the simulation/reconstruction teaching recipe into examples with an
  explicit output directory and the v2 Hz schedule. It is not normal CI.
- Local Python runtime suite: 15 passed. This unit changes no numerical kernels;
  CPU/GPU numerical evidence remains the P2 gate, not a new P3 hardware claim.

Next: P4 makes process ownership, sampling and backend selection explicit.

## P4 unit 1: explicit deployment and RF preparation

- Extracted process-group TERM/KILL/reap ownership and made launcher batch size
  an installation argument. Generated launchers are readable Python argument
  arrays, including paths with spaces.
- RF preparation accepts an explicit Grid, preserves actual sample times and
  performs one geometry preparation. Removed nominal-arrival windows and stride
  aliases; decimation rejects non-integers and uses polyphase FIR.
- Helmholtz backend is an instance property. Setup no longer mutates an
  environment variable, and the global wustUseGPU helper was removed.
- WUST Python: 16 passed. A100 MATLAB CPU/GPU FD and LDR9 checks retained their
  P2 errors; four-update loss remained 3.883e-17 -> 8.960e-18. The test now changes
  setup after constructing a GPU solver and verifies it stays on GPU.
- Nonzero RF time-origin FIR test: pass ratio 0.999878, alias -72.39 dB;
  fractional-factor rejection and asymmetric DTFT ingestion passed.
  Log: /tmp/usct_refactor_p4_runtime_verified.log on A100. Diff check passed.

Next: P4 native CLI/error ownership and BenchLab typed public configuration;
P5 will simplify the solver constructor/options without changing its backend.
