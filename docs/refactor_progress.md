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

## P4 unit 2: native entry and index ownership

- Main now owns the failure exit; logger and solver layers propagate errors.
  Help/version return before CUDA selection. Integer arguments are parsed once,
  with complete-string and range checks, including OMP settings.
- Index imports validate unique encoded indices against the explicit physical
  grid/batch layout, ordered cuboid corners and one-based storage. Conversion
  rejects repeated base changes. Inclusive cuboid capacity includes both ends.
- A100 CUDA 11.4 rebuilt successfully. Help/version work with no visible GPU;
  malformed integers, zero/duplicate/out-of-range indices and reversed corners
  were rejected. A one-voxel cuboid emitted 100 finite time samples.
- The first batch check exposed a physical-only range check; using the declared
  batch layout corrected it. Repeated 3-TX serial/batch RF and DTFT comparisons
  both gave relative L2 error 0. Results: /tmp/usct_refactor_p4_native_layout;
  boundary logs: /tmp/usct_refactor_p4_boundaries. Required shared-library path
  was supplied explicitly; no server-wide binaries or libraries were replaced.

Next: finish BenchLab P4 public configuration and continue P5 numerical modules.

## P5: prepared numerical kernels and independent native comparison

- Helmholtz instances own their backend/options. Oracle preparation caches geometry,
  receiver counts and observation norms; the source-projected objective and exact
  mass-stencil derivative are unchanged. Removed alternative LU branches and
  archived the former source before simplifying the maintained CUDA path.
- Native batch expansion now rescales each TX using its local sound speed, with
  the same Dirichlet/additive formula as k-Wave's scaleSourceTerms. A heterogeneous
  asymmetric fixture exposed the previous shared-source amplitude error.
- A100: MATLAB CPU gradient FD 3.22e-10, LDR9 FD 1.82e-10; GPU gradient FD
  1.11e-3, LDR9 FD 4.30e-4. Both backend tests completed with exit 0. Native CUDA
  11.4 rebuild passed. The corrected 3-TX independent MATLAB/native comparison
  reported RF error 1.77e-6 and spectral error 8.05e-7 (threshold 1e-4), with
  report saved at /tmp/usct_refactor_p5_native_result_v3/report.json on A100.
  That SSH/MATLAB session returned 255 after publishing the passing report;
  P6 must repeat the maintained gate with a confirmed normal process exit.
- Local Python: 17 passed, 10 subtests. git diff --check passed.

Next: P6 consolidates maintained tests and current user documentation, then runs
installation and the complete cross-repository acceptance on A100.

## P6: readable runtime, one fixture and final gates

Implemented: readable slowness update loop and MATLAB tests, a shared coherent
fixture with separate preparation/reconstruction timing, explicit single-backend
probe, normal Python native imports, and current English/Chinese runtime docs.
The full-size native validator is archived; the asymmetric three-TX/four-RX
MATLAB/native test is the maintained hardware gate. Removed duplicate batch test
and export fixture. The source-scale rejection and mathematical references remain.

Validation on A100 (MATLAB R2021b, CUDA 11.4 build, A100-SXM4-80GB):

| Gate | Actual result |
| --- | --- |
| Python transport/native tests | 17 passed (1.19 s); also pass locally |
| CPU continuum / LDR9 directional derivative | 3.46637e-10 / 1.97022e-10 relative error |
| GPU continuum / LDR9 directional derivative | 1.112e-3 / 4.30214e-4 relative error |
| Four-update CPU loss | 3.88345e-17 -> 8.95907e-18 |
| Four-update GPU loss | 3.88327e-17 -> 8.95982e-18 |
| RF ingestion | Rectangular unequal-spacing geometry, reordered frequencies, conjugation, valid zero and masked missing data passed |
| FIR decimation | Passband ratio 0.999878; aliased tone -72.39 dB |
| Native versus independent serial MATLAB | RF relative L2 1.7749e-6; frequency relative L2 8.0549e-7, both below 1e-4 |
| BenchLab CPU/GPU protocol | One reconstruct process each; same 23x27 fixture, two updates of a three-entry schedule; passed |
| Teaching pipeline | 32x32 MATLAB k-Wave simulation -> FIR -> DTFT -> two-frequency CPU FWI; finite result saved, process exit 0 |

CPU/GPU mathematical commands exited 0. The native comparison initially printed
its successful report but returned 255 during batch exit; repeating the same gate
with explicit exit(0) after all assertions confirmed MATLAB exit code 0. The final
native report is /tmp/usct_refactor_p6_native_exit/report.json. No server startup
or unrelated job was modified. Eight obsolete sync residues were moved by exact
path into the isolated A100 research archive before rerunning Python discovery.

Black (10 Python files), Ruff and git diff --check pass. Regenerating all seven
schemas produces no diff. No tracked data arrays, compiled outputs or files over
2 MB. Original histories, research material and observations remain preserved.

P1-P6 is complete at the declared small numerical/interface validation scope.
Next: maintainer review of the two local branches. Larger acquisition performance
and clinical validity are not established by these gates. No remote push or merge
has been performed as part of this refactor.
