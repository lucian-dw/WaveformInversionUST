# Provenance and modification map

## Upstream

Actual GitHub fork of [rehmanali1994/WaveformInversionUST](https://github.com/rehmanali1994/WaveformInversionUST).
Upstream base: `9fb31657c4ba9141a2bdf5c2eeac644320d090e8`.
Original Functions, Simulations, demo scripts, figures and README are retained
byte-for-byte in reference/upstream-9fb31657; MIT LICENSE.txt stays at root.
They are historical references and must not enter the maintained MATLAB path.

Paper: Rehman Ali et al., *2-D Slicewise Waveform Inversion of Sound Speed and
Acoustic Attenuation for Ring Array Ultrasound Tomography Based on a Block LU
Solver*, IEEE TMI, DOI [10.1109/TMI.2024.3383816](https://doi.org/10.1109/TMI.2024.3383816).

Benchlab requirements reference (not modified):
`lucian-dw/usct-benchlab`, branch `review/numerics-contracts-20260911`,
commit `d6e24bd0d4d3e4ce65aa9231ab2dbf02e16575ad`.

## Local development provenance

The source donor is the user's USCT research workspace. Selected sources, not
its data/credentials/checkpoints, were extracted. `source-manifest.json` hashes
the maintained files; `source-map.json` records moves from the runtime starting
commit. The original manifest is byte-preserved in `historical-validation/`.
Relative historical source paths below are provenance,
not runtime dependencies.

| Runtime component | Donor / relation | Explicit changes in this fork |
|---|---|---|
| solver/HelmholtzSolver.m | project/matlab/current/Functions, derived upstream | explicit CPU/GPU switch; optional LDR9; exact mass stencil adjoint; sparse CPU double RHS |
| CUDA MEX / block_lu_mex_utils.h | current/Functions | retained checked dimensions/CUDA errors and device-resident solves; rebuilt locally |
| assembleBlockTridiagonalsGPU.m | current/Functions | direct GPU block assembly, upstream operator topology retained |
| stencilOptParams.m | current/Functions | fixed-b default, optional free-b retained; runtime freezes velocity bounds |
| kwaveLdr9WavenumberSquared.m | project/matlab/experiments | retained explicit dt/reference-speed local dispersion model |
| downsampleKWaveChannelData.m | project/matlab/experiments | archived FIR policy retained; production wrapper disallows stride |
| native/run_native_source_batch.py | project/python/kwave_cuda_acceleration | rejects unsupported physics, non-2D, bad indices and overwriting source |
| wfi_simulate / prepare | new bounded extraction | no cropping/label manipulation; actual positions; RF retained; TX-wise filtering |
| wfi_oracle / reconstruct | extracted mathematical steps of runProfiledFWI | no research-mode dispatch; exact derivative audit; explicit schedule/mask/cost |
| Python launcher / tests | new | no DPS/benchlab dependencies; explicit errors and MAT schema |

## k-Wave CUDA licensing

Vendored source originates from the locally used `native-batch-full` tree,
based on the k-Wave CUDA source release identified in its original Makefile as
`468dc31c2842a7df5f2a07c3a13c16c9b0b2b770`. This embedded release identifier is
provenance, not an independently verified Git ancestor of the custom tree.
The source header identifies kspaceFirstOrder 3.6 / 2020; see the original
Readme.md, License.md, COPYING and COPYING.LESSER.

The CUDA tree is **LGPL-3.0-or-later**, not MIT. Original copyright notices are
retained, full preferred source and build instructions are shipped, not binaries.
The rest of this fork follows the upstream MIT license; the separate k-Wave
program does not inherit that license. External MATLAB/k-Wave toolboxes are not
redistributed. For the toolbox's execution interface see
[official kspaceFirstOrder2DG documentation](https://www.k-wave.org/documentation/kspaceFirstOrder2DG.php).

Native modifications are concentrated in MatrixContainer field allocation,
CudaParameters/constant-memory field dimensions, KSpaceFirstOrderSolver FFT setup,
CufftComplexMatrix planMany, and SolverCudaKernels physical-index/batch addressing.
Each TX has a separate plane; the batch axis is not transformed as a spatial axis.
The imported Makefile targets sm_80 by default; override CUDA_ARCH for other GPUs.
The A100 extraction audit additionally fixed device-link architecture propagation
and added repeatable launcher library directories, because the MATLAB k-Wave
bridge deliberately clears LD_LIBRARY_PATH before starting a binary.

No RF, SoS arrays, model weights, private configuration, compiled libraries or
server paths were newly published. Upstream example figures remain upstream's.
