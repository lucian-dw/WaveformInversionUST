# Validation status — 2026-09-12

Release scope: standalone runtime 0.1.0, not a claim of
complete cross-platform certification or identical historical trajectories.

## Executed locally

MATLAB R2023a Update 5, Windows, local NVIDIA GPU, existing CUDA compiler.

| Test | Result |
|---|---|
| Python independent-TX encoding, order, no overwrite, unsupported physics | 3 unittest tests passed |
| CPU source-projected discrete adjoint | relative finite-difference error 9.50e-11 |
| CPU LDR9 derivative | 2.41e-10 |
| GPU complex-single discrete adjoint | 1.31e-3 |
| GPU complex-single LDR9 derivative | 8.35e-4 |
| FWI source batch 2 vs 3 | loss and gradient relative difference <1e-4 |
| CPU tiny 4-update FWI | loss 3.88345e-17 → 8.95907e-18 |
| GPU tiny 4-update FWI | loss 3.88329e-17 → 8.96001e-18 |
| FIR passband tone | amplitude ratio 1.00006 |
| FIR specified stopband tone | alias suppression 75.17 dB |
| k-Wave serial → RF FIR → 2-frequency DTFT → FWI | passed, asymmetric source geometry, positions verified |
| Python → MATLAB batch → MAT result | passed; process wall 9.64 s on tiny case |
| CUDA Block-LU MEX rebuild | both factor and apply binaries compiled locally |

Finite-difference step is smaller in CPU double and larger in GPU single to
resolve differences above solver roundoff. Raw losses depend on source scale;
these numbers certify smoke functionality, not anatomical reconstruction quality.

## A100 executed gates

After connectivity recovered, ran on idle A100-SXM4-80GB GPUs 0/1/2. Freshly
built included native CUDA source with CUDA 11.4 and existing user-space HDF5;
MATLAB R2021b. No system installations or unrelated processes were changed.
Raw numerical RF/HDF5 outputs are retained on the test server, not in Git.

| Gate | Result |
|---|---|
| 128 serial CUDA TX vs native128, same binary/native off vs on | RF relative L2 = 0; 29-frequency DTFT relative L2 = 0 |
| Same tiny comparison, process wall including startup/IO | 121.8487 s serial vs 2.8149 s native; 43.29x |
| 128 native TX vs 8 independent MATLAB TX (indices 1/2/17/33/65/81/97/128) | RF relative L2 3.7962e-6; DTFT 3.6213e-6 |
| Wrapper position/order/time, including unsorted array and nonzero source time offset | passed, FIR ds2 |
| A100 CPU exact projected gradient / LDR9 | 1.6433e-10 / 2.3172e-10 |
| A100 GPU exact projected gradient / LDR9 | 1.1120e-3 / 4.3021e-4 |
| CUDA MEX rebuild, tiny FWI improvement, FIR | passed |

The **43.29x is a tiny-grid result with process-launch/IO overhead included**,
not a kernel-only speedup, and is not extrapolated to the production grid.
Both serial/native test scripts and the independent MATLAB check are included.

### Representative production computational size

Artificial asymmetric numerical medium, 1004x1004 physical grid plus outside
PML → 1024x1024, 128 TX, 4751 time steps, dt=4.6312675e-8 s, zero attenuation.

| Stage / resource | Measurement |
|---|---|
| Generation, including native process and HDF5 bridge | 91.2950 s |
| FIR ds2 | 1.3395 s |
| 29-frequency preparation | 0.3917 s |
| RF MAT save (separate) | 5.7530 s |
| RF shape | 2376x128x128 |
| Frequency shape | 128x128x29 |
| Peak assigned-GPU memory, 0.5 s samples | 7634 MiB (7.46 GiB) |
| Peak process-tree RSS, 0.5 s samples | 7,817,056,256 bytes (7.28 GiB) |

All values finite. This tests representative **computational size**, not an
anatomical accuracy dataset or a full-grid serial/native speedup comparison.
RSS sums processes and can double-count shared mappings; 0.5 s sampling may miss
brief peaks. No claim that all 80 GB must be occupied to achieve good throughput.
The tiny wrapper-only test peaked at 600 MiB GPU / 1,465,229,312-byte process RSS.

Deployment fixes caught by A100 testing: explicit loader library directories
after k-Wave clears LD_LIBRARY_PATH, and CUDA_ARCH propagated to device link.
The server's custom MATLAB shutdown script emitted a Settings warning after the
FWI assertions completed; later native test jobs used exit(0,'force') to bypass
that site-specific finish script. No global MATLAB configuration was edited.

Remaining limits: no real-RF calibration certification, no 3D/absorption support
in native128, no anatomical full-FWI benchmark, and no historical bitwise-equality
claim for the newly corrected adjoint. Machine-readable results: a100_validation.json.
