# Validation status — 2026-09-12

Release scope: standalone runtime 0.1.0 **release candidate**, not a claim of
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

## A100 status

User authorized A100 validation. At the time of this record, both direct SSH
and the configured server connector timed out before connection. **No A100 job
was submitted and no new native128 speedup is reported.**

Prepared gate: `native/validate_native.py`, with a single-TX full-array HDF5 input,
explicit device and fresh output directory. It compares the same binary with
native disabled (128 serial runs) versus enabled, then compares raw RF and 29
DTFT values. Threshold: relative L2 <1e-4 for both. This first gate uses native
output files; the MATLAB wrapper's outside-PML/source-index and RF-reordering
path must also be checked end-to-end against the serial MATLAB path on A100.

Before calling this a production-certified pin, complete:
1. Build included Linux native source on the target CUDA/HDF5 installation.
2. Tiny asymmetric 128-array serial/native gate (include an unsorted TX order).
3. Wrapper-level same-input RF/DTFT comparison, ds2 FIR, positions and source offset.
4. Representative production grid run; record peak RAM/VRAM with sampling interval,
   generation/IO/filter/DTFT timings separately, and numerical comparison.

No GPU memory benchmark was performed locally. The native source donor had prior
A100 production use, but that does not certify this newly extracted wrapper.
