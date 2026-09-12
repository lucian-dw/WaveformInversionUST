# Runtime contracts v1

## Ownership and axes

The caller owns anatomy selection, ROI, crop/resize, labels, normalization,
splits and metrics. Supply the final physical computational grid. There is no
300→256, intermediate 480, automatic water padding or image transpose.

- c_mps: [Ny,Nx], metres/second. x_m indexes columns; y_m indexes rows.
- Physical positions: [N,2], x,y in metres.
- tx_index/rx_index: MATLAB 1-based column-major linear indices.
- RF pressure: [T,RX,TX]. time_s includes the source pulse offset.
- Complex Y and boolean mask: [TX,RX,F].
- frequencies_hz: increasing positive Hz, strictly below saved Nyquist.
- DTFT: sum RF(t)*exp(-2*pi*i*f*t)*dt. Helmholtz sign is -1.
- Schedule: 1-based frequency indices, repeated for each desired update.
- Snapped TX/RX coordinates are recorded. Merged elements fail.
- k-Wave's first spatial axis explicitly maps to physical y.
- No nominal circular array is substituted for the actual grid positions.

## Preparation

Factor 2 uses the archived Kaiser FIR/polyphase: half-length 24, beta 8.6,
delay compensated. Factor 1 keeps samples. Stride exists only in the archived
helper for audit tests; it is not exposed by simulation. Nyquist alone does not
guarantee spatial/time accuracy: choose grid, CFL and duration for the bandwidth.

window='none' or 'legacy-nominal' (Gaussian early-arrival taper, width 5% of max
homogeneous travel time, infinite late tail).
phase_correction='none' or 'homogeneous-tof':
exp(-i*2*pi*f*(distance_snapped-distance_actual)/c_geom).
This is an approximate homogeneous correction, not exact heterogeneous physics.

Caller supplies the mask. Historical near-neighbour rejection / percentile
outlier removal must be explicit caller policies; zeros are not silently
treated as missing observations. prepare_config is stored in the MAT input.

## FWI

Loss: 1/2 sum_TX ||mask*(a_TX * P H(c)^-1 q_TX - Y_TX)||².
Complex source scale a=(p* d)/(p* p) is analytically eliminated on included RX.
It removes scalar source amplitude/phase, not angle-dependent multipath.
The envelope theorem avoids differentiating a separately.

Real slowness s=1/c is optimized, with zero attenuation, nine-point stencil/PML.
Frozen stencil_bounds are required: otherwise stencil-coefficient optimization
would change while differentiating the medium.

continuum uses (2*pi*f/c)^2. kwave-ldr9 requires explicit dispersion.time_step_s,
reference_speed_mps and model_reference_speed_mps. dt is the ORIGINAL simulator
dt, not decimated RF dt. Never infer these synthetic metadata from real RF.

This runtime uses the exact derivative of the assembled optimized mass stencil:
H=K+M_mass diag(PML*q(c)).
The perturbation source is M_mass(v*delta_s); backprojection applies M_mass*
to the adjoint. Historical pointwise-only virtual sources omit off-diagonals.
Therefore this release is NOT claimed bitwise identical to every historical
experiment. Original upstream scripts remain unchanged for comparison.

NCG: clipped PR/FR beta, per-frequency restart, one linearized quadratic step,
explicit damping, velocity step cap, bounds and update mask.
It is not monotone line-search FWI, O0/PSPA, or attenuation inversion.
Linearized curvature freezes source scale, as in the reference method.
There are no hidden trial forwards.

Records include assembly+factor time, forward/adjoint/linearized time, RHS counts,
one fresh factorization per update. Residuals are BEFORE update. Selection is
final, never best-GT. CPU double vs GPU complex-single is explicit; GPU failures
do not silently fall back. Caller controls GPU visibility/selection.

## Output integrity

Batch results are saved via temporary MAT then renamed after success.
Use unique output paths for parallel calls; no distributed lock is supplied.
Requests are trusted local configuration, not a sandbox for untrusted inputs.
Outputs record config, timings, MATLAB and runtime versions.
Caller should retain Git pin and hashes of input artifacts.
