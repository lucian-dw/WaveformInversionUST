# Numerical contracts

## Layout

| Quantity | WUST convention |
| --- | --- |
| Sound speed / update mask | [y,x], m/s / boolean |
| Grid | shape_yx, spacing_yx_m, origin_yx_m at outer pixel edge |
| TX/RX positions | [element,xy], meters |
| Sampling indices | MATLAB one-based column-major |
| RF | [time,rx,tx], time_s in seconds including its actual origin |
| Canonical spectrum / mask | [tx,rx,frequency] |
| Frequencies | distinct positive Hz, sorted during preparation |
| Transform | $\Delta t\sum p(t)e^{-2\pi ift}$ |
| Schedule input | physical Hz entries, repetitions are explicit |

WUST owns snapping, frequency sorting and conjugation. Nearest-grid ties choose
the first index; same-array elements collapsing to one sample are rejected.
Invalid samples are masked before arithmetic; valid zeros remain data.
There is no implicit crop, resize, nominal circular geometry or water padding.

## Prepared data

RF decimation uses delay-compensated polyphase FIR; factor one preserves samples.
The actual saved time axis determines the DTFT. An optional homogeneous-tof
phase correction accounts approximately for snapped versus physical distances;
it is not exact heterogeneous propagation. Source onset is not inferred by
resetting acquisition time to zero.

## Source-projected objective

For each frequency, real slowness $s=1/c$ parametrizes the Helmholtz system.
For admitted receivers and each TX,

$$
a^\star=\frac{p^Hd}{p^Hp},\qquad
\Phi_\omega(s)=\frac12\sum_{\rm TX}\|a^\star p(s)-d\|_2^2 .
$$

At least two valid receivers per scheduled TX/frequency are required. Source-scale
elimination removes a scalar amplitude/phase, not arbitrary receiver distortion.
Its derivative drops out of the gradient by stationarity of the eliminated scale.

The optimized stencil is $H=K+M_{\rm mass}\operatorname{diag}(\mathrm{PML}\,q(s))$.
The virtual source uses the full mass stencil; the adjoint applies its conjugate
transpose. Stencil speed bounds remain frozen while differentiating.

Continuum uses $q=(2\pi f/c)^2$. kwave-ldr9 additionally requires explicit original
simulator time_step_s, reference_speed_mps and model_reference_speed_mps. They
cannot be inferred from decimated RF or clinical data.

## Update and schedule

The loop restarts nonlinear CG at frequency changes, uses clipped PR/FR beta,
then a linearized curvature step with frozen source scale. Damping, a speed-step
cap, bounds and an update mask constrain the candidate. Optional ringing removal
filters the gradient. This is not monotone line-search FWI and adds no explicit
quadratic image regularizer.

A null/omitted schedule means each canonical frequency once. The complete Hz plan
is resolved first, then truncated by max_updates. Every entry is one model update.
Each loss belongs to the **pre-update model at that entry's frequency**.
A final-model residual is null unless explicitly computed; this runtime does not
compute one.

Schedule completion is completion, not convergence. The runtime supports update
count and a hard elapsed deadline; it rejects numerical stopping thresholds and
forward/adjoint caps. Relative slowness update is a diagnostic over the update
mask, normalized by the previous full-slowness norm.

## Transport and ownership

Python owns JSON/HDF5 paths and MATLAB process groups. MATLAB owns physical
preparation and numerical state. Arrays close before the success manifest is
published; failure leaves partial artifacts and a separate failure record.
Use separate output directories for concurrent calls. Runtime/VERSION identifies
software; protocol version 2 identifies the transport. No source identity gate is
required. [API](RUNTIME_API.md) defines exact fields.
