# WaveformInversionUST

Independent MATLAB runtime for **2-D sound-speed full-waveform inversion**.
Frequency continuation, per-source complex-scale elimination and nonlinear
conjugate gradients use an optimized nine-point Helmholtz discretization.
CPU uses sparse double-precision solves; GPU uses complex-single CUDA Block LU.
This is research software, not a clinically validated reconstruction system.

## Install

Python 3.10+ with NumPy/h5py provides the external launcher. MATLAB is required
for reconstruction. GPU execution additionally needs Parallel Computing Toolbox
and CUDA MEX; FIR resampling needs Signal Processing Toolbox. k-Wave is only a
simulation dependency. See [deployment](Runtime/README.md).

~~~bash
python -m pip install numpy h5py
python Runtime/python/wust_runtime.py describe --json
~~~

Discovery is static and does not start MATLAB. An explicit host check is:

~~~bash
python Runtime/python/wust_runtime.py describe --probe --backend cpu --matlab /path/to/matlab
~~~

## One reconstruction

Prepare a protocol-v2 frequency input and initial-model artifact, then invoke:

~~~bash
python Runtime/python/wust_runtime.py run /absolute/request.json \
  --matlab /path/to/matlab --timeout-s 300
~~~

The [API guide](docs/RUNTIME_API.md) contains a complete request. One MATLAB process
normalizes the supplied observations and runs the entire resolved frequency plan.
WUST does not import BenchLab and never generates production observations from GT.
RF preparation and simulation are separate data tools.

## Read results

The output JSON names its HDF5 arrays. Use Runtime/python/wust_io.read_artifact
for explicit axes/units and real/imaginary datasets. Final c_mps is [y,x] in m/s.
Metadata records backend, executed schedule, pre-update losses and completion
reason. Finishing a schedule is not convergence. Attenuation is fixed to zero.

Add only Runtime/matlab to a production MATLAB path, not genpath of the repository.
[Contracts](docs/CONTRACTS.md) define the math;
[Chinese pipeline](docs/PIPELINE_ZH.md) describes the flow;
[research archive](docs/research_index.md) indexes historical material.

## Attribution and license

Derived from [Rehman Ali's WaveformInversionUST](https://github.com/rehmanali1994/WaveformInversionUST).
Please cite Ali et al., *2-D Slicewise Waveform Inversion of Sound Speed and
Acoustic Attenuation for Ring Array Ultrasound Tomography Based on a Block LU
Solver*, [IEEE TMI, 2024](https://doi.org/10.1109/TMI.2024.3383816).
See [ATTRIBUTION.md](ATTRIBUTION.md) for source and modifications.

[LICENSE.txt](LICENSE.txt) is MIT. The separately licensed
[k-Wave CUDA source](Runtime/third_party/kspaceFirstOrder-CUDA/License.md) is
LGPL-3.0-or-later and is not relicensed as MIT. MATLAB and external k-Wave are not included.
