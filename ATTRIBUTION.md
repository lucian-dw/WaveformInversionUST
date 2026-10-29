# Attribution and licenses

The maintained runtime derives from [Rehman Ali's WaveformInversionUST](https://github.com/rehmanali1994/WaveformInversionUST), upstream revision `9fb31657c4ba9141a2bdf5c2eeac644320d090e8`.
The root [MIT license](LICENSE.txt) and original author notices are retained.

Please cite Rehman Ali et al., *2-D Slicewise Waveform Inversion of Sound Speed and
Acoustic Attenuation for Ring Array Ultrasound Tomography Based on a Block LU
Solver*, IEEE TMI, [10.1109/TMI.2024.3383816](https://doi.org/10.1109/TMI.2024.3383816).
Active reconstruction in this fork is sound-speed-only.

## Maintained changes

- Explicit CPU/GPU Helmholtz backends, optional LDR9 dispersion and discrete mass-stencil adjoints.
- Dimension-checked MEX calls with owned CUDA and MATLAB resources.
- Explicit frequency-pressure ingestion, source-scale elimination and Hz update schedules.
- Separate RF preparation and simulation tools, FIR anti-aliasing and actual-time DTFT.
- Native k-Wave source batching with separate physical and batch dimensions, checked HDF5 storage and resource ownership.
- Independent JSON/HDF5 runtime API; no dependency on BenchLab or diffusion packages.

Historical upstream examples and migration records are listed in
[the research index](docs/research_index.md). They are not another runtime backend.

## Third-party k-Wave CUDA source

`Runtime/third_party/kspaceFirstOrder-CUDA` is **LGPL-3.0-or-later**, not MIT.
It derives from kspaceFirstOrder 3.6 (2020); the original imported Makefile
identified release `468dc31c2842a7df5f2a07c3a13c16c9b0b2b770`.
This is source attribution, not a runtime identity requirement.

The full [license and citations](Runtime/third_party/kspaceFirstOrder-CUDA/License.md),
[GPL text](Runtime/third_party/kspaceFirstOrder-CUDA/COPYING),
[LGPL text](Runtime/third_party/kspaceFirstOrder-CUDA/COPYING.LESSER)
and upstream notices remain with the source. MATLAB and the k-Wave MATLAB toolbox
are external dependencies and are not redistributed.
