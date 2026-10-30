# Native boundary behavior

`Runtime/native/run_native_source_batch.py` expands one pressure-source input
into independent 2D TX planes. The Python `expand_input` API requires either
`source_indices=[...]` (one-based physical HDF5 indices) or `colocated=True`.
There is no implicit assumption that transmitters are the first receivers.

The launcher accepts `--source-indices ...` or `--colocated`, `-i INPUT`, `-o OUTPUT`,
`--batch-size N` and `--native-binary PATH`. The existing MATLAB launcher passes
explicit physical indices through `KWAVE_NATIVE_TX_INDICES_H5`; only the CLI
boundary decodes this transport field. The expansion function does not read it.
Invalid physics/layout/index input is rejected before creating an expanded file.
Source attributes and data are read in the same HDF5 open. Batch count is
derived by C++ from the expanded `p_source_index` length when
`KWAVE_NATIVE_SOURCE_BATCH` is enabled; the receipt records the actual batch size.

`SimulationLayout` is prepared once from the native input. Matrix allocation,
ND FFT planning and device constants share that layout. Dynamic single-domain
allocations use full physical dimensions. Batched 2D simulation stores TX planes
separately from per-domain spectral correction data.
Block-LU RHS batches are unrelated to these simulation batches.

`ringingRemovalFilt(..., cutoff=0, ...)` disables filtering. Positive `cutoff`
retains the existing dimensionless fraction of the spatial ringing frequency
`2*f/c0` in cycles/m; it is not itself a dimensional wavenumber.

The standalone HDF5 regression `Runtime/tests/test_hdf5_storage.cpp` links against
the built native `Hdf5/Hdf5File.o`, `Logger/Logger.o` and HDF5 libraries. Supply a
temporary output file argument. The default checks need no GPU. Compile with
`TEST_CUDA_STORAGE`, and link `ComplexMatrix.o`, `BaseFloatMatrix.o` and CUDA,
to additionally test the actual complex-matrix writer on a rectangular 3D grid.
MEX input and partial-thread-block checks are included in `test_runtime('gpu')`.

Containers own matrices/streams with `unique_ptr`. `CudaBuffer` owns registered
host storage plus device storage or a mapped view; constructors unwind partial
allocation. FFT plans have one owner. Explicit flush/close reports errors;
destructors only release resources. Block-LU helpers propagate exceptions to the
MEX entrypoint after GPU arrays, library handles and events have been released.
