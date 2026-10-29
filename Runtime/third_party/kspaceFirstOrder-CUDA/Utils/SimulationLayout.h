#ifndef SIMULATION_LAYOUT_H
#define SIMULATION_LAYOUT_H
#include <Utils/DimensionSizes.h>

// Batch planes are independent 2D simulations, never a third physical axis.
struct SimulationLayout
{
  DimensionSizes physical;
  DimensionSizes spectrum;
  size_t batchCount = 1;
  bool is3D() const { return physical.nz > 1; }
  DimensionSizes field() const
  {
    return batchCount == 1 ? physical : DimensionSizes(physical.nx, physical.ny, batchCount);
  }
  DimensionSizes fieldSpectrum() const
  {
    return batchCount == 1 ? spectrum : DimensionSizes(spectrum.nx, spectrum.ny, batchCount);
  }
};
#endif
