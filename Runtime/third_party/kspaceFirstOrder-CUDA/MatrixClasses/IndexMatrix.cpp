/**
 * @file      IndexMatrix.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file containing the class for 64b integer index matrices.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      26 July      2011, 15:16 (created) \n
 *            11 February  2020, 16:17 (revised)
 *
 * @copyright Copyright (C) 2011 - 2020 SC\@FIT Research Group, Brno University of Technology, Brno, CZ.
 *
 * This file is part of the C++ extension of the [k-Wave Toolbox](http://www.k-wave.org).
 *
 * k-Wave is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
 */

#include <MatrixClasses/IndexMatrix.h>
#include <Logger/Logger.h>
#include <Parameters/Parameters.h>
#include <set>
#include <stdexcept>

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Constructor allocating memory.
 */
IndexMatrix::IndexMatrix(const DimensionSizes& dimensionSizes)
  : BaseIndexMatrix()
{
  initDimensions(dimensionSizes);
  allocateMemory();
}// end of IndexMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destructor.
 */
IndexMatrix::~IndexMatrix()
{
}// end of ~IndexMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Read data from HDF5 file (only from the root group).
 */
void IndexMatrix::readData(Hdf5File&         file,
                           const MatrixName& matrixName)
{
  // Check the datatype
  if (file.readMatrixDataType(file.getRootGroup(), matrixName) != Hdf5File::MatrixDataType::kIndex)
  {
    throw std::ios::failure(Logger::formatMessage(kErrFmtMatrixNotIndex, matrixName.c_str()));
  }

  // Check the domain type
  if (file.readMatrixDomainType(file.getRootGroup(), matrixName) != Hdf5File::MatrixDomainType::kReal)
  {
    throw std::ios::failure(Logger::formatMessage(kErrFmtMatrixNotReal,matrixName.c_str()));
  }

  // Read data
  file.readCompleteDataset(file.getRootGroup(), matrixName, mDimensionSizes, mHostData);
  mZeroBased = false;
  if (matrixName == "sensor_mask_corners" || matrixName == "sensor_mask_index" ||
      matrixName == "p_source_index" || matrixName == "u_source_index")
  {
    size_t nx, ny, nz;
    file.readScalarValue(file.getRootGroup(), "Nx", nx);
    file.readScalarValue(file.getRootGroup(), "Ny", ny);
    file.readScalarValue(file.getRootGroup(), "Nz", nz);
    if (matrixName == "sensor_mask_corners")
    {
      if (mSize % 6 != 0) throw std::invalid_argument("Cuboids require six coordinates");
      const size_t limits[] = {nx, ny, nz};
      for (size_t i = 0; i < mSize; i += 6)
        for (size_t axis = 0; axis < 3; ++axis)
          if (mHostData[i + axis] < 1 || mHostData[i + axis + 3] > limits[axis] ||
              mHostData[i + axis] > mHostData[i + axis + 3])
            throw std::invalid_argument("Invalid inclusive sensor cuboid");
    }
    else
    {
      // Native independent-TX inputs encode the batch offset in each index.
      const size_t capacity = nx * ny * nz * Parameters::getInstance().getLayout().batchCount;
      std::set<size_t> seen;
      for (size_t i = 0; i < mSize; ++i)
        if (mHostData[i] < 1 || mHostData[i] > capacity || !seen.insert(mHostData[i]).second)
          throw std::invalid_argument("Spatial indices must be unique and inside the declared grid/batch layout");
    }
  }
}// end of readData
//----------------------------------------------------------------------------------------------------------------------

/**
 * Write data to HDF5 file.
 */
void IndexMatrix::writeData(Hdf5File&         file,
                            const MatrixName& matrixName,
                            const size_t      compressionLevel)
{
  // Set chunks - may be necessary for long index based sensor masks
  DimensionSizes chunks = mDimensionSizes;
  chunks.nz = 1;

  // 1D matrices - chunk sizes were empirically set.
  if ((mDimensionSizes.ny == 1) && (mDimensionSizes.nz == 1))
  {
    if (mDimensionSizes.nx > 4 * kChunkSize1D8MB)
    { // Chunk = 8MB for big matrices (> 32MB).
      chunks.nx = kChunkSize1D8MB;
    }
    else if (mDimensionSizes.nx > 4 * kChunkSize1D1MB)
    { // Chunk = 1MB for big matrices (> 4 - 32MB).
      chunks.nx = kChunkSize1D1MB;
    }
    else if (mDimensionSizes.nx > 4 * kChunkSize1D128kB)
    { // Chunk = 128kB for big matrices (< 1MB).
      chunks.nx = kChunkSize1D128kB;
    }
  }

  // Create dataset and write a slab
  hid_t dataset = file.createDataset(file.getRootGroup(),
                                     matrixName,
                                     mDimensionSizes,
                                     chunks,
                                     Hdf5File::MatrixDataType::kIndex,
                                     compressionLevel);

  // Write at position [0,0,0].
  file.writeHyperSlab(dataset, DimensionSizes(0, 0, 0), mDimensionSizes, mHostData);

  file.closeDataset(dataset);

  // Write data and domain types
  file.writeMatrixDataType(file.getRootGroup(),   matrixName, Hdf5File::MatrixDataType::kIndex);
  file.writeMatrixDomainType(file.getRootGroup(), matrixName, Hdf5File::MatrixDomainType::kReal);
}// end of writeData
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get the top left corner of the index-th cuboid.\n
 * Cuboids are stored as 6-tuples (two 3D coordinates). This gives the first three coordinates.
 * This routine works only on the CPU side.
 */
DimensionSizes IndexMatrix::getTopLeftCorner(const size_t& index) const
{
  size_t x =  mHostData[6 * index    ];
  size_t y =  mHostData[6 * index + 1];
  size_t z =  mHostData[6 * index + 2];

  return DimensionSizes(x, y, z);
}// end of getTopLeftCorner
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get the top bottom right of the index-th cuboid. Cuboids are stored as 6-tuples (two 3D
 * coordinates). This gives the first three coordinates. This routine works only on the CPU side.
 */
DimensionSizes IndexMatrix::getBottomRightCorner(const size_t& index) const
{
  size_t x =  mHostData[6 * index + 3];
  size_t y =  mHostData[6 * index + 4];
  size_t z =  mHostData[6 * index + 5];

  return DimensionSizes(x, y, z);
}// end of GetBottomRightCorner
//----------------------------------------------------------------------------------------------------------------------

/**
 * Recompute indeces, MATLAB -> C++.
 */
void IndexMatrix::recomputeIndicesToCPP()
{
  if (mZeroBased) throw std::logic_error("Indices are already zero-based");
  for (size_t i = 0; i < mSize; ++i)
    if (mHostData[i] == 0) throw std::invalid_argument("Expected one-based indices");
  #pragma omp parallel for
  for (size_t i = 0; i < mSize; i++)
  {
    mHostData[i]--;
  }
  mZeroBased = true;
}// end of recomputeIndicesToCPP
//----------------------------------------------------------------------------------------------------------------------

/**
 * Recompute indeces, C++ -> MATLAB.
 */
void IndexMatrix::recomputeIndicesToMatlab()
{
  if (!mZeroBased) throw std::logic_error("Indices are already one-based");
  #pragma omp parallel for
  for (size_t i = 0; i < mSize; i++)
  {
    mHostData[i]++;
  }
  mZeroBased = false;
}// end of recomputeIndicesToMatlab
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get total number of elements in all cuboids to be able to allocate output file.
 */
size_t IndexMatrix::getSizeOfAllCuboids() const
{
  size_t elementSum = 0;
  for (size_t cuboidIdx = 0; cuboidIdx < mDimensionSizes.ny; cuboidIdx++)
  {
    const DimensionSizes first = getTopLeftCorner(cuboidIdx);
    const DimensionSizes last = getBottomRightCorner(cuboidIdx);
    elementSum += (last.nx - first.nx + 1) * (last.ny - first.ny + 1) * (last.nz - first.nz + 1);
  }

  return elementSum;
}// end of getSizeOfAllCuboids
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Protected methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Set necessary dimensions and auxiliary variables.
 */
void IndexMatrix::initDimensions(const DimensionSizes& dimensionSizes)
{
  mDimensionSizes = dimensionSizes;

  mSize = dimensionSizes.nx * dimensionSizes.ny * dimensionSizes.nz;
  mCapacity = mSize;
}// end of initDimensions
//----------------------------------------------------------------------------------------------------------------------
