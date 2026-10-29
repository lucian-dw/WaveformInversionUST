/**
 * @file      BaseIndexMatrix.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file containing the base class for index matrices (based on the size_t datatype).
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      26 July      2011, 14:17 (created) \n
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

#include <immintrin.h>

#include <MatrixClasses/BaseIndexMatrix.h>
#include <Utils/DimensionSizes.h>
#include <Logger/Logger.h>

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Default constructor.
 */
BaseIndexMatrix::BaseIndexMatrix()
  : BaseMatrix(),
    mDimensionSizes(),
    mSize(0),
    mCapacity(0),
    mHostData(nullptr),
    mDeviceData(nullptr)
{

}// end of BaseIndexMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Zero all allocated elements.
 */
void BaseIndexMatrix::zeroMatrix()
{
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < mCapacity; i++)
  {
    mHostData[i] = size_t(0);
  }
}// end of zeroMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy data from host -> device (CPU -> GPU).
 */
void BaseIndexMatrix::copyToDevice()
{
  cudaCheckErrors(cudaMemcpy(mDeviceData, mHostData, mCapacity * sizeof(size_t), cudaMemcpyHostToDevice));
}// end of copyToDevice
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy data from device-> host (GPU -> CPU).
 */
void BaseIndexMatrix::copyFromDevice()
{
  cudaCheckErrors(cudaMemcpy(mHostData, mDeviceData, mCapacity * sizeof(size_t), cudaMemcpyDeviceToHost));
}// end of copyFromDevice
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Protected methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Memory allocation based on the capacity. \n
 *
 * CPU memory is aligned and registered; readData fills the complete index buffer.
 * The GPU memory is allocated on GPU but not zeroed (no reason).
 */
void BaseIndexMatrix::allocateMemory()
{
  mBuffer.allocate(mCapacity, kDataAlignment);
  mHostData = mBuffer.host();
  mDeviceData = mBuffer.device();
}// end of allocateMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * Free memory.
 */
void BaseIndexMatrix::freeMemory()
{
  mBuffer.reset();
  mHostData = nullptr;
  mDeviceData = nullptr;
}// end of freeMemory
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Private methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//
