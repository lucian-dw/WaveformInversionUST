/**
 * @file      BaseOutputStream.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file of the class saving RealMatrix data into the output HDF5 file.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      11 July      2012, 10:30 (created) \n
 *            11 February  2020, 16:21 (revised)
 *
 * @copyright Copyright (C) 2012 - 2020 SC\@FIT Research Group, Brno University of Technology, Brno, CZ.
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

#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include <limits>

// Windows build needs to undefine macro MINMAX to support std::limits
#ifdef _WIN64
  #ifndef NOMINMAX
    # define NOMINMAX
  #endif

  #include <windows.h>
#endif

#include <OutputStreams/BaseOutputStream.h>
#include <OutputStreams/OutputStreamsCudaKernels.cuh>

#include <Logger/Logger.h>
#include <Parameters/Parameters.h>

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Constructor - there is no sensor mask by default!
 */
BaseOutputStream::BaseOutputStream(Hdf5File&            file,
                                   const MatrixName&    rootObjectName,
                                   const RealMatrix&    sourceMatrix,
                                   const ReduceOperator reduceOp)
  : mFile(file),
    mRootObjectName(rootObjectName),
    mSourceMatrix(sourceMatrix),
    mReduceOp(reduceOp),
    mBufferSize(0),
    mHostBuffer(nullptr),
    mDeviceBuffer(nullptr)
{

}// end of BaseOutputStream
//----------------------------------------------------------------------------------------------------------------------

/**
 * Apply post-processing on the buffer (done on the device).
 */
void BaseOutputStream::postProcess()
{
  switch (mReduceOp)
  {
    case ReduceOperator::kNone:
    {
      // Do nothing
      break;
    }

    case ReduceOperator::kRms:
    {
      const float scalingCoeff = 1.0f / (Parameters::getInstance().getNt() -
                                         Parameters::getInstance().getSamplingStartTimeIndex());

      OutputStreamsCudaKernels::postProcessingRms(mDeviceBuffer, scalingCoeff, mBufferSize);
      break;
    }

    case ReduceOperator::kMax:
    {
      // Do nothing
      break;
    }

    case ReduceOperator::kMin:
    {
      // Do nothing
      break;
    }
  }// switch
}// end of postProcess
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Protected methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Allocate memory using a proper memory alignment.
 */
void BaseOutputStream::allocateMemory()
{
  const bool mapped = mReduceOp == ReduceOperator::kNone;
  mBuffer.allocate(mBufferSize, kDataAlignment, mapped);
  mHostBuffer = mBuffer.host();
  mDeviceBuffer = mBuffer.device();
  float initial = 0.0f;
  if (mReduceOp == ReduceOperator::kMax) initial = -std::numeric_limits<float>::max();
  if (mReduceOp == ReduceOperator::kMin) initial = std::numeric_limits<float>::max();
  // First touch on the socket hosting the GPU; raw streams use mapped host memory.
  std::fill_n(mHostBuffer, mBufferSize, initial);
  if (!mapped) copyToDevice();
}// end of allocateMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * Free memory.
 */
void BaseOutputStream::freeMemory()
{
  mBuffer.reset();
  mHostBuffer = nullptr;
  mDeviceBuffer = nullptr;
}// end of FreeMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 *  Copy data hostBuffer -> deviceBuffer
 */
void BaseOutputStream::copyToDevice()
{
  cudaCheckErrors(cudaMemcpy(mDeviceBuffer, mHostBuffer, mBufferSize * sizeof(float), cudaMemcpyHostToDevice));
}// end of copyToDevice
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy data deviceBuffer -> hostBuffer
 */
void BaseOutputStream::copyFromDevice()
{
  cudaCheckErrors(cudaMemcpy(mHostBuffer, mDeviceBuffer, mBufferSize * sizeof(float), cudaMemcpyDeviceToHost));
}// end of copyFromDevice
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//
