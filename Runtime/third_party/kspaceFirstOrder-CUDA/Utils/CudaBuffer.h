#ifndef CUDA_BUFFER_H
#define CUDA_BUFFER_H

#include <cuda_runtime.h>
#include <mm_malloc.h>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

// Own aligned registered host storage and either device storage or its mapped view.
template<typename T>
class CudaBuffer
{
public:
  CudaBuffer() = default;
  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;
  ~CudaBuffer() noexcept { reset(); }

  void allocate(size_t capacity, size_t alignment, bool mapped = false)
  {
    if (mHost || capacity == 0 || capacity > std::numeric_limits<size_t>::max() / sizeof(T))
      throw std::invalid_argument("Invalid CUDA buffer allocation");
    const size_t bytes = capacity * sizeof(T);
    mHost = static_cast<T*>(_mm_malloc(bytes, alignment));
    if (!mHost) throw std::bad_alloc();
    try
    {
      check(cudaHostRegister(mHost, bytes, cudaHostRegisterPortable |
                             (mapped ? cudaHostRegisterMapped : 0)));
      mRegistered = true;
      mMapped = mapped;
      if (mapped) check(cudaHostGetDevicePointer(&mDevice, mHost, 0));
      else check(cudaMalloc(&mDevice, bytes));
    }
    catch (...)
    {
      reset();
      throw;
    }
  }

  void reset() noexcept
  {
    if (mDevice && !mMapped) cudaFree(mDevice);
    if (mRegistered) cudaHostUnregister(mHost);
    if (mHost) _mm_free(mHost);
    mHost = nullptr;
    mDevice = nullptr;
    mRegistered = false;
    mMapped = false;
  }
  T* host() const { return mHost; }
  T* device() const { return mDevice; }

private:
  static void check(cudaError_t status)
  {
    if (status != cudaSuccess)
      throw std::runtime_error(std::string("CUDA buffer: ") + cudaGetErrorString(status));
  }
  T* mHost = nullptr;
  T* mDevice = nullptr;
  bool mRegistered = false;
  bool mMapped = false;
};
#endif
