#ifndef CUFFT_PLAN_H
#define CUFFT_PLAN_H
#include <cufft.h>
#include <stdexcept>

class CufftPlan
{
public:
  CufftPlan() = default;
  CufftPlan(const CufftPlan&) = delete;
  CufftPlan& operator=(const CufftPlan&) = delete;
  ~CufftPlan() noexcept { close(); }
  cufftHandle get() const { return mHandle; }
  cufftHandle* address()
  {
    if (mHandle) throw std::logic_error("FFT plan already initialized");
    return &mHandle;
  }
  cufftResult close() noexcept
  {
    const cufftHandle handle = mHandle;
    mHandle = 0;
    return handle ? cufftDestroy(handle) : CUFFT_SUCCESS;
  }
private:
  cufftHandle mHandle = 0;
};
#endif
