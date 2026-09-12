#ifndef USCT_BLOCK_LU_MEX_UTILS_H
#define USCT_BLOCK_LU_MEX_UTILS_H

#include "mex.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cusolverDn.h"
#include <vector>

#define USCT_CUDA_CHECK(call) do { \
    cudaError_t usct_status_ = (call); \
    if (usct_status_ != cudaSuccess) { \
        mexErrMsgIdAndTxt("usct:blocklu:CUDAError", \
            "%s failed: %s (%d)", #call, cudaGetErrorString(usct_status_), \
            static_cast<int>(usct_status_)); \
    } \
} while (0)

#define USCT_CUBLAS_CHECK(call) do { \
    cublasStatus_t usct_status_ = (call); \
    if (usct_status_ != CUBLAS_STATUS_SUCCESS) { \
        mexErrMsgIdAndTxt("usct:blocklu:CUBLASError", \
            "%s failed with cuBLAS status %d", #call, \
            static_cast<int>(usct_status_)); \
    } \
} while (0)

#define USCT_CUSOLVER_CHECK(call) do { \
    cusolverStatus_t usct_status_ = (call); \
    if (usct_status_ != CUSOLVER_STATUS_SUCCESS) { \
        mexErrMsgIdAndTxt("usct:blocklu:CUSOLVERError", \
            "%s failed with cuSOLVER status %d", #call, \
            static_cast<int>(usct_status_)); \
    } \
} while (0)

#define USCT_KERNEL_CHECK() USCT_CUDA_CHECK(cudaPeekAtLastError())

inline void usctCheckDeviceInfo(const int *deviceInfo, int count,
                                const char *operation)
{
    std::vector<int> hostInfo(count, 0);
    USCT_CUDA_CHECK(cudaMemcpy(hostInfo.data(), deviceInfo,
        static_cast<size_t>(count) * sizeof(int), cudaMemcpyDeviceToHost));
    for (int index = 0; index < count; ++index) {
        if (hostInfo[index] < 0) {
            mexErrMsgIdAndTxt("usct:blocklu:InvalidSolverArgument",
                "%s block %d reported illegal argument %d", operation,
                index + 1, -hostInfo[index]);
        }
        if (hostInfo[index] > 0) {
            mexErrMsgIdAndTxt("usct:blocklu:SingularSchurBlock",
                "%s block %d is singular at pivot %d", operation,
                index + 1, hostInfo[index]);
        }
    }
}

#endif
