#ifndef USCT_BLOCK_LU_MEX_UTILS_H
#define USCT_BLOCK_LU_MEX_UTILS_H

#include "mex.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cusolverDn.h"
#include <vector>
#include <functional>
#include <stdexcept>
#include <string>
#include <cstdarg>
#include <cstdio>
#include "gpu/mxGPUArray.h"

/* Helpful Operator Overloading -- Using cuComplex.h */
inline __host__ __device__ cuFloatComplex operator*(cuFloatComplex a, cuFloatComplex b) {
    return cuCmulf(a,b);
}
inline __host__ __device__ void operator*=(cuFloatComplex &a, cuFloatComplex b) {
    const cuFloatComplex c = a * b;
    a.x = c.x; a.y = c.y;
}
inline __host__ __device__ cuFloatComplex operator/(cuFloatComplex a, cuFloatComplex b) {
    return cuCdivf(a,b);
}
inline __host__ __device__ void operator/=(cuFloatComplex &a, cuFloatComplex b) {
    const cuFloatComplex c = a / b;
    a.x = c.x; a.y = c.y;
}
inline __host__ __device__ cuFloatComplex operator+(cuFloatComplex a, cuFloatComplex b) {
    return cuCaddf(a,b);
}
inline __host__ __device__ void operator+=(cuFloatComplex &a, cuFloatComplex b) {
    const cuFloatComplex c = a + b;
    a.x = c.x; a.y = c.y;
}
inline __host__ __device__ cuFloatComplex operator-(cuFloatComplex a, cuFloatComplex b) {
    return cuCsubf(a,b);
}
inline __host__ __device__ void operator-=(cuFloatComplex &a, cuFloatComplex b) {
    const cuFloatComplex c = a - b;
    a.x = c.x; a.y = c.y;
}



struct WustMexError : std::runtime_error {
    std::string id;
    WustMexError(const char* identifier, const std::string& message)
        : std::runtime_error(message), id(identifier) {}
};

inline void wustFail(const char* id, const char* format, ...) {
    va_list args;
    va_start(args, format);
    va_list copy;
    va_copy(copy, args);
    const int count = vsnprintf(nullptr, 0, format, copy);
    va_end(copy);
    std::vector<char> message(static_cast<size_t>(count >= 0 ? count : 0) + 1);
    vsnprintf(message.data(), message.size(), format, args);
    va_end(args);
    throw WustMexError(id, message.data());
}

// The entrypoint owns resources; numerical helpers throw, never longjmp past cleanup.
class WustMexResources {
public:
    ~WustMexResources() noexcept { release(); }
    const mxGPUArray* input(const mxArray* value) {
        const mxGPUArray* array = mxGPUCreateFromMxArray(value);
        cleanup.emplace_back([array] { mxGPUDestroyGPUArray(array); return 0; });
        return array;
    }
    mxGPUArray* array(mwSize rank, const mwSize* dims, mxClassID type,
                      mxComplexity complexity, mxGPUInitialize initialize) {
        mxGPUArray* value = mxGPUCreateGPUArray(rank, dims, type, complexity, initialize);
        cleanup.emplace_back([value] { mxGPUDestroyGPUArray(value); return 0; });
        return value;
    }
    cublasStatus_t create(cublasHandle_t* handle) {
        const cublasStatus_t status = cublasCreate(handle);
        if (status == CUBLAS_STATUS_SUCCESS) {
            const auto value = *handle;
            cleanup.emplace_back([value] { return static_cast<int>(cublasDestroy(value)); });
        }
        return status;
    }
    cusolverStatus_t create(cusolverDnHandle_t* handle) {
        const cusolverStatus_t status = cusolverDnCreate(handle);
        if (status == CUSOLVER_STATUS_SUCCESS) {
            const auto value = *handle;
            cleanup.emplace_back([value] { return static_cast<int>(cusolverDnDestroy(value)); });
        }
        return status;
    }
    cudaError_t event(cudaEvent_t* handle) {
        const cudaError_t status = cudaEventCreate(handle);
        if (status == cudaSuccess) {
            const auto value = *handle;
            cleanup.emplace_back([value] { return static_cast<int>(cudaEventDestroy(value)); });
        }
        return status;
    }
    cudaError_t device(void** pointer, size_t bytes) {
        const cudaError_t status = cudaMalloc(pointer, bytes);
        if (status == cudaSuccess) {
            const auto value = *pointer;
            cleanup.emplace_back([value] { return static_cast<int>(cudaFree(value)); });
        }
        return status;
    }
    void* host(size_t bytes) {
        void* value = mxMalloc(bytes);
        cleanup.emplace_back([value] { mxFree(value); return 0; });
        return value;
    }
#if CUDART_VERSION >= 11000
    cusolverStatus_t params(cusolverDnParams_t* handle) {
        const cusolverStatus_t status = cusolverDnCreateParams(handle);
        if (status == CUSOLVER_STATUS_SUCCESS) {
            const auto value = *handle;
            cleanup.emplace_back([value] { return static_cast<int>(cusolverDnDestroyParams(value)); });
        }
        return status;
    }
#endif
    void close() {
        const int status = release();
        if (status) wustFail("WUST:ResourceCleanup", "GPU resource release failed: %d", status);
    }
private:
    std::vector<std::function<int()>> cleanup;
    int release() noexcept {
        int first = 0;
        for (auto it = cleanup.rbegin(); it != cleanup.rend(); ++it) {
            const int status = (*it)();
            if (!first) first = status;
        }
        cleanup.clear();
        return first;
    }
};

#define USCT_CUDA_CHECK(call) do { \
    cudaError_t usct_status_ = (call); \
    if (usct_status_ != cudaSuccess) { \
        wustFail("WUST:CUDAError", \
            "%s failed: %s (%d)", #call, cudaGetErrorString(usct_status_), \
            static_cast<int>(usct_status_)); \
    } \
} while (0)

#define USCT_CUBLAS_CHECK(call) do { \
    cublasStatus_t usct_status_ = (call); \
    if (usct_status_ != CUBLAS_STATUS_SUCCESS) { \
        wustFail("WUST:CUBLASError", \
            "%s failed with cuBLAS status %d", #call, \
            static_cast<int>(usct_status_)); \
    } \
} while (0)

#define USCT_CUSOLVER_CHECK(call) do { \
    cusolverStatus_t usct_status_ = (call); \
    if (usct_status_ != CUSOLVER_STATUS_SUCCESS) { \
        wustFail("WUST:CUSOLVERError", \
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
            wustFail("WUST:InvalidSolverArgument",
                "%s block %d reported illegal argument %d", operation,
                index + 1, -hostInfo[index]);
        }
        if (hostInfo[index] > 0) {
            wustFail("WUST:SingularSchurBlock",
                "%s block %d is singular at pivot %d", operation,
                index + 1, hostInfo[index]);
        }
    }
}

#endif
