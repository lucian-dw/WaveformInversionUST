/*
 * This MEX file uses the mxGPUArray API with CUDA and CUSOLVER to perform
 * the block LU decomposition of the 9-point Helmholtz equation.
 *
 * This was initially implemented in MATLAB using the following code:
 *
 * % Calculate the Schur Complements and Their Inverses
 * invT = complex(zeros(Ny, Ny, Nx, 'single', 'gpuArray'));
 * D = diag(Dd(:,1)) + diag(Dl(:,1),-1) + diag(Du(:,1),1); T = D;
 * invTcurr = complex(zeros(Ny, 'single', 'gpuArray'));
 * for j = 2:Nx
 *     % Invert T over only the interior points
 *     invTcurr(2:end-1,2:end-1) = inv(T(2:end-1,2:end-1));
 *     invT(:,:,j-1) = invTcurr;
 *     D = diag(Dd(:,j)) + diag(Dl(:,j),-1) + diag(Du(:,j),1);
 *     L = diag(Ld(:,j-1)) + diag(Ll(:,j-1),-1) + diag(Lu(:,j-1),1);
 *     U = diag(Ud(:,j-1)) + diag(Ul(:,j-1),-1) + diag(Uu(:,j-1),1);
 *     T = D - L * invT(:,:,j-1) * U;
 * end
 * invT(:,:,Nx) = invInterior(T);
 * clearvars T D invTcurr; % Remove last T and invTcurr from memory
 *
 * schurUpdateFused evaluates the tridiagonal products directly. Each interior
 * Schur block is pivot-factorized and applied to identity pages below.
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "cuda.h"
#include "typeinfo"
#include "cufft.h"
#include "math.h"
#include <vector>
#include "cusolverDn.h" // To compile: mexcuda -lcusolver decompBlockLU.cu
#include "block_lu_mex_utils.h"

/* Choose a reasonably sized number of threads for the block. */
const unsigned int threadsPerBlockX = 16;
const unsigned int threadsPerBlockY = 16;

/* Device code - Create tridiagonal matrix */
void __global__ triDiagMat(cuComplex const * const Ad,
                           cuComplex const * const Al,
                           cuComplex const * const Au,
                           cuComplex * const A,
                           int const size)
{
    /* Calculate the global linear index, assuming a 1-d grid. */
    unsigned int const row_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int const col_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_idx < size && col_idx < size) {
        if (row_idx == col_idx) {
            A[row_idx + size_t(size)*col_idx] = Ad[row_idx];
        } else if (row_idx == col_idx+1) {
            A[row_idx + size_t(size)*col_idx] = Al[col_idx];
        } else if (row_idx == col_idx-1) {
            A[row_idx + size_t(size)*col_idx] = Au[row_idx];
        } else {
            A[row_idx + size_t(size)*col_idx] = make_cuFloatComplex(0,0);
        }
    }
}

/* Device code - Copy Interior of Matrix */


/* Device code - Initialize Interior Identity Matrix */


/* Device code - Tridiagonal Matrix Multiply from Left */


/* Device code - Tridiagonal Matrix Multiply from Right */


/* Fuse A*U and D-L*(A*U), avoiding a full Ny-by-Ny scratch round trip. */
void __global__ schurUpdateFused(cuComplex const * const A,
                                 cuComplex const * const Ld,
                                 cuComplex const * const Ll,
                                 cuComplex const * const Lu,
                                 cuComplex const * const Ud,
                                 cuComplex const * const Ul,
                                 cuComplex const * const Uu,
                                 cuComplex const * const Dd,
                                 cuComplex const * const Dl,
                                 cuComplex const * const Du,
                                 cuComplex * const T,
                                 int const size)
{
    extern __shared__ cuComplex tile[];
    int const tileRows = blockDim.x + 2;
    int const tileCols = blockDim.y + 2;
    int const rowStart = blockIdx.x * blockDim.x;
    int const colStart = blockIdx.y * blockDim.y;
    int const localThread = threadIdx.x + blockDim.x * threadIdx.y;
    int const threadCount = blockDim.x * blockDim.y;

    for (int index = localThread; index < tileRows * tileCols;
         index += threadCount) {
        int const tileRow = index % tileRows;
        int const tileCol = index / tileRows;
        int const globalRow = rowStart + tileRow - 1;
        int const globalCol = colStart + tileCol - 1;
        tile[index] = (globalRow >= 0 && globalRow < size &&
                       globalCol >= 0 && globalCol < size)
            ? A[globalRow + size_t(size)* globalCol]
            : make_cuFloatComplex(0, 0);
    }
    __syncthreads();

    int const row = rowStart + threadIdx.x;
    int const col = colStart + threadIdx.y;
    if (row >= size || col >= size) {
        return;
    }
    int const center = (threadIdx.x + 1) +
                       tileRows * (threadIdx.y + 1);

    cuComplex rightMinus = make_cuFloatComplex(0, 0);
    cuComplex rightCenter = Ud[col] * tile[center];
    cuComplex rightPlus = make_cuFloatComplex(0, 0);
    if (row > 0) {
        rightMinus = Ud[col] * tile[center - 1];
    }
    if (row < size - 1) {
        rightPlus = Ud[col] * tile[center + 1];
    }
    if (col > 0) {
        rightCenter += Uu[col - 1] * tile[center - tileRows];
        if (row > 0) {
            rightMinus += Uu[col - 1] * tile[center - 1 - tileRows];
        }
        if (row < size - 1) {
            rightPlus += Uu[col - 1] * tile[center + 1 - tileRows];
        }
    }
    if (col < size - 1) {
        rightCenter += Ul[col] * tile[center + tileRows];
        if (row > 0) {
            rightMinus += Ul[col] * tile[center - 1 + tileRows];
        }
        if (row < size - 1) {
            rightPlus += Ul[col] * tile[center + 1 + tileRows];
        }
    }

    cuComplex diagonal = make_cuFloatComplex(0, 0);
    if (row == col) {
        diagonal = Dd[row];
    } else if (row == col + 1) {
        diagonal = Dl[col];
    } else if (row + 1 == col) {
        diagonal = Du[row];
    }

    cuComplex value = diagonal - Ld[row] * rightCenter;
    if (row > 0) {
        value -= Ll[row - 1] * rightMinus;
    }
    if (row < size - 1) {
        value -= Lu[row] * rightPlus;
    }
    T[row + size_t(size)* col] = value;
}

void __global__ initializeIdentityPages(cuComplex * const pages,
                                        int const size,
                                        int const pageCount)
{
    int const index = blockIdx.x * blockDim.x + threadIdx.x;
    int const interior = size - 2;
    int const total = interior * pageCount;
    if (index < total) {
        int const page = index / interior;
        int const row = index - page * interior + 1;
        pages[static_cast<size_t>(page) * size_t(size)* size + row + size_t(size)* row] =
            make_cuFloatComplex(1, 0);
    }
}


/*
 * Host code
 */
void runMex(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
    WustMexResources resources;
    /* Declare all variables.*/
    mxGPUArray const *Ld;
    mxGPUArray const *Ll;
    mxGPUArray const *Lu;
    mxGPUArray const *Dd;
    mxGPUArray const *Dl;
    mxGPUArray const *Du;
    mxGPUArray const *Ud;
    mxGPUArray const *Ul;
    mxGPUArray const *Uu;
    mxGPUArray *T;
    mxGPUArray *invT;
    cuComplex const *d_Ld;
    cuComplex const *d_Ll;
    cuComplex const *d_Lu;
    cuComplex const *d_Dd;
    cuComplex const *d_Dl;
    cuComplex const *d_Du;
    cuComplex const *d_Ud;
    cuComplex const *d_Ul;
    cuComplex const *d_Uu;
    cuComplex *d_T;
    cuComplex *d_invT;

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* Throw an error if the input is not a GPU array. */
    if (nlhs < 1 || nlhs > 2) {
        wustFail("WUST:InvalidOutput",
                          "decompBlockLU requires invT and optionally a timing struct.");
    }
    bool const collectTimings = (nlhs == 2);
    if ( (nrhs!=9) ||
        !(mxIsGPUArray(prhs[0])) ||
        !(mxIsGPUArray(prhs[1])) ||
        !(mxIsGPUArray(prhs[2])) ||
        !(mxIsGPUArray(prhs[3])) ||
        !(mxIsGPUArray(prhs[4])) ||
        !(mxIsGPUArray(prhs[5])) ||
        !(mxIsGPUArray(prhs[6])) ||
        !(mxIsGPUArray(prhs[7])) ||
        !(mxIsGPUArray(prhs[8])) ) {
        wustFail("WUST:InvalidInput",
                          "Invalid input to triDiagMultLeftGPUmex: Expecting 9 gpuArray inputs");
    }

    /* Assemble GPU Arrays from Inputs */
    Ld = resources.input(prhs[0]);
    Ll = resources.input(prhs[1]);
    Lu = resources.input(prhs[2]);
    Dd = resources.input(prhs[3]);
    Dl = resources.input(prhs[4]);
    Du = resources.input(prhs[5]);
    Ud = resources.input(prhs[6]);
    Ul = resources.input(prhs[7]);
    Uu = resources.input(prhs[8]);

    const mxGPUArray* inputs[] = {Ld, Ll, Lu, Dd, Dl, Du, Ud, Ul, Uu};
    for (const auto* input : inputs) {
        if (mxGPUGetNumberOfDimensions(input) != 2 ||
            mxGPUGetClassID(input) != mxSINGLE_CLASS ||
            mxGPUGetComplexity(input) != mxCOMPLEX) {
            wustFail("WUST:InvalidInput", "Block-LU requires 2D complex-single GPU blocks.");
        }
    }
    if (mxGPUGetDimensions(Dd)[0] < 3 || mxGPUGetDimensions(Dd)[1] < 2 ||
        mxGPUGetNumberOfElements(Dd) > 2147483647) {
        wustFail("WUST:InvalidInput", "Block-LU grid must have an interior and fit int indexing.");
    }

    /* Error Checking with Array Dimensions */
    int Ny = (int) mxGPUGetDimensions(Dd)[0];
    int Nx = (int) mxGPUGetDimensions(Dd)[1];
    if (mxGPUGetDimensions(Ld)[0] != Ny || mxGPUGetDimensions(Ld)[1] != Nx-1 ||
        mxGPUGetDimensions(Ll)[0] != Ny-1 || mxGPUGetDimensions(Ll)[1] != Nx-1 ||
        mxGPUGetDimensions(Lu)[0] != Ny-1 || mxGPUGetDimensions(Lu)[1] != Nx-1 ||
        mxGPUGetDimensions(Dd)[0] != Ny || mxGPUGetDimensions(Dd)[1] != Nx ||
        mxGPUGetDimensions(Dl)[0] != Ny-1 || mxGPUGetDimensions(Dl)[1] != Nx ||
        mxGPUGetDimensions(Du)[0] != Ny-1 || mxGPUGetDimensions(Du)[1] != Nx ||
        mxGPUGetDimensions(Ud)[0] != Ny || mxGPUGetDimensions(Ud)[1] != Nx-1 ||
        mxGPUGetDimensions(Ul)[0] != Ny-1 || mxGPUGetDimensions(Ul)[1] != Nx-1 ||
        mxGPUGetDimensions(Uu)[0] != Ny-1 || mxGPUGetDimensions(Uu)[1] != Nx-1) {
        wustFail("WUST:InvalidInput", "Block-LU diagonal/page dimensions disagree.");
    }

    /*
     * Now that we have verified the data type, extract a pointer to the input
     * data on the device.
     */
    d_Ld = (cuComplex const *)(mxGPUGetDataReadOnly(Ld));
    d_Ll = (cuComplex const *)(mxGPUGetDataReadOnly(Ll));
    d_Lu = (cuComplex const *)(mxGPUGetDataReadOnly(Lu));
    d_Dd = (cuComplex const *)(mxGPUGetDataReadOnly(Dd));
    d_Dl = (cuComplex const *)(mxGPUGetDataReadOnly(Dl));
    d_Du = (cuComplex const *)(mxGPUGetDataReadOnly(Du));
    d_Ud = (cuComplex const *)(mxGPUGetDataReadOnly(Ud));
    d_Ul = (cuComplex const *)(mxGPUGetDataReadOnly(Ul));
    d_Uu = (cuComplex const *)(mxGPUGetDataReadOnly(Uu));

    /* Create GPUArrays to hold the intermediate results and get their underlying pointers. */
    mwSize dims_T[2] = {(mwSize) Ny, (mwSize) Ny};
    T = resources.array(2, dims_T,
                            mxGPUGetClassID(Dd),
                            mxGPUGetComplexity(Dd),
                            MX_GPU_DO_NOT_INITIALIZE);
    d_T = (cuComplex *)(mxGPUGetData(T));
    /* Create a GPUArray to hold the final result and get its underlying pointer. */
    mwSize dims_invT[3] = {(mwSize) Ny, (mwSize) Ny, (mwSize) Nx};
    invT = resources.array(3, dims_invT,
                               mxGPUGetClassID(Dd),
                               mxGPUGetComplexity(Dd),
                               MX_GPU_INITIALIZE_VALUES);
    d_invT = (cuComplex *)(mxGPUGetData(invT));

    /*
     * Call the kernel using the CUDA runtime API. We are using a 2-d grid here,
     * and it would be possible for the number of elements to be too large for
     * the grid. For this example we are not guarding against this possibility.
     */
    dim3 dimBlockBP(threadsPerBlockX, threadsPerBlockY, 1);
	dim3 dimGridBP((Ny + dimBlockBP.x - 1) / dimBlockBP.x,
		(Ny + dimBlockBP.y - 1) / dimBlockBP.y, 1);

    /* Prepare cuSOLVER for matrix inversion */
    cusolverDnHandle_t cusolverH;
    USCT_CUSOLVER_CHECK(resources.create(&cusolverH));
    int Lwork = 0; // size of work space: determine size of work space on next line
    USCT_CUSOLVER_CHECK(cusolverDnCgetrf_bufferSize(cusolverH, Ny-2, Ny-2,
                                &d_T[Ny+1], Ny, &Lwork));
    mwSize mwSizeLwork = (mwSize) Lwork;

    /* Prepare additional inputs for cuSOLVER: Workspace, devIpiv, devInfo */
    mxGPUArray * workspace = resources.array(1, &mwSizeLwork,
                                                 mxGPUGetClassID(Dd),
                                                 mxGPUGetComplexity(Dd),
                                                 MX_GPU_DO_NOT_INITIALIZE);
    void * d_work = mxGPUGetData(workspace);
    mwSize N = (mwSize) Ny;
    mxGPUArray * Ipiv = resources.array(1, &N,
                                            mxINT32_CLASS, mxREAL,
                                            MX_GPU_DO_NOT_INITIALIZE);
    int * d_Ipiv = (int *)(mxGPUGetData(Ipiv)); // Assuming int is 32 bit
    mwSize infoCount = (mwSize) Nx;
    mxGPUArray * factorInfo = resources.array(1, &infoCount,
                                            mxINT32_CLASS, mxREAL,
                                            MX_GPU_INITIALIZE_VALUES);
    mxGPUArray * solveInfo = resources.array(1, &infoCount,
                                            mxINT32_CLASS, mxREAL,
                                            MX_GPU_INITIALIZE_VALUES);
    int * d_factorInfo = (int *)(mxGPUGetData(factorInfo));
    int * d_solveInfo = (int *)(mxGPUGetData(solveInfo));

    cudaEvent_t totalStart = nullptr;
    cudaEvent_t totalEnd = nullptr;
    if (collectTimings) {
        USCT_CUDA_CHECK(resources.event(&totalStart));
        USCT_CUDA_CHECK(resources.event(&totalEnd));
        USCT_CUDA_CHECK(cudaEventRecord(totalStart));
    }

    // One pivoted LU and inverse solve for each Schur page.
    auto invertPage = [&](int page) {
        USCT_CUSOLVER_CHECK(cusolverDnCgetrf(cusolverH, Ny-2, Ny-2,
            &d_T[Ny+1], Ny, (cuComplex *)d_work, d_Ipiv, &d_factorInfo[page]));
        USCT_CUSOLVER_CHECK(cusolverDnCgetrs(cusolverH, CUBLAS_OP_N, Ny-2, Ny-2,
            &d_T[Ny+1], Ny, d_Ipiv, &d_invT[size_t(Ny)*Ny*page+Ny+1], Ny, &d_solveInfo[page]));
    };

    /* Initialize all inverse pages once, then run the serial recurrence. */
    int const identityThreads = 256;
    int const identityCount = (Ny - 2) * Nx;
    initializeIdentityPages<<<(identityCount + identityThreads - 1) / identityThreads,
                              identityThreads>>>(d_invT, Ny, Nx);
    USCT_KERNEL_CHECK();

    /* Calculate the Schur Complements and Their Inverses */
    triDiagMat<<<dimGridBP, dimBlockBP>>>(d_Dd, d_Dl, d_Du, d_T, Ny);
    USCT_KERNEL_CHECK();
    size_t const schurSharedBytes =
        (dimBlockBP.x + 2) * (dimBlockBP.y + 2) * sizeof(cuComplex);
    for (int x_idx = 1; x_idx < Nx; x_idx++) {
        invertPage(x_idx-1);
        schurUpdateFused<<<dimGridBP, dimBlockBP, schurSharedBytes>>>(
            &d_invT[size_t(Ny)*Ny*(x_idx-1)],
            &d_Ld[size_t(Ny)*(x_idx-1)], &d_Ll[size_t(Ny-1)*(x_idx-1)],
            &d_Lu[size_t(Ny-1)*(x_idx-1)], &d_Ud[size_t(Ny)*(x_idx-1)],
            &d_Ul[size_t(Ny-1)*(x_idx-1)], &d_Uu[size_t(Ny-1)*(x_idx-1)],
            &d_Dd[size_t(Ny)*x_idx], &d_Dl[size_t(Ny-1)*x_idx],
            &d_Du[size_t(Ny-1)*x_idx], d_T, Ny);
        USCT_KERNEL_CHECK();
    }
    invertPage(Nx-1);
    if (collectTimings) {
        USCT_CUDA_CHECK(cudaEventRecord(totalEnd));
    }

    usctCheckDeviceInfo(d_factorInfo, Nx, "getrf");
    usctCheckDeviceInfo(d_solveInfo, Nx, "getrs");

    /* Must destroy cuSOLVER handle at the end */

    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(invT);
    if (collectTimings) {
        USCT_CUDA_CHECK(cudaEventSynchronize(totalEnd));
        float totalMilliseconds = 0;
        USCT_CUDA_CHECK(cudaEventElapsedTime(&totalMilliseconds, totalStart, totalEnd));
        char const *fieldNames[] = {"total_gpu_ms", "block_count"};
        plhs[1] = mxCreateStructMatrix(1, 1, 2, fieldNames);
        mxSetField(plhs[1], 0, "total_gpu_ms", mxCreateDoubleScalar(totalMilliseconds));
        mxSetField(plhs[1], 0, "block_count", mxCreateDoubleScalar(Nx));
    }

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */

    resources.close();
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
    try { runMex(nlhs, plhs, nrhs, prhs); }
    catch (const WustMexError& error) { mexErrMsgIdAndTxt(error.id.c_str(), "%s", error.what()); }
    catch (const std::exception& error) { mexErrMsgIdAndTxt("WUST:RuntimeError", "%s", error.what()); }
}
