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
 *     T = D - triDiagMultLeftGPUmex(Ld(:,j-1), Ll(:,j-1), Lu(:,j-1), ...
 *         triDiagMultRight(invT(:,:,j-1), Ud(:,j-1).', Ul(:,j-1).', Uu(:,j-1).'));
 * end
 * invT(:,:,Nx) = invInterior(T);
 * clearvars T D invTcurr; % Remove last T and invTcurr from memory
 *
 * See the separate triDiagMultRight.m and triDiagMultLeftGPUmex.cu file 
 * for additional details regarding the algorithm above. Although neither 
 * triDiagMultRight.m nor triDiagMultLeftGPUmex.cu are used in 
 * this MEX file, key components of those algorithms were used here.
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

/* Choose a reasonably sized number of threads for the block. */
#ifdef USCT_BLOCK_SIZE_32
const unsigned int threadsPerBlockX = 32;
const unsigned int threadsPerBlockY = 32;
#else
const unsigned int threadsPerBlockX = 16;
const unsigned int threadsPerBlockY = 16;
#endif

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
            A[row_idx + size*col_idx] = Ad[row_idx];
        } else if (row_idx == col_idx+1) {
            A[row_idx + size*col_idx] = Al[col_idx];
        } else if (row_idx == col_idx-1) {
            A[row_idx + size*col_idx] = Au[row_idx];
        } else {
            A[row_idx + size*col_idx] = make_cuFloatComplex(0,0);
        }
    }
}


/* Device code - Copy Interior of Matrix */
void __global__ copyInterior(cuComplex const * const A,
                             cuComplex * const B, int const N)
{
    /* Calculate the global linear index, assuming a 1-d grid. */
    unsigned int const row_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int const col_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_idx < N && col_idx < N) {
        // Border
        if ( (row_idx == 0) || (row_idx == N-1) || 
             (col_idx == 0) || (col_idx == N-1) ) {
            B[row_idx + N*col_idx] = make_cuFloatComplex(0,0);
        } 
        // Interior
        else {
            B[row_idx + N*col_idx] = A[row_idx + N*col_idx];
        }
    }
}

/* Device code - Initialize Interior Identity Matrix */
void __global__ initializeIdentity(cuComplex * const A, int const N)
{
    /* Calculate the global linear index, assuming a 1-d grid. */
    unsigned int const row_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int const col_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_idx < N && col_idx < N) {
        // Border
        if ( (row_idx != 0) && (row_idx != N-1) && (row_idx == col_idx) ) {
            A[row_idx + N*col_idx] = make_cuFloatComplex(1,0);
        } 
        // Interior
        else {
            A[row_idx + N*col_idx] = make_cuFloatComplex(0,0);
        }
    }
}

/* Device code - Tridiagonal Matrix Multiply from Left */
void __global__ triDiagMultLeftPlusD(cuComplex const * const A,
                                     cuComplex const * const Bd,
                                     cuComplex const * const Bl,
                                     cuComplex const * const Bu,
                                     cuComplex const * const Dd,
                                     cuComplex const * const Dl,
                                     cuComplex const * const Du,
                                     cuComplex * const C,
                                     int const size)
{
    /* Shared memory for loading A from global memory */
    __shared__ cuComplex sA[threadsPerBlockY][threadsPerBlockX+2];
    /* Calculate the global linear index, assuming a 1-d grid. */
    unsigned int const row_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int const col_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_idx < size && col_idx < size) {
        // Regular Entries
        sA[threadIdx.y][threadIdx.x+1] = A[row_idx +size*col_idx];
        // Entries at the Edge of Each Block of Threads
        if ( (threadIdx.x == 0) && (row_idx != 0) ) 
            sA[threadIdx.y][threadIdx.x] = A[row_idx-1 + size*col_idx];
        else if ( (threadIdx.x == blockDim.x-1) && (row_idx != size-1) )
            sA[threadIdx.y][threadIdx.x+2] = A[row_idx+1 + size*col_idx];
        __syncthreads();
        // Additional Tri-Diagonal D Input
        cuComplex D = make_cuFloatComplex(0,0);
        if (row_idx < size && col_idx < size) {
            if (row_idx == col_idx) {
                D = Dd[row_idx];
            } else if (row_idx == col_idx+1) {
                D = Dl[col_idx];
            } else if (row_idx == col_idx-1) {
                D = Du[row_idx];
            } 
        }
        // Actual Tridiagonal Matrix Multiply
        if (row_idx == 0) {
            C[row_idx + size*col_idx] = D - 
                (Bd[row_idx]*sA[threadIdx.y][threadIdx.x+1] +
                Bu[row_idx]*sA[threadIdx.y][threadIdx.x+2]);
        } else if (row_idx == size-1) {
            C[row_idx + size*col_idx] = D - 
                (Bd[row_idx]*sA[threadIdx.y][threadIdx.x+1] +
                Bl[row_idx-1]*sA[threadIdx.y][threadIdx.x]);
        } else {
            C[row_idx + size*col_idx] = D - 
                (Bd[row_idx]*sA[threadIdx.y][threadIdx.x+1] + 
                Bl[row_idx-1]*sA[threadIdx.y][threadIdx.x] +
                Bu[row_idx]*sA[threadIdx.y][threadIdx.x+2]); 
        }
    }
}

/* Device code - Tridiagonal Matrix Multiply from Right */
void __global__ triDiagMultRight(cuComplex const * const A,
                                 cuComplex const * const Bd,
                                 cuComplex const * const Bl,
                                 cuComplex const * const Bu,
                                 cuComplex * const C,
                                 int const numRows,
                                 int const numCols)
{
    /* Shared memory for loading A from global memory */
    __shared__ cuComplex sA[threadsPerBlockX][threadsPerBlockY+2];
    /* Calculate the global linear index, assuming a 1-d grid. */
    unsigned int const row_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int const col_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_idx < numRows && col_idx < numCols) {
        // Regular Entries
        sA[threadIdx.x][threadIdx.y+1] = A[row_idx + numRows*col_idx];
        // Entries at the Edge of Each Block of Threads
        if ( (threadIdx.y == 0) && (col_idx != 0) ) 
            sA[threadIdx.x][threadIdx.y] = A[row_idx + numRows*(col_idx-1)];
        else if ( (threadIdx.y == blockDim.y-1) && (col_idx != numCols-1) )
            sA[threadIdx.x][threadIdx.y+2] = A[row_idx + numRows*(col_idx+1)];
        __syncthreads();
        // Actual Tridiagonal Matrix Multiply
        if (col_idx == 0) {
            C[row_idx + numRows*col_idx] = 
                Bd[col_idx]*sA[threadIdx.x][threadIdx.y+1] +
                Bl[col_idx]*sA[threadIdx.x][threadIdx.y+2];
        } else if (col_idx == numCols-1) {
            C[row_idx + numRows*col_idx] = 
                Bd[col_idx]*sA[threadIdx.x][threadIdx.y+1] +
                Bu[col_idx-1]*sA[threadIdx.x][threadIdx.y];
        } else {
            C[row_idx + numRows*col_idx] = 
                Bd[col_idx]*sA[threadIdx.x][threadIdx.y+1] + 
                Bu[col_idx-1]*sA[threadIdx.x][threadIdx.y] +
                Bl[col_idx]*sA[threadIdx.x][threadIdx.y+2]; 
        }
    }
}

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
            ? A[globalRow + size * globalCol]
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
    T[row + size * col] = value;
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
        pages[static_cast<size_t>(page) * size * size + row + size * row] =
            make_cuFloatComplex(1, 0);
    }
}

#ifdef USCT_TRSM_INVERSE
void __global__ initializePivotedIdentity(cuComplex * const matrix,
                                           int64_t const * const pivots,
                                           int const size,
                                           int const leadingDimension)
{
    extern __shared__ int64_t permutation[];
    for (int index = threadIdx.x; index < size; index += blockDim.x) {
        permutation[index] = index;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int index = 0; index < size; ++index) {
            int const pivot = static_cast<int>(pivots[index] - 1);
            int64_t const temporary = permutation[index];
            permutation[index] = permutation[pivot];
            permutation[pivot] = temporary;
        }
    }
    __syncthreads();
    for (int row = threadIdx.x; row < size; row += blockDim.x) {
        matrix[row + leadingDimension * permutation[row]] =
            make_cuFloatComplex(1, 0);
    }
}

#ifdef USCT_CUBLAS_GETRF
void __global__ initializePivotedIdentity32(cuComplex * const matrix,
                                             int const * const pivots,
                                             int const size,
                                             int const leadingDimension)
{
    extern __shared__ int permutation32[];
    for (int index = threadIdx.x; index < size; index += blockDim.x) {
        permutation32[index] = index;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int index = 0; index < size; ++index) {
            int const pivot = pivots[index] - 1;
            int const temporary = permutation32[index];
            permutation32[index] = permutation32[pivot];
            permutation32[pivot] = temporary;
        }
    }
    __syncthreads();
    for (int row = threadIdx.x; row < size; row += blockDim.x) {
        matrix[row + leadingDimension * permutation32[row]] =
            make_cuFloatComplex(1, 0);
    }
}
#endif
#endif



/*
 * Host code
 */
#ifndef USCT_FACTORS_BACKEND
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
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
        mexErrMsgIdAndTxt("usct:blocklu:InvalidOutput", 
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
        mexErrMsgIdAndTxt("parallel:gpu:triDiagMultLeftGPUmex:InvalidInput", 
                          "Invalid input to triDiagMultLeftGPUmex: Expecting 9 gpuArray inputs");
    }

    /* Assemble GPU Arrays from Inputs */
    Ld = mxGPUCreateFromMxArray(prhs[0]);
    Ll = mxGPUCreateFromMxArray(prhs[1]);
    Lu = mxGPUCreateFromMxArray(prhs[2]);
    Dd = mxGPUCreateFromMxArray(prhs[3]);
    Dl = mxGPUCreateFromMxArray(prhs[4]);
    Du = mxGPUCreateFromMxArray(prhs[5]);
    Ud = mxGPUCreateFromMxArray(prhs[6]);
    Ul = mxGPUCreateFromMxArray(prhs[7]);
    Uu = mxGPUCreateFromMxArray(prhs[8]);

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
        // Print Statements for Nx and Ny
        printf("Nx = %d\n", Nx);
        printf("Ny = %d\n", Ny);
        // Print Statements for Ld, Ll, and Lu
        printf("Number of Rows in Ld = %d\n", (int)mxGPUGetDimensions(Ld)[0]);
        printf("Number of Columns in Ld = %d\n", (int)mxGPUGetDimensions(Ld)[1]);
        printf("Number of Rows in Ll = %d\n", (int)mxGPUGetDimensions(Ll)[0]);
        printf("Number of Columns in Ll = %d\n", (int)mxGPUGetDimensions(Ll)[1]);
        printf("Number of Rows in Lu = %d\n", (int)mxGPUGetDimensions(Lu)[0]);
        printf("Number of Columns in Lu = %d\n", (int)mxGPUGetDimensions(Lu)[1]);
        // Print Statements for Dd, Dl, and Du
        printf("Number of Rows in Dd = %d\n", (int)mxGPUGetDimensions(Dd)[0]);
        printf("Number of Columns in Dd = %d\n", (int)mxGPUGetDimensions(Dd)[1]);
        printf("Number of Rows in Dl = %d\n", (int)mxGPUGetDimensions(Dl)[0]);
        printf("Number of Columns in Dl = %d\n", (int)mxGPUGetDimensions(Dl)[1]);
        printf("Number of Rows in Du = %d\n", (int)mxGPUGetDimensions(Du)[0]);
        printf("Number of Columns in Du = %d\n", (int)mxGPUGetDimensions(Du)[1]);
        // Print Statements for Ud, Ul, and Uu
        printf("Number of Rows in Ud = %d\n", (int)mxGPUGetDimensions(Ud)[0]);
        printf("Number of Columns in Ud = %d\n", (int)mxGPUGetDimensions(Ud)[1]);
        printf("Number of Rows in Ul = %d\n", (int)mxGPUGetDimensions(Ul)[0]);
        printf("Number of Columns in Ul = %d\n", (int)mxGPUGetDimensions(Ul)[1]);
        printf("Number of Rows in Uu = %d\n", (int)mxGPUGetDimensions(Uu)[0]);
        printf("Number of Columns in Uu = %d\n", (int)mxGPUGetDimensions(Uu)[1]);
        // Throw Error for Incorrect Array Sizes
        mexErrMsgIdAndTxt("parallel:gpu:triDiagMultLeftGPUmex:InvalidInput", 
                          "Invalid input to triDiagMultLeftGPUmex: Input sizes incorrect");
    }

    /* Verify that A really is a single array before extracting the pointer. */
    if (mxGPUGetClassID(Ld) != mxSINGLE_CLASS || 
        mxGPUGetClassID(Ll) != mxSINGLE_CLASS || 
        mxGPUGetClassID(Lu) != mxSINGLE_CLASS ||
        mxGPUGetClassID(Dd) != mxSINGLE_CLASS || 
        mxGPUGetClassID(Dl) != mxSINGLE_CLASS || 
        mxGPUGetClassID(Du) != mxSINGLE_CLASS ||  
        mxGPUGetClassID(Ud) != mxSINGLE_CLASS || 
        mxGPUGetClassID(Ul) != mxSINGLE_CLASS || 
        mxGPUGetClassID(Uu) != mxSINGLE_CLASS) {
        mexErrMsgIdAndTxt("parallel:gpu:triDiagMultLeftGPUmex:InvalidInput", 
                          "Invalid input to triDiagMultLeftGPUmex: Inputs should all be single precision");
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
    T = mxGPUCreateGPUArray(2, dims_T,
                            mxGPUGetClassID(Dd),
                            mxGPUGetComplexity(Dd),
                            MX_GPU_DO_NOT_INITIALIZE);
    d_T = (cuComplex *)(mxGPUGetData(T));
    /* Create a GPUArray to hold the final result and get its underlying pointer. */
    mwSize dims_invT[3] = {(mwSize) Ny, (mwSize) Ny, (mwSize) Nx};
    invT = mxGPUCreateGPUArray(3, dims_invT,
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
    USCT_CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
#ifdef USCT_TRSM_INVERSE
    cublasHandle_t cublasH;
    USCT_CUBLAS_CHECK(cublasCreate(&cublasH));
    cuComplex const one = make_cuFloatComplex(1, 0);
#endif
#ifdef USCT_GENERIC_GETRF
    cusolverDnParams_t solverParams;
    USCT_CUSOLVER_CHECK(cusolverDnCreateParams(&solverParams));
#ifdef USCT_GENERIC_ALG1
    USCT_CUSOLVER_CHECK(cusolverDnSetAdvOptions(
        solverParams, CUSOLVERDN_GETRF, CUSOLVER_ALG_1));
#endif
    size_t deviceWorkspaceBytes = 0;
    size_t hostWorkspaceBytes = 0;
    USCT_CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(
        cusolverH, solverParams, Ny-2, Ny-2, CUDA_C_32F,
        &d_T[Ny+1], Ny, CUDA_C_32F,
        &deviceWorkspaceBytes, &hostWorkspaceBytes));
    mwSize mwSizeLwork = static_cast<mwSize>(
        (deviceWorkspaceBytes + sizeof(cuComplex) - 1) / sizeof(cuComplex));
#else
    int Lwork = 0; // size of work space: determine size of work space on next line
    USCT_CUSOLVER_CHECK(cusolverDnCgetrf_bufferSize(cusolverH, Ny-2, Ny-2, 
                                &d_T[Ny+1], Ny, &Lwork));
    mwSize mwSizeLwork = (mwSize) Lwork;
#endif

    /* Prepare additional inputs for cuSOLVER: Workspace, devIpiv, devInfo */
    mxGPUArray * workspace = mxGPUCreateGPUArray(1, &mwSizeLwork,
                                                 mxGPUGetClassID(Dd),
                                                 mxGPUGetComplexity(Dd),
                                                 MX_GPU_DO_NOT_INITIALIZE);
    void * d_work = mxGPUGetData(workspace);
    mwSize N = (mwSize) Ny;
#ifdef USCT_GENERIC_GETRF
    mxGPUArray * Ipiv = mxGPUCreateGPUArray(1, &N,
                                            mxINT64_CLASS, mxREAL,
                                            MX_GPU_DO_NOT_INITIALIZE);
    int64_t * d_Ipiv = (int64_t *)(mxGPUGetData(Ipiv));
    void * h_work = hostWorkspaceBytes > 0 ? mxMalloc(hostWorkspaceBytes) : nullptr;
#ifdef USCT_CUBLAS_GETRF
    mxGPUArray *Ipiv32 = mxGPUCreateGPUArray(1, &N,
                                             mxINT32_CLASS, mxREAL,
                                             MX_GPU_DO_NOT_INITIALIZE);
    int *d_Ipiv32 = (int *)(mxGPUGetData(Ipiv32));
    cuComplex **d_matrixPointer = nullptr;
    USCT_CUDA_CHECK(cudaMalloc((void **)&d_matrixPointer, sizeof(cuComplex *)));
    cuComplex *h_matrixPointer = &d_T[Ny+1];
    USCT_CUDA_CHECK(cudaMemcpy(d_matrixPointer, &h_matrixPointer,
        sizeof(cuComplex *), cudaMemcpyHostToDevice));
#endif
#else
    mxGPUArray * Ipiv = mxGPUCreateGPUArray(1, &N,
                                            mxINT32_CLASS, mxREAL,
                                            MX_GPU_DO_NOT_INITIALIZE);
    int * d_Ipiv = (int *)(mxGPUGetData(Ipiv)); // Assuming int is 32 bit
#endif
    mwSize infoCount = (mwSize) Nx;
    mxGPUArray * factorInfo = mxGPUCreateGPUArray(1, &infoCount,
                                            mxINT32_CLASS, mxREAL,
                                            MX_GPU_INITIALIZE_VALUES);
    mxGPUArray * solveInfo = mxGPUCreateGPUArray(1, &infoCount,
                                            mxINT32_CLASS, mxREAL,
                                            MX_GPU_INITIALIZE_VALUES);
    int * d_factorInfo = (int *)(mxGPUGetData(factorInfo));
    int * d_solveInfo = (int *)(mxGPUGetData(solveInfo));

    std::vector<cudaEvent_t> getrfStart(Nx), getrfEnd(Nx);
    std::vector<cudaEvent_t> getrsStart(Nx), getrsEnd(Nx);
    std::vector<cudaEvent_t> schurStart(Nx-1), schurEnd(Nx-1);
    cudaEvent_t totalStart = nullptr;
    cudaEvent_t totalEnd = nullptr;
    if (collectTimings) {
        for (int page = 0; page < Nx; ++page) {
            USCT_CUDA_CHECK(cudaEventCreate(&getrfStart[page]));
            USCT_CUDA_CHECK(cudaEventCreate(&getrfEnd[page]));
            USCT_CUDA_CHECK(cudaEventCreate(&getrsStart[page]));
            USCT_CUDA_CHECK(cudaEventCreate(&getrsEnd[page]));
            if (page < Nx - 1) {
                USCT_CUDA_CHECK(cudaEventCreate(&schurStart[page]));
                USCT_CUDA_CHECK(cudaEventCreate(&schurEnd[page]));
            }
        }
        USCT_CUDA_CHECK(cudaEventCreate(&totalStart));
        USCT_CUDA_CHECK(cudaEventCreate(&totalEnd));
        USCT_CUDA_CHECK(cudaEventRecord(totalStart));
    }

    /* Initialize all inverse pages once, then run the serial recurrence. */
    int const identityThreads = 256;
#if !defined(USCT_TRSM_INVERSE) || defined(USCT_NO_PIVOT_GETRF)
    int const identityCount = (Ny - 2) * Nx;
    initializeIdentityPages<<<(identityCount + identityThreads - 1) / identityThreads,
                              identityThreads>>>(d_invT, Ny, Nx);
    USCT_KERNEL_CHECK();
#endif

    /* Calculate the Schur Complements and Their Inverses */
    triDiagMat<<<dimGridBP, dimBlockBP>>>(d_Dd, d_Dl, d_Du, d_T, Ny);  
    USCT_KERNEL_CHECK();
    size_t const schurSharedBytes =
        (dimBlockBP.x + 2) * (dimBlockBP.y + 2) * sizeof(cuComplex);
    for (int x_idx = 1; x_idx < Nx; x_idx++) {
        /* Factor T in place and solve against the preinitialized identity. */
        if (collectTimings) {
            USCT_CUDA_CHECK(cudaEventRecord(getrfStart[x_idx-1]));
        }
#ifdef USCT_GENERIC_GETRF
#ifdef USCT_CUBLAS_GETRF
        USCT_CUBLAS_CHECK(cublasCgetrfBatched(cublasH, Ny-2,
            d_matrixPointer, Ny, d_Ipiv32, &d_factorInfo[x_idx-1], 1));
#else
#ifdef USCT_NO_PIVOT_GETRF
        int64_t *factorPivots = nullptr;
#else
        int64_t *factorPivots = d_Ipiv;
#endif
        USCT_CUSOLVER_CHECK(cusolverDnXgetrf(
            cusolverH, solverParams, Ny-2, Ny-2, CUDA_C_32F,
            &d_T[Ny+1], Ny, factorPivots, CUDA_C_32F,
            d_work, deviceWorkspaceBytes, h_work, hostWorkspaceBytes,
            &d_factorInfo[x_idx-1]));
#endif
        if (collectTimings) {
            USCT_CUDA_CHECK(cudaEventRecord(getrfEnd[x_idx-1]));
            USCT_CUDA_CHECK(cudaEventRecord(getrsStart[x_idx-1]));
        }
#ifdef USCT_TRSM_INVERSE
#ifndef USCT_NO_PIVOT_GETRF
#ifdef USCT_CUBLAS_GETRF
        initializePivotedIdentity32<<<1, identityThreads,
            static_cast<size_t>(Ny-2) * sizeof(int)>>>(
            &d_invT[Ny*Ny*(x_idx-1)+Ny+1], d_Ipiv32, Ny-2, Ny);
#else
        initializePivotedIdentity<<<1, identityThreads,
            static_cast<size_t>(Ny-2) * sizeof(int64_t)>>>(
            &d_invT[Ny*Ny*(x_idx-1)+Ny+1], d_Ipiv, Ny-2, Ny);
#endif
        USCT_KERNEL_CHECK();
#endif
        USCT_CUBLAS_CHECK(cublasCtrsm(cublasH, CUBLAS_SIDE_LEFT,
            CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
            Ny-2, Ny-2, &one, &d_T[Ny+1], Ny,
            &d_invT[Ny*Ny*(x_idx-1)+Ny+1], Ny));
        USCT_CUBLAS_CHECK(cublasCtrsm(cublasH, CUBLAS_SIDE_LEFT,
            CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
            Ny-2, Ny-2, &one, &d_T[Ny+1], Ny,
            &d_invT[Ny*Ny*(x_idx-1)+Ny+1], Ny));
#else
        USCT_CUSOLVER_CHECK(cusolverDnXgetrs(
            cusolverH, solverParams, CUBLAS_OP_N, Ny-2, Ny-2,
            CUDA_C_32F, &d_T[Ny+1], Ny, d_Ipiv, CUDA_C_32F,
            &d_invT[Ny*Ny*(x_idx-1)+Ny+1], Ny,
            &d_solveInfo[x_idx-1]));
#endif
#else
        USCT_CUSOLVER_CHECK(cusolverDnCgetrf(cusolverH, Ny-2, Ny-2, 
                         &d_T[Ny+1], Ny, (cuComplex *)d_work, d_Ipiv,
                         &d_factorInfo[x_idx-1]));
        if (collectTimings) {
            USCT_CUDA_CHECK(cudaEventRecord(getrfEnd[x_idx-1]));
            USCT_CUDA_CHECK(cudaEventRecord(getrsStart[x_idx-1]));
        }
        USCT_CUSOLVER_CHECK(cusolverDnCgetrs(cusolverH, CUBLAS_OP_N, Ny-2, Ny-2, /* nrhs */
                         &d_T[Ny+1], Ny, d_Ipiv, 
                         &d_invT[Ny*Ny*(x_idx-1)+Ny+1], Ny, &d_solveInfo[x_idx-1]));
#endif
        if (collectTimings) {
            USCT_CUDA_CHECK(cudaEventRecord(getrsEnd[x_idx-1]));
            USCT_CUDA_CHECK(cudaEventRecord(schurStart[x_idx-1]));
        }
        schurUpdateFused<<<dimGridBP, dimBlockBP, schurSharedBytes>>>(
            &d_invT[Ny*Ny*(x_idx-1)],
            &d_Ld[Ny*(x_idx-1)], &d_Ll[(Ny-1)*(x_idx-1)],
            &d_Lu[(Ny-1)*(x_idx-1)], &d_Ud[Ny*(x_idx-1)],
            &d_Ul[(Ny-1)*(x_idx-1)], &d_Uu[(Ny-1)*(x_idx-1)],
            &d_Dd[Ny*x_idx], &d_Dl[(Ny-1)*x_idx],
            &d_Du[(Ny-1)*x_idx], d_T, Ny);
        USCT_KERNEL_CHECK();
        if (collectTimings) {
            USCT_CUDA_CHECK(cudaEventRecord(schurEnd[x_idx-1]));
        }
    }
    /* Run cuSOLVER to get inverses */
    if (collectTimings) {
        USCT_CUDA_CHECK(cudaEventRecord(getrfStart[Nx-1]));
    }
#ifdef USCT_GENERIC_GETRF
#ifdef USCT_CUBLAS_GETRF
    USCT_CUBLAS_CHECK(cublasCgetrfBatched(cublasH, Ny-2,
        d_matrixPointer, Ny, d_Ipiv32, &d_factorInfo[Nx-1], 1));
#else
#ifdef USCT_NO_PIVOT_GETRF
    int64_t *lastFactorPivots = nullptr;
#else
    int64_t *lastFactorPivots = d_Ipiv;
#endif
    USCT_CUSOLVER_CHECK(cusolverDnXgetrf(
        cusolverH, solverParams, Ny-2, Ny-2, CUDA_C_32F,
        &d_T[Ny+1], Ny, lastFactorPivots, CUDA_C_32F,
        d_work, deviceWorkspaceBytes, h_work, hostWorkspaceBytes,
        &d_factorInfo[Nx-1]));
#endif
    if (collectTimings) {
        USCT_CUDA_CHECK(cudaEventRecord(getrfEnd[Nx-1]));
        USCT_CUDA_CHECK(cudaEventRecord(getrsStart[Nx-1]));
    }
#ifdef USCT_TRSM_INVERSE
#ifndef USCT_NO_PIVOT_GETRF
#ifdef USCT_CUBLAS_GETRF
    initializePivotedIdentity32<<<1, identityThreads,
        static_cast<size_t>(Ny-2) * sizeof(int)>>>(
        &d_invT[Ny*Ny*(Nx-1)+Ny+1], d_Ipiv32, Ny-2, Ny);
#else
    initializePivotedIdentity<<<1, identityThreads,
        static_cast<size_t>(Ny-2) * sizeof(int64_t)>>>(
        &d_invT[Ny*Ny*(Nx-1)+Ny+1], d_Ipiv, Ny-2, Ny);
#endif
    USCT_KERNEL_CHECK();
#endif
    USCT_CUBLAS_CHECK(cublasCtrsm(cublasH, CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
        Ny-2, Ny-2, &one, &d_T[Ny+1], Ny,
        &d_invT[Ny*Ny*(Nx-1)+Ny+1], Ny));
    USCT_CUBLAS_CHECK(cublasCtrsm(cublasH, CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
        Ny-2, Ny-2, &one, &d_T[Ny+1], Ny,
        &d_invT[Ny*Ny*(Nx-1)+Ny+1], Ny));
#else
    USCT_CUSOLVER_CHECK(cusolverDnXgetrs(
        cusolverH, solverParams, CUBLAS_OP_N, Ny-2, Ny-2,
        CUDA_C_32F, &d_T[Ny+1], Ny, d_Ipiv, CUDA_C_32F,
        &d_invT[Ny*Ny*(Nx-1)+Ny+1], Ny, &d_solveInfo[Nx-1]));
#endif
#else
    USCT_CUSOLVER_CHECK(cusolverDnCgetrf(cusolverH, Ny-2, Ny-2, 
                     &d_T[Ny+1], Ny, (cuComplex *)d_work, d_Ipiv,
                     &d_factorInfo[Nx-1]));
    if (collectTimings) {
        USCT_CUDA_CHECK(cudaEventRecord(getrfEnd[Nx-1]));
        USCT_CUDA_CHECK(cudaEventRecord(getrsStart[Nx-1]));
    }
    USCT_CUSOLVER_CHECK(cusolverDnCgetrs(cusolverH, CUBLAS_OP_N, Ny-2, Ny-2, /* nrhs */
                     &d_T[Ny+1], Ny, d_Ipiv, 
                     &d_invT[Ny*Ny*(Nx-1)+Ny+1], Ny, &d_solveInfo[Nx-1]));
#endif
    if (collectTimings) {
        USCT_CUDA_CHECK(cudaEventRecord(getrsEnd[Nx-1]));
        USCT_CUDA_CHECK(cudaEventRecord(totalEnd));
    }

    usctCheckDeviceInfo(d_factorInfo, Nx, "getrf");
    usctCheckDeviceInfo(d_solveInfo, Nx, "getrs");

    /* Must destroy cuSOLVER handle at the end */
#ifdef USCT_GENERIC_GETRF
    if (h_work != nullptr) {
        mxFree(h_work);
    }
    USCT_CUSOLVER_CHECK(cusolverDnDestroyParams(solverParams));
#endif
#ifdef USCT_CUBLAS_GETRF
    USCT_CUDA_CHECK(cudaFree(d_matrixPointer));
    mxGPUDestroyGPUArray(Ipiv32);
#endif
#ifdef USCT_TRSM_INVERSE
    USCT_CUBLAS_CHECK(cublasDestroy(cublasH));
#endif
    USCT_CUSOLVER_CHECK(cusolverDnDestroy(cusolverH)); 

    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(invT);
    if (collectTimings) {
        USCT_CUDA_CHECK(cudaEventSynchronize(totalEnd));
        float getrfMilliseconds = 0;
        float getrsMilliseconds = 0;
        float schurMilliseconds = 0;
        float elapsed = 0;
        for (int page = 0; page < Nx; ++page) {
            USCT_CUDA_CHECK(cudaEventElapsedTime(
                &elapsed, getrfStart[page], getrfEnd[page]));
            getrfMilliseconds += elapsed;
            USCT_CUDA_CHECK(cudaEventElapsedTime(
                &elapsed, getrsStart[page], getrsEnd[page]));
            getrsMilliseconds += elapsed;
            USCT_CUDA_CHECK(cudaEventDestroy(getrfStart[page]));
            USCT_CUDA_CHECK(cudaEventDestroy(getrfEnd[page]));
            USCT_CUDA_CHECK(cudaEventDestroy(getrsStart[page]));
            USCT_CUDA_CHECK(cudaEventDestroy(getrsEnd[page]));
            if (page < Nx - 1) {
                USCT_CUDA_CHECK(cudaEventElapsedTime(
                    &elapsed, schurStart[page], schurEnd[page]));
                schurMilliseconds += elapsed;
                USCT_CUDA_CHECK(cudaEventDestroy(schurStart[page]));
                USCT_CUDA_CHECK(cudaEventDestroy(schurEnd[page]));
            }
        }
        float totalMilliseconds = 0;
        USCT_CUDA_CHECK(cudaEventElapsedTime(
            &totalMilliseconds, totalStart, totalEnd));
        USCT_CUDA_CHECK(cudaEventDestroy(totalStart));
        USCT_CUDA_CHECK(cudaEventDestroy(totalEnd));
        char const *fieldNames[] = {"getrf_ms", "inverse_solve_ms",
            "schur_update_ms", "other_ms", "total_gpu_ms", "block_count"};
        plhs[1] = mxCreateStructMatrix(1, 1, 6, fieldNames);
        mxSetField(plhs[1], 0, "getrf_ms", mxCreateDoubleScalar(getrfMilliseconds));
        mxSetField(plhs[1], 0, "inverse_solve_ms", mxCreateDoubleScalar(getrsMilliseconds));
        mxSetField(plhs[1], 0, "schur_update_ms", mxCreateDoubleScalar(schurMilliseconds));
        mxSetField(plhs[1], 0, "other_ms", mxCreateDoubleScalar(
            totalMilliseconds - getrfMilliseconds - getrsMilliseconds - schurMilliseconds));
        mxSetField(plhs[1], 0, "total_gpu_ms", mxCreateDoubleScalar(totalMilliseconds));
        mxSetField(plhs[1], 0, "block_count", mxCreateDoubleScalar(Nx));
    }

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(Ld);
    mxGPUDestroyGPUArray(Ll);
    mxGPUDestroyGPUArray(Lu);
    mxGPUDestroyGPUArray(Dd);
    mxGPUDestroyGPUArray(Dl);
    mxGPUDestroyGPUArray(Du);
    mxGPUDestroyGPUArray(Ud);
    mxGPUDestroyGPUArray(Ul);
    mxGPUDestroyGPUArray(Uu);
    mxGPUDestroyGPUArray(T);
    mxGPUDestroyGPUArray(workspace);
    mxGPUDestroyGPUArray(Ipiv);
    mxGPUDestroyGPUArray(factorInfo);
    mxGPUDestroyGPUArray(solveInfo);
    mxGPUDestroyGPUArray(invT);
}
#else

void __global__ zeroMatrixBoundary(cuComplex *A, int N)
{
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N &&
        (row == 0 || row == static_cast<unsigned int>(N - 1) ||
         col == 0 || col == static_cast<unsigned int>(N - 1))) {
        A[row + N * col] = make_cuFloatComplex(0, 0);
    }
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    mxInitGPU();
    if (nlhs != 2) {
        mexErrMsgIdAndTxt("usct:blocklu:InvalidOutput",
            "decompBlockLUFactors requires [luFactors, pivots].");
    }
    if (nrhs != 9) {
        mexErrMsgIdAndTxt("usct:blocklu:InvalidInput",
            "decompBlockLUFactors expects nine gpuArray inputs.");
    }
    for (int input = 0; input < 9; ++input) {
        if (!mxIsGPUArray(prhs[input])) {
            mexErrMsgIdAndTxt("usct:blocklu:InvalidInput",
                "All decompBlockLUFactors inputs must be gpuArray values.");
        }
    }

    mxGPUArray const *Ld = mxGPUCreateFromMxArray(prhs[0]);
    mxGPUArray const *Ll = mxGPUCreateFromMxArray(prhs[1]);
    mxGPUArray const *Lu = mxGPUCreateFromMxArray(prhs[2]);
    mxGPUArray const *Dd = mxGPUCreateFromMxArray(prhs[3]);
    mxGPUArray const *Dl = mxGPUCreateFromMxArray(prhs[4]);
    mxGPUArray const *Du = mxGPUCreateFromMxArray(prhs[5]);
    mxGPUArray const *Ud = mxGPUCreateFromMxArray(prhs[6]);
    mxGPUArray const *Ul = mxGPUCreateFromMxArray(prhs[7]);
    mxGPUArray const *Uu = mxGPUCreateFromMxArray(prhs[8]);

    int Ny = static_cast<int>(mxGPUGetDimensions(Dd)[0]);
    int Nx = static_cast<int>(mxGPUGetDimensions(Dd)[1]);
    int interior = Ny - 2;
    if (Nx < 3 || Ny < 3) {
        mexErrMsgIdAndTxt("usct:blocklu:InvalidGrid",
            "Block-LU requires Nx and Ny to be at least three.");
    }
    if (mxGPUGetDimensions(Ld)[0] != Ny || mxGPUGetDimensions(Ld)[1] != Nx-1 ||
        mxGPUGetDimensions(Ll)[0] != Ny-1 || mxGPUGetDimensions(Ll)[1] != Nx-1 ||
        mxGPUGetDimensions(Lu)[0] != Ny-1 || mxGPUGetDimensions(Lu)[1] != Nx-1 ||
        mxGPUGetDimensions(Dl)[0] != Ny-1 || mxGPUGetDimensions(Dl)[1] != Nx ||
        mxGPUGetDimensions(Du)[0] != Ny-1 || mxGPUGetDimensions(Du)[1] != Nx ||
        mxGPUGetDimensions(Ud)[0] != Ny || mxGPUGetDimensions(Ud)[1] != Nx-1 ||
        mxGPUGetDimensions(Ul)[0] != Ny-1 || mxGPUGetDimensions(Ul)[1] != Nx-1 ||
        mxGPUGetDimensions(Uu)[0] != Ny-1 || mxGPUGetDimensions(Uu)[1] != Nx-1) {
        mexErrMsgIdAndTxt("usct:blocklu:InvalidDimensions",
            "Block diagonal arrays have inconsistent dimensions.");
    }
    mxGPUArray const *inputs[] = {Ld,Ll,Lu,Dd,Dl,Du,Ud,Ul,Uu};
    for (int input = 0; input < 9; ++input) {
        if (mxGPUGetClassID(inputs[input]) != mxSINGLE_CLASS ||
            mxGPUGetComplexity(inputs[input]) != mxCOMPLEX) {
            mexErrMsgIdAndTxt("usct:blocklu:InvalidType",
                "All block arrays must be complex single gpuArray values.");
        }
    }

    cuComplex const *d_Ld = reinterpret_cast<cuComplex const *>(mxGPUGetDataReadOnly(Ld));
    cuComplex const *d_Ll = reinterpret_cast<cuComplex const *>(mxGPUGetDataReadOnly(Ll));
    cuComplex const *d_Lu = reinterpret_cast<cuComplex const *>(mxGPUGetDataReadOnly(Lu));
    cuComplex const *d_Dd = reinterpret_cast<cuComplex const *>(mxGPUGetDataReadOnly(Dd));
    cuComplex const *d_Dl = reinterpret_cast<cuComplex const *>(mxGPUGetDataReadOnly(Dl));
    cuComplex const *d_Du = reinterpret_cast<cuComplex const *>(mxGPUGetDataReadOnly(Du));
    cuComplex const *d_Ud = reinterpret_cast<cuComplex const *>(mxGPUGetDataReadOnly(Ud));
    cuComplex const *d_Ul = reinterpret_cast<cuComplex const *>(mxGPUGetDataReadOnly(Ul));
    cuComplex const *d_Uu = reinterpret_cast<cuComplex const *>(mxGPUGetDataReadOnly(Uu));

    mwSize matrixDims[2] = {static_cast<mwSize>(Ny), static_cast<mwSize>(Ny)};
    mxGPUArray *T = mxGPUCreateGPUArray(2, matrixDims, mxSINGLE_CLASS,
        mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArray *solveScratch = mxGPUCreateGPUArray(2, matrixDims,
        mxSINGLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    cuComplex *d_T = reinterpret_cast<cuComplex *>(mxGPUGetData(T));
    cuComplex *d_solveScratch = reinterpret_cast<cuComplex *>(mxGPUGetData(solveScratch));

    mwSize factorDims[3] = {static_cast<mwSize>(Ny), static_cast<mwSize>(Ny),
        static_cast<mwSize>(Nx)};
    mxGPUArray *luFactors = mxGPUCreateGPUArray(3, factorDims,
        mxSINGLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    cuComplex *d_luFactors = reinterpret_cast<cuComplex *>(mxGPUGetData(luFactors));
    mwSize pivotDims[2] = {static_cast<mwSize>(interior), static_cast<mwSize>(Nx)};
    mxGPUArray *pivots = mxGPUCreateGPUArray(2, pivotDims,
        mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int *d_pivots = reinterpret_cast<int *>(mxGPUGetData(pivots));

    cusolverDnHandle_t handle;
    USCT_CUSOLVER_CHECK(cusolverDnCreate(&handle));
    int workspaceSize = 0;
    USCT_CUSOLVER_CHECK(cusolverDnCgetrf_bufferSize(handle, interior, interior,
        &d_luFactors[Ny + 1], Ny, &workspaceSize));
    mwSize workspaceElements = static_cast<mwSize>(workspaceSize);
    mxGPUArray *workspace = mxGPUCreateGPUArray(1, &workspaceElements,
        mxSINGLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    cuComplex *d_workspace = reinterpret_cast<cuComplex *>(mxGPUGetData(workspace));
    mwSize infoElements = static_cast<mwSize>(Nx);
    mxGPUArray *factorInfo = mxGPUCreateGPUArray(1, &infoElements,
        mxINT32_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    mxGPUArray *solveInfo = mxGPUCreateGPUArray(1, &infoElements,
        mxINT32_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    int *d_factorInfo = reinterpret_cast<int *>(mxGPUGetData(factorInfo));
    int *d_solveInfo = reinterpret_cast<int *>(mxGPUGetData(solveInfo));

    dim3 block(threadsPerBlockX, threadsPerBlockY, 1);
    dim3 grid((Ny + block.x - 1) / block.x,
              (Ny + block.y - 1) / block.y, 1);
    triDiagMat<<<grid, block>>>(d_Dd, d_Dl, d_Du, d_T, Ny);
    USCT_KERNEL_CHECK();
    for (int page = 0; page < Nx; ++page) {
        cuComplex *factorPage = &d_luFactors[static_cast<size_t>(Ny) * Ny * page];
        copyInterior<<<grid, block>>>(d_T, factorPage, Ny);
        USCT_KERNEL_CHECK();
        USCT_CUSOLVER_CHECK(cusolverDnCgetrf(handle, interior, interior,
            &factorPage[Ny + 1], Ny, d_workspace,
            &d_pivots[interior * page], &d_factorInfo[page]));
        if (page < Nx - 1) {
            triDiagMat<<<grid, block>>>(
                &d_Ud[Ny * page], &d_Ul[(Ny - 1) * page],
                &d_Uu[(Ny - 1) * page], d_solveScratch, Ny);
            USCT_KERNEL_CHECK();
            USCT_CUSOLVER_CHECK(cusolverDnCgetrs(handle, CUBLAS_OP_N,
                interior, interior, &factorPage[Ny + 1], Ny,
                &d_pivots[interior * page], &d_solveScratch[Ny + 1], Ny,
                &d_solveInfo[page]));
            zeroMatrixBoundary<<<grid, block>>>(d_solveScratch, Ny);
            USCT_KERNEL_CHECK();
            triDiagMultLeftPlusD<<<grid, block>>>(d_solveScratch,
                &d_Ld[Ny * page], &d_Ll[(Ny - 1) * page],
                &d_Lu[(Ny - 1) * page], &d_Dd[Ny * (page + 1)],
                &d_Dl[(Ny - 1) * (page + 1)],
                &d_Du[(Ny - 1) * (page + 1)], d_T, Ny);
            USCT_KERNEL_CHECK();
        }
    }
    usctCheckDeviceInfo(d_factorInfo, Nx, "getrf");
    usctCheckDeviceInfo(d_solveInfo, Nx, "Schur getrs");
    USCT_CUSOLVER_CHECK(cusolverDnDestroy(handle));

    plhs[0] = mxGPUCreateMxArrayOnGPU(luFactors);
    plhs[1] = mxGPUCreateMxArrayOnGPU(pivots);
    for (int input = 0; input < 9; ++input) {
        mxGPUDestroyGPUArray(inputs[input]);
    }
    mxGPUDestroyGPUArray(T);
    mxGPUDestroyGPUArray(solveScratch);
    mxGPUDestroyGPUArray(workspace);
    mxGPUDestroyGPUArray(factorInfo);
    mxGPUDestroyGPUArray(solveInfo);
    mxGPUDestroyGPUArray(luFactors);
    mxGPUDestroyGPUArray(pivots);
}
#endif
