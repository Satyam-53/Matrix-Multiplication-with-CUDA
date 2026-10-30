// ;-*- mode: c;-*-
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"
using namespace std;

#include <stdio.h>

#ifdef NAIVE
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {

    int I =  blockIdx.y*blockDim.y + threadIdx.y;
    int J =  blockIdx.x*blockDim.x + threadIdx.x;

    if((I < N) && (J < N)){
        _FTYPE_ _c = 0;
        for (unsigned int k = 0; k < N; k++) {
            _FTYPE_ a = A[I * N + k];
            _FTYPE_ b = B[k * N + J];
            _c += a * b;
        }
        C[I * N + J] = _c;
    }
}

#else
//You should be changing the kernel here for the non naive implementation.
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {
    int M = N, K = N;
    __shared__ _FTYPE_ As[TILEDIM_M][TILEDIM_K], Bs[TILEDIM_K][TILEDIM_N];
    int ty = threadIdx.y, tx = threadIdx.x;
    int by = blockIdx.y, bx = blockIdx.x;
    int I = by * TILEDIM_M + ty, J = bx*TILEDIM_N + tx;
    _FTYPE_ _Cij = 0;

    for(int kk=0;kk<K/TILEDIM_K;kk++){

        As[ty][tx] = A[I*K + kk*TILEDIM_K + tx];
        Bs[ty][tx] = B[(kk*TILEDIM_K + ty) * N + J];
        __syncthreads();
        
        for(int k=0; k<TILEDIM_K; k++){
            _Cij += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    C[I*N  + J] = _Cij;
}
#endif
