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
    // int M = N, K = N;
    // extern __shared__ _FTYPE_ As[TILEDIM_M][TILEDIM_K], Bs[TILEDIM_K][TILEDIM_N];
    extern __shared__ _FTYPE_ smem[];

    _FTYPE_ (*As)[TILEDIM_K] =
        reinterpret_cast<_FTYPE_ (*)[TILEDIM_K]>(smem);

    _FTYPE_ (*Bs)[TILEDIM_N] =
        reinterpret_cast<_FTYPE_ (*)[TILEDIM_N]>(
            smem + TILEDIM_M * TILEDIM_K
        );

    int ty = threadIdx.y, tx = threadIdx.x;
    int by = blockIdx.y, bx = blockIdx.x;
    int tthreads = blockDim.x * blockDim.y;
    int tid = ty * blockDim.x + tx;
    int I_base = by * TILEDIM_M, J_base = bx*TILEDIM_N;
    // Multiple but independent output accumulators for ILP.
    _FTYPE_ _Cij[TILESCALE_M][TILESCALE_N] = {};

    for(int kk=0; kk < (N+TILEDIM_K-1)/TILEDIM_K; kk++){
    
        // let's cooperatively load the tiles of A and B into shared memory.
        // let's write loop method to load A.
        for(int p=tid; p < TILEDIM_M * TILEDIM_K; p += tthreads){
            int i = p / TILEDIM_K;
            int j = p % TILEDIM_K;
            As[i][j] = (I_base + i < N && kk*TILEDIM_K + j < N) ? A[(I_base + i) * N + kk*TILEDIM_K + j] : 0;
        }
        // let's write loop method to load B.
        for(int p=tid; p < TILEDIM_K * TILEDIM_N; p += tthreads){
            int i = p / TILEDIM_N;
            int j = p % TILEDIM_N;
            Bs[i][j] = (kk*TILEDIM_K + i < N && J_base + j < N) ? B[(kk*TILEDIM_K + i) * N + J_base + j] : 0;
        }
        __syncthreads();

        // once all the threads have loaded the tiles into shared memory, we can start the computation.
        int kLimit = min(TILEDIM_K, N - kk * TILEDIM_K);

        for (int k0 = 0; k0 < kLimit; k0+=TILESCALE_K) {
            _FTYPE_ a_reg[TILESCALE_K][TILESCALE_M];
            _FTYPE_ b_reg[TILESCALE_K][TILESCALE_N];
            #pragma unroll
            for (int q = 0; q < TILESCALE_K; q++) {

                int k = k0 + q;
                // if (k >= kLimit) continue;
                

                #pragma unroll
                for (int i = 0; i < TILESCALE_M; i++) {
                    a_reg[q][i] = As[ty + i * blockDim.y][k];
                }

                #pragma unroll
                for (int j = 0; j < TILESCALE_N; j++) {
                    b_reg[q][j] = Bs[k][tx + j * blockDim.x];
                }
            }

            #pragma unroll
            for (int q = 0; q < TILESCALE_K; q++) {
                #pragma unroll
                for (int i = 0; i < TILESCALE_M; i++) {
                    #pragma unroll
                    for (int j = 0; j < TILESCALE_N; j++) {
                        _Cij[i][j] += a_reg[q][i] * b_reg[q][j];
                    }
                }
            }
        }
        
        // for(int k=0; k<kLimit; k++){
            
        //     #pragma unroll
        //     for(int i=0; i<TILESCALE_M; i++){
        //         #pragma unroll
        //         for(int j=0; j<TILESCALE_N; j++){
        //             _Cij[i][j] += As[ty + i * blockDim.y][k] * Bs[k][tx + blockDim.x * j];
        //         }
        //     }
        // }
        __syncthreads();

    }

    // Now store multiple CTA outputs.
    #pragma unroll
    for(int i=0; i<TILESCALE_M; i++){
        #pragma unroll
        for(int j=0; j<TILESCALE_N; j++){
            int I = I_base + ty + i * blockDim.y;
            int J = J_base + tx + j * blockDim.x;
            if(I < N && J < N){
                C[I * N + J] = _Cij[i][j];
            }
        }
    }
    // #pragma unroll
    // for (int i = 0; i < TILESCALE_M; i++) {
    //     #pragma unroll
    //     for (int j = 0; j < TILESCALE_N; j++) {
    //         _FTYPE_ result = 0;
    //         #pragma unroll
    //         for (int q = 0; q < TILESCALE_K; q++) {
    //             result += _Cij[q][i][j];
    //         }
    //         int I = I_base + ty + i * blockDim.y;
    //         int J = J_base + tx + j * blockDim.x;
    //         if (I < N && J < N) {
    //             C[I * N + J] = result;
    //         }
    //     }
    // }

    
    // for(int i=0; i<TILESCALE_M; i++){}


    // for(int kk=0;kk<(N +TILEDIM_K-1)/TILEDIM_K;kk++){
    //     int kLimit = min(TILEDIM_K, N - kk*TILEDIM_K);
    //     As[ty][tx] = (I < N && kk*TILEDIM_K + tx < N) ? A[I*N + kk*TILEDIM_K + tx] : 0;
    //     Bs[ty][tx] = (kk*TILEDIM_K + ty < N && J < N) ? B[(kk*TILEDIM_K + ty) * N + J] : 0;
    //     __syncthreads();
        
    //     for(int k=0; k<kLimit; k++){
    //         _Cij += As[ty][k] * Bs[k][tx];
    //     }
        
    //     __syncthreads();
    // }
    
    // if (I < N && J < N){
    //     C[I*N  + J] = _Cij;
    // }
    //     C[I*N  + J] = _Cij;
}
#endif


// for ILP, we will need to see TileScale and TileStep in mytypes.h. We are going to assume TileStep to be 1 only.
