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

    int I =  blockIdx.y*BLOCKDIM_Y + threadIdx.y;
    int J =  blockIdx.x*BLOCKDIM_X + threadIdx.x;

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
    constexpr int TTHREADS = BLOCKDIM_X * BLOCKDIM_Y;
    constexpr int A_TILE_ELEMENTS = TILEDIM_M * TILEDIM_K;
    constexpr int B_TILE_ELEMENTS = TILEDIM_K * TILEDIM_N;


    extern __shared__ _FTYPE_ smem[];

    _FTYPE_ (*As)[TILEDIM_K] =
        reinterpret_cast<_FTYPE_ (*)[TILEDIM_K]>(smem);

    _FTYPE_ (*Bs)[TILEDIM_N] =
        reinterpret_cast<_FTYPE_ (*)[TILEDIM_N]>(
            smem + TILEDIM_M * TILEDIM_K
        );

    int ty = threadIdx.y, tx = threadIdx.x;
    int by = blockIdx.y, bx = blockIdx.x;
    // int tthreads = BLOCKDIM_X * BLOCKDIM_Y;
    int tid = ty * BLOCKDIM_X + tx;
    int I_base = by * TILEDIM_M, J_base = bx*TILEDIM_N;
    // Multiple but independent output accumulators for ILP.
    _FTYPE_ _Cij[TILESCALE_M][TILESCALE_N] = {};

    for(int kk=0; kk < (N+TILEDIM_K-1)/TILEDIM_K; kk++){
    
        // let's cooperatively load the tiles of A and B into shared memory.
        // let's write loop method to load A.
        for(int p=tid; p < A_TILE_ELEMENTS; p += TTHREADS){
            int i = p / TILEDIM_K;
            int j = p % TILEDIM_K;
            As[i][j] = (I_base + i < N && kk*TILEDIM_K + j < N) ? A[(I_base + i) * N + kk*TILEDIM_K + j] : 0;
        }
        // let's write loop method to load B.
        for(int p=tid; p < B_TILE_ELEMENTS; p += TTHREADS){
            int i = p / TILEDIM_N;
            int j = p % TILEDIM_N;
            Bs[i][j] = (kk*TILEDIM_K + i < N && J_base + j < N) ? B[(kk*TILEDIM_K + i) * N + J_base + j] : 0;
        }
        __syncthreads();

        // once all the threads have loaded the tiles into shared memory, we can start the computation.
        int kLimit = min(TILEDIM_K, N - kk * TILEDIM_K);

        for (int k0 = 0; k0 < kLimit; k0+=TILESCALE_K) {
            // _FTYPE_ a_reg[TILESCALE_K][TILESCALE_M];
            // _FTYPE_ b_reg[TILESCALE_K][TILESCALE_N];

            #pragma unroll
            for (int q = 0; q < TILESCALE_K; q++) {
                int k = k0 + q;

                _FTYPE_ a_reg[TILESCALE_M];
                _FTYPE_ b_reg[TILESCALE_N];

                #pragma unroll
                for (int i = 0; i < TILESCALE_M; i++) {
                    a_reg[i] = As[ty + i * BLOCKDIM_Y][k];
                }

                #pragma unroll
                for (int j = 0; j < TILESCALE_N; j++) {
                    b_reg[j] = Bs[k][tx + j * BLOCKDIM_X];
                }

                #pragma unroll
                for (int i = 0; i < TILESCALE_M; i++) {
                    #pragma unroll
                    for (int j = 0; j < TILESCALE_N; j++) {
                        _Cij[i][j] += a_reg[i] * b_reg[j];
                    }
                }
            }
        }
        
        __syncthreads();

    }

    // Now store multiple CTA outputs.
    #pragma unroll
    for(int i=0; i<TILESCALE_M; i++){
        #pragma unroll
        for(int j=0; j<TILESCALE_N; j++){
            int I = I_base + ty + i * BLOCKDIM_Y;
            int J = J_base + tx + j * BLOCKDIM_X;
            if(I < N && J < N){
                C[I * N + J] = _Cij[i][j];
            }
        }
    }
}
#endif


// for ILP, we will need to see TileScale and TileStep in mytypes.h. We are going to assume TileStep to be 1 only.
