#pragma once
// SGEMM kernels. All matrices are row-major:
//   C[M x N] = alpha * A[M x K] * B[K x N] + beta * C[M x N]

#include <cuda_runtime.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// ---------------------------------------------------------------------------
// Kernel 1: naive.
// threadIdx.x indexes the *row* of C, so consecutive threads in a warp read
// A with a stride of K floats and write C with a stride of N floats. Neither
// access is coalesced: every warp-wide load touches 32 separate 32B sectors.
// ---------------------------------------------------------------------------
__global__ void sgemm_naive(int M, int N, int K, float alpha,
                            const float* __restrict__ A,
                            const float* __restrict__ B, float beta,
                            float* __restrict__ C) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: global-memory coalescing.
// Same arithmetic, but threadIdx.x now indexes the *column* of C. A warp reads
// one broadcast value of A and 32 contiguous floats of B (one 128B
// transaction), and writes 32 contiguous floats of C.
// ---------------------------------------------------------------------------
__global__ void sgemm_coalesced(int M, int N, int K, float alpha,
                                const float* __restrict__ A,
                                const float* __restrict__ B, float beta,
                                float* __restrict__ C) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: 32x32 shared-memory tiling + coalesced global loads.
// Each 32x32 thread block computes one 32x32 tile of C. It walks along K in
// steps of TILE: every thread loads one element of the A tile and one of the
// B tile (threadIdx.x -> contiguous column => coalesced 128B loads), the block
// synchronises, and each thread does TILE FMAs out of shared memory.
// Global traffic per output drops from 2K loads to 2K/TILE loads.
//
// Shared-memory bank behaviour in the inner loop:
//   As[ty][k] - same address across the warp (ty is constant in a warp) -> broadcast
//   Bs[k][tx] - 32 consecutive words -> 32 distinct banks, conflict-free
// ---------------------------------------------------------------------------
template <int TILE>
__global__ void __launch_bounds__(TILE * TILE)
sgemm_smem_tiled(int M, int N, int K, float alpha,
                 const float* __restrict__ A,
                 const float* __restrict__ B, float beta,
                 float* __restrict__ C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;  // column within tile
    const int ty = threadIdx.y;  // row within tile
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    for (int t = 0; t < K; t += TILE) {
        const int a_col = t + tx;
        const int b_row = t + ty;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
}
