#pragma once
// SGEMM kernels:  C = alpha * A @ B + beta * C
// All matrices are row-major. A is MxK, B is KxN, C is MxN.

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// 1. Naive kernel.
// threadIdx.x walks down *rows*, so neighbouring threads in a warp read
// A rows that are K floats apart and write C elements N floats apart.
// This is deliberately uncoalesced so it serves as a baseline.
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
// 2. Coalesced kernel.
// Same math, but threadIdx.x now walks along *columns*: a warp reads 32
// consecutive floats of B and writes 32 consecutive floats of C.
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
// 3. Shared-memory tiled kernel (TILE x TILE threads per block).
// Each block loads a TILE x TILE tile of A and of B into shared memory with
// coalesced loads, then every thread does TILE FMAs from shared memory.
// Global memory traffic drops by a factor of ~TILE.
//
// No padding is needed: in the inner loop As[ty][k] is the same address for
// the whole warp (broadcast) and Bs[k][tx] hits 32 consecutive banks, so
// neither access has bank conflicts.
// ---------------------------------------------------------------------------
template <int TILE>
__global__ void sgemm_smem_tiled(int M, int N, int K, float alpha,
                                 const float* __restrict__ A,
                                 const float* __restrict__ B, float beta,
                                 float* __restrict__ C) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;
  const int col = blockIdx.x * TILE + tx;

  float acc = 0.0f;
  for (int t = 0; t < K; t += TILE) {
    As[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.0f;
    Bs[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

// ---------------------------------------------------------------------------
// 4. 1D block-tiled kernel: each thread computes TM outputs in a column.
// Raises arithmetic intensity per thread (more FMAs per shared-memory load).
// Requires M % BM == 0, N % BN == 0, K % BK == 0 (checked on the host).
// Threads per block = BM * BN / TM. BM*BK and BK*BN must equal that count.
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM>
__global__ void sgemm_blocktile_1d(int M, int N, int K, float alpha,
                                   const float* __restrict__ A,
                                   const float* __restrict__ B, float beta,
                                   float* __restrict__ C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];
  (void)M;  // bounds are guaranteed by the host-side divisibility check

  const int cRow = blockIdx.y, cCol = blockIdx.x;
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  const int innerColA = threadIdx.x % BK, innerRowA = threadIdx.x / BK;
  const int innerColB = threadIdx.x % BN, innerRowB = threadIdx.x / BN;

  float results[TM] = {0.0f};

  for (int bk = 0; bk < K; bk += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();
    A += BK;
    B += BK * N;
#pragma unroll
    for (int dot = 0; dot < BK; ++dot) {
      const float b = Bs[dot * BN + threadCol];
#pragma unroll
      for (int r = 0; r < TM; ++r) results[r] += As[(threadRow * TM + r) * BK + dot] * b;
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < TM; ++r) {
    const int idx = (threadRow * TM + r) * N + threadCol;
    C[idx] = alpha * results[r] + beta * C[idx];
  }
}
