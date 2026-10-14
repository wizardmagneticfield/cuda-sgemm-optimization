// Benchmark + correctness harness for the SGEMM kernels in kernels.cuh.
//
// Usage:
//   ./sgemm                      # default sizes 256..8192
//   ./sgemm 1024 2048            # custom square sizes
//   ./sgemm --kernel tiled32 4096
//
// Every kernel is checked against cuBLAS before it is timed.
// Results are also written to results.csv.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "kernels.cuh"

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e),       \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(x)                                                        \
  do {                                                                         \
    cublasStatus_t s = (x);                                                    \
    if (s != CUBLAS_STATUS_SUCCESS) {                                          \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s, __FILE__, __LINE__); \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)


struct Kernel {
  std::string name;
  // Returns false if the kernel can't run at this size.
  bool (*launch)(int M, int N, int K, const float* A, const float* B, float* C);
};

static const float kAlpha = 1.0f, kBeta = 0.0f;

static bool launch_naive(int M, int N, int K, const float* A, const float* B, float* C) {
  dim3 block(32, 32);
  dim3 grid((M + 31) / 32, (N + 31) / 32);
  sgemm_naive<<<grid, block>>>(M, N, K, kAlpha, A, B, kBeta, C);
  return true;
}

static bool launch_coalesced(int M, int N, int K, const float* A, const float* B, float* C) {
  dim3 block(32, 32);
  dim3 grid((N + 31) / 32, (M + 31) / 32);
  sgemm_coalesced<<<grid, block>>>(M, N, K, kAlpha, A, B, kBeta, C);
  return true;
}

template <int TILE>
static bool launch_tiled(int M, int N, int K, const float* A, const float* B, float* C) {
  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  sgemm_smem_tiled<TILE><<<grid, block>>>(M, N, K, kAlpha, A, B, kBeta, C);
  return true;
}

template <int BM, int BN, int BK, int TM>
static bool launch_blocktile(int M, int N, int K, const float* A, const float* B, float* C) {
  if (M % BM || N % BN || K % BK) return false;
  static_assert(BM * BK == BM * BN / TM, "A tile load must be 1 element/thread");
  static_assert(BK * BN == BM * BN / TM, "B tile load must be 1 element/thread");
  dim3 block(BM * BN / TM);
  dim3 grid(N / BN, M / BM);
  sgemm_blocktile_1d<BM, BN, BK, TM><<<grid, block>>>(M, N, K, kAlpha, A, B, kBeta, C);
  return true;
}

static std::vector<Kernel> all_kernels() {
  return {
      {"naive", launch_naive},
      {"coalesced", launch_coalesced},
      {"tiled16", launch_tiled<16>},
      {"tiled32", launch_tiled<32>},
      {"blocktile1d_64x64x8_tm8", launch_blocktile<64, 64, 8, 8>},
  };
}

// cuBLAS is column-major. For row-major C = A @ B we compute C^T = B^T @ A^T,
// which in column-major terms is just swapping the operands.
static void cublas_sgemm(cublasHandle_t h, int M, int N, int K, const float* A,
                         const float* B, float* C) {
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &kAlpha, B, N,
                           A, K, &kBeta, C, N));
}

// Average time per call in ms. One warm-up call, one call to estimate the
// cost, then enough calls to fill ~0.5 s (clamped to [3, 1000]) so fast
// kernels get a stable average and the slow naive kernel still finishes.
template <typename F>
static float time_ms(F&& fn) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  auto timed = [&](int iters) {
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  };
  fn();  // warm-up
  const float one = timed(1);
  const int iters = std::min(1000, std::max(3, (int)(500.0f / std::max(one, 1e-3f))));
  const float ms = timed(iters) / iters;
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

// Max relative error, normalised by the reference's max magnitude.
static float max_rel_error(const std::vector<float>& ref, const std::vector<float>& out) {
  float max_ref = 1e-12f, max_diff = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i) {
    max_ref = std::max(max_ref, std::fabs(ref[i]));
    max_diff = std::max(max_diff, std::fabs(ref[i] - out[i]));
  }
  return max_diff / max_ref;
}

int main(int argc, char** argv) {
  std::vector<int> sizes;
  std::string only_kernel;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--kernel") && i + 1 < argc) only_kernel = argv[++i];
    else sizes.push_back(atoi(argv[i]));
  }
  if (sizes.empty()) sizes = {256, 512, 1024, 2048, 4096, 8192};
  if (!only_kernel.empty()) {
    bool known = false;
    for (const auto& k : all_kernels()) known |= k.name == only_kernel;
    if (!known) {
      fprintf(stderr, "Unknown kernel '%s'. Options:", only_kernel.c_str());
      for (const auto& k : all_kernels()) fprintf(stderr, " %s", k.name.c_str());
      fprintf(stderr, "\n");
      return 1;
    }
  }

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("Device: %s (SM %d.%d, %d SMs)\n\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount);

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  // Default math mode = plain FP32 SGEMM (TF32 tensor cores stay off), so the
  // comparison is fair without handicapping cuBLAS the way PEDANTIC mode can.
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

  FILE* csv = fopen("results.csv", "w");
  if (csv) fprintf(csv, "kernel,n,ms,gflops,pct_cublas,speedup_vs_naive,max_rel_err\n");

  const float tol = 1e-4f;
  bool all_ok = true;

  printf("%-26s %6s %10s %10s %9s %9s %s\n", "kernel", "N", "ms", "GFLOP/s",
         "%cuBLAS", "vsNaive", "correct");
  printf("%s\n", std::string(86, '-').c_str());

  for (int n : sizes) {
    const int M = n, N = n, K = n;
    const size_t bytesA = (size_t)M * K * sizeof(float);
    const size_t bytesB = (size_t)K * N * sizeof(float);
    const size_t bytesC = (size_t)M * N * sizeof(float);

    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : hA) x = dist(rng);
    for (auto& x : hB) x = dist(rng);

    float *dA, *dB, *dC, *dRef;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMalloc(&dRef, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

    const double flops = 2.0 * M * N * K;

    cublas_sgemm(handle, M, N, K, dA, dB, dRef);
    std::vector<float> hRef((size_t)M * N), hOut((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, bytesC, cudaMemcpyDeviceToHost));

    const float cublas_ms = time_ms([&] { cublas_sgemm(handle, M, N, K, dA, dB, dRef); });
    const double cublas_gflops = flops / (cublas_ms * 1e6);
    printf("%-26s %6d %10.3f %10.1f %8.1f%% %9s %s\n", "cuBLAS", n, cublas_ms,
           cublas_gflops, 100.0, "-", "ref");
    if (csv) fprintf(csv, "cublas,%d,%.4f,%.1f,100.0,,\n", n, cublas_ms, cublas_gflops);

    double naive_ms = -1;
    for (const auto& k : all_kernels()) {
      if (!only_kernel.empty() && k.name != only_kernel && k.name != "naive") continue;

      CUDA_CHECK(cudaMemset(dC, 0, bytesC));
      if (!k.launch(M, N, K, dA, dB, dC)) {
        printf("%-26s %6d %10s (size not supported)\n", k.name.c_str(), n, "skip");
        continue;
      }
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(hOut.data(), dC, bytesC, cudaMemcpyDeviceToHost));
      const float err = max_rel_error(hRef, hOut);
      const bool ok = err < tol;
      all_ok &= ok;

      const float ms = time_ms([&] { k.launch(M, N, K, dA, dB, dC); });
      if (k.name == "naive") naive_ms = ms;
      const double gflops = flops / (ms * 1e6);
      char speedup[16] = "-";
      if (naive_ms > 0) snprintf(speedup, sizeof speedup, "%.1fx", naive_ms / ms);
      printf("%-26s %6d %10.3f %10.1f %8.1f%% %9s %s (err %.1e)\n", k.name.c_str(),
             n, ms, gflops, 100.0 * gflops / cublas_gflops, speedup,
             ok ? "PASS" : "FAIL", err);
      if (csv)
        fprintf(csv, "%s,%d,%.4f,%.1f,%.2f,%.2f,%.2e\n", k.name.c_str(), n, ms,
                gflops, 100.0 * gflops / cublas_gflops,
                naive_ms > 0 ? naive_ms / ms : 0.0, err);
    }
    printf("\n");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dRef));
  }

  CUBLAS_CHECK(cublasDestroy(handle));
  if (csv) {
    fclose(csv);
    printf("Wrote results.csv\n");
  }
  printf(all_ok ? "All kernels passed correctness checks.\n"
                : "Some kernels FAILED correctness checks.\n");
  return all_ok ? 0 : 1;
}
