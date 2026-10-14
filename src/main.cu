// SGEMM benchmark: naive vs. coalesced vs. 32x32 shared-memory tiling vs. cuBLAS.
//
// Usage: ./sgemm [kernel] [sizes...]
//   kernel: all (default) | cublas | naive | coalesced | tiled
//   sizes : square problem sizes, default 256 512 1024 2048 4096

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "kernels.cuh"

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(err_), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t st_ = (call);                                          \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__, \
                    __LINE__);                                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

constexpr int TILE = 32;

enum Kernel { CUBLAS = 0, NAIVE, COALESCED, TILED, NUM_KERNELS };
static const char* kNames[NUM_KERNELS] = {"cublas", "naive", "coalesced",
                                          "tiled32"};

static void run_kernel(Kernel k, cublasHandle_t handle, int M, int N, int K,
                       float alpha, const float* A, const float* B, float beta,
                       float* C) {
    switch (k) {
        case CUBLAS:
            // cuBLAS is column-major. Row-major C = A*B is column-major
            // C^T = B^T * A^T, so swap the operands instead of transposing.
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                     &alpha, B, N, A, K, &beta, C, N));
            break;
        case NAIVE: {
            dim3 block(32, 32);
            dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
            sgemm_naive<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }
        case COALESCED: {
            dim3 block(32, 32);
            dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
            sgemm_coalesced<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }
        case TILED: {
            dim3 block(TILE, TILE);
            dim3 grid(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
            sgemm_smem_tiled<TILE><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }
        default:
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

static bool verify(const std::vector<float>& ref, const std::vector<float>& out,
                   int K) {
    // FP32 accumulation-order differences grow roughly with sqrt(K).
    const float tol = 1e-4f * std::sqrt((float)K) + 1e-3f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float diff = std::fabs(ref[i] - out[i]);
        if (diff > tol * std::fmax(1.0f, std::fabs(ref[i]))) {
            fprintf(stderr, "  mismatch at %zu: ref=%f got=%f\n", i, ref[i],
                    out[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    std::vector<Kernel> kernels;
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) {
        bool matched = false;
        if (!strcmp(argv[i], "all")) {
            matched = true;
        }
        for (int k = 0; k < NUM_KERNELS && !matched; ++k) {
            if (!strcmp(argv[i], kNames[k]) ||
                (k == TILED && !strcmp(argv[i], "tiled"))) {
                kernels.push_back((Kernel)k);
                matched = true;
            }
        }
        if (!matched) sizes.push_back(atoi(argv[i]));
    }
    if (kernels.empty())
        for (int k = 0; k < NUM_KERNELS; ++k) kernels.push_back((Kernel)k);
    if (sizes.empty()) sizes = {256, 512, 1024, 2048, 4096};

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s (SM %d.%d, %d SMs)\n\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    // Keep cuBLAS on true FP32 (no TF32 tensor cores) for a fair comparison.
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    FILE* csv = fopen("results.csv", "w");
    if (csv) fprintf(csv, "kernel,size,ms,gflops,pct_cublas,speedup_vs_naive\n");

    printf("%-10s %6s %10s %10s %10s %10s\n", "kernel", "N", "ms", "GFLOP/s",
           "%cuBLAS", "x naive");

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const float alpha = 1.0f, beta = 0.0f;

    for (int n : sizes) {
        const int M = n, N = n, K = n;
        std::vector<float> hA((size_t)M * K), hB((size_t)K * N),
            hRef((size_t)M * N), hOut((size_t)M * N);
        for (auto& x : hA) x = dist(rng);
        for (auto& x : hB) x = dist(rng);

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, hRef.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

        // cuBLAS reference result for correctness checks.
        CUDA_CHECK(cudaMemset(dC, 0, hRef.size() * sizeof(float)));
        run_kernel(CUBLAS, handle, M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaMemcpy(hRef.data(), dC, hRef.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));

        const double flops = 2.0 * M * N * K;
        double ms_k[NUM_KERNELS] = {0};
        // Always time cuBLAS and naive so the ratios can be reported.
        std::vector<Kernel> to_run = kernels;
        auto has = [&](Kernel k) {
            for (auto x : to_run) if (x == k) return true;
            return false;
        };
        if (!has(CUBLAS)) to_run.insert(to_run.begin(), CUBLAS);
        if (!has(NAIVE)) to_run.insert(to_run.begin() + 1, NAIVE);

        for (Kernel k : to_run) {
            CUDA_CHECK(cudaMemset(dC, 0, hOut.size() * sizeof(float)));
            run_kernel(k, handle, M, N, K, alpha, dA, dB, beta, dC);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(hOut.data(), dC, hOut.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            if (k != CUBLAS && !verify(hRef, hOut, K)) {
                fprintf(stderr, "%s FAILED verification at N=%d\n", kNames[k], n);
                return EXIT_FAILURE;
            }

            // Warm-up, then time enough iterations to reach ~0.5s total.
            for (int i = 0; i < 3; ++i)
                run_kernel(k, handle, M, N, K, alpha, dA, dB, beta, dC);
            CUDA_CHECK(cudaEventRecord(start));
            run_kernel(k, handle, M, N, K, alpha, dA, dB, beta, dC);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float one_ms = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(&one_ms, start, stop));
            int iters = (int)(500.0f / std::fmax(one_ms, 1e-3f));
            iters = iters < 5 ? 5 : (iters > 1000 ? 1000 : iters);

            CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < iters; ++i)
                run_kernel(k, handle, M, N, K, alpha, dA, dB, beta, dC);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float total_ms = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
            ms_k[k] = total_ms / iters;
        }

        for (Kernel k : kernels) {
            const double gflops = flops / (ms_k[k] * 1e6);
            const double pct = 100.0 * ms_k[CUBLAS] / ms_k[k];
            const double speedup = ms_k[NAIVE] / ms_k[k];
            printf("%-10s %6d %10.3f %10.1f %9.1f%% %9.1fx\n", kNames[k], n,
                   ms_k[k], gflops, pct, speedup);
            if (csv)
                fprintf(csv, "%s,%d,%.4f,%.1f,%.2f,%.2f\n", kNames[k], n,
                        ms_k[k], gflops, pct, speedup);
        }
        printf("\n");

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    if (csv) {
        fclose(csv);
        printf("Wrote results.csv\n");
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
