# CUDA SGEMM Optimization

Single-precision GEMM (`C = αAB + βC`, row-major) written from scratch in CUDA C++,
benchmarked against cuBLAS and profiled with Nsight Compute. Target: **RTX 4090 (Ada, sm_89)**.

| # | Kernel | Idea |
|---|--------|------|
| 0 | `cublas` | `cublasSgemm`, FP32 (TF32 disabled) — reference & upper bound |
| 1 | `naive` | one thread per output; `threadIdx.x` → row, so warp loads are strided (uncoalesced) |
| 2 | `coalesced` | same math, `threadIdx.x` → column, so each warp reads/writes one contiguous 128 B line |
| 3 | `tiled32` | 32×32 thread block computes a 32×32 C tile; A/B tiles staged in shared memory with coalesced loads, cutting global loads per output from `2K` to `2K/32` |

Kernels live in [`src/kernels.cuh`](src/kernels.cuh); the benchmark harness is [`src/main.cu`](src/main.cu).

## Build & run

Requires CUDA 11.8+ (sm_89) and cuBLAS.

```bash
# CMake
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/sgemm                      # all kernels, N = 256..4096

# or plain make
make && ./sgemm
./sgemm tiled 4096                 # one kernel, one size
make ARCH=sm_86                    # other GPUs
```

Each run checks every kernel against cuBLAS, prints time / GFLOP/s / % of cuBLAS / speedup over
naive, and writes `results.csv`.

## Profiling (Nsight Compute)

```bash
./scripts/profile_ncu.sh 4096      # writes profiles/{naive,coalesced,tiled}.ncu-rep
ncu-ui profiles/tiled.ncu-rep
```

Things to look at in each report:

- **Global-load sectors / request** (`l1tex__t_sectors_pipe_lsu_mem_global_op_ld` ÷ `..._requests_...`):
  naive is close to 32 sectors per request (each thread hits its own sector). Coalesced and tiled
  are close to 4, i.e. one 128 B line per warp.
- **DRAM / L1 throughput vs. FMA pipe utilisation**: naive and coalesced are memory-bound. Tiling
  moves the bottleneck to shared-memory bandwidth and `__syncthreads` stalls.
- **Shared-memory bank conflicts**: zero. `As[ty][k]` is a broadcast within a warp, and `Bs[k][tx]`
  hits 32 different banks.
- **Warp stall reasons**: for `tiled32` most stalls are `MIO Throttle` / `Short Scoreboard`. Each FMA
  needs two shared-memory loads, so the kernel is limited by shared-memory traffic, not compute.

## Results (RTX 4090, N = 4096)

> Fill in from `results.csv` after running on your GPU.

| Kernel | GFLOP/s | % cuBLAS | × naive |
|--------|--------:|---------:|--------:|
| naive | | | 1.0× |
| coalesced | | | |
| tiled32 | | | |
| cuBLAS | | 100% | |

## Why tiling helps, and where it stops helping

The naive kernel does 2 global loads per FMA. It reaches only a few hundred GFLOP/s because
uncoalesced access wastes most of each 32 B sector it fetches. Coalescing alone gives a large
gain. Shared-memory tiling then reuses each loaded element 32 times from on-chip memory.

`tiled32` still does 2 shared-memory loads per FMA, because each thread computes only one output.
Each 1024-thread block uses 40 registers per thread and 8 KB of shared memory (ptxas, sm_89), so
only one block fits on an SM. That is 32 of 48 warps (67% occupancy), and every `__syncthreads`
stalls all of them. That is why it sits well below cuBLAS. Next steps:

1. **Register blocking (1D/2D thread tiles)**: each thread computes 8 or 8×8 outputs, reusing values
   from registers.
2. **Vectorized `float4` loads** and a transposed A tile in shared memory.
3. **Double buffering / `cp.async`** to overlap global→shared copies with compute.
4. **Warp tiling** to match the cuBLAS/CUTLASS hierarchy.
