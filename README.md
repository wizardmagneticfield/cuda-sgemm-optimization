# CUDA SGEMM Optimization

Hand-written single-precision matrix multiply (`C = αAB + βC`, row-major) kernels in CUDA C++,
optimized step by step, benchmarked against cuBLAS, and profiled with Nsight Compute.
Target GPU: RTX 4090 (Ada, `sm_89`).

| Kernel | Idea |
|---|---|
| `naive` | One thread per output. Thread x-index walks rows, so global loads/stores are **uncoalesced**. Baseline. |
| `coalesced` | Same math, but consecutive threads handle consecutive columns, so a warp reads/writes 32 contiguous floats (one 128 B line). |
| `tiled16` / `tiled32` | Shared-memory tiling: each block stages a TILE×TILE tile of A and B in shared memory with coalesced loads and reuses it TILE times. |
| `blocktile1d_64x64x8_tm8` | Each thread computes 8 outputs, raising arithmetic intensity per shared-memory load. |

Every kernel is checked against cuBLAS (plain FP32, TF32 off) before it is timed. Matrices are
square, from 256 to 8192. Kernels are in [`src/kernels.cuh`](src/kernels.cuh), the benchmark in
[`src/main.cu`](src/main.cu).

## Build and run

Needs an NVIDIA GPU, the CUDA Toolkit (nvcc), and cuBLAS (comes with the toolkit).

```bash
make ARCH=sm_89               # sm_89 = RTX 40xx, sm_86 = RTX 30xx, sm_75 = T4
./sgemm                       # all kernels, sizes 256..8192
./sgemm 1024 4096             # specific sizes
./sgemm --kernel tiled32 4096 # one kernel (naive always runs too, for the speedup column)
```

Or with CMake: `cmake -B build && cmake --build build -j && ./build/sgemm`.

Each run prints time, GFLOP/s, % of cuBLAS, speedup over naive, and PASS/FAIL, and writes
everything to `results.csv`. Each kernel is timed over ~0.5 s of repeated launches for a stable
average.

**No local GPU?** Use Google Colab (Runtime → Change runtime type → T4 GPU):

```bash
!git clone https://github.com/wizardmagneticfield/cuda-sgemm-optimization.git
%cd cuda-sgemm-optimization
!make ARCH=sm_75 && ./sgemm
```

## Profiling with Nsight Compute

```bash
make profile                        # = ./scripts/profile_ncu.sh 4096
ncu-ui profiles/tiled32.ncu-rep
```

The script writes one report per kernel to `profiles/` and prints the key metrics. Things to look at:

- **Global-load sectors per request** (`l1tex__t_sectors_..._ld` ÷ `l1tex__t_requests_..._ld`):
  ~32 for `naive` (every thread hits its own 32 B sector), ~4 for the others (one 128 B line per warp).
- **Shared-memory bank conflicts**: ~0 for `tiled32`. `As[ty][k]` is a broadcast within a warp and
  `Bs[k][tx]` hits 32 consecutive banks, so no padding is needed.
- **Occupancy**: `tiled32` uses 1024-thread blocks, so only one block fits per SM (32 of 48 warps).
- **Warp State Statistics**: `Stall MIO Throttle` / `Stall Short Scoreboard` mean waiting on shared
  memory; `Stall Long Scoreboard` means waiting on global memory.

## Results

> Fill this in with numbers from **your** runs (`results.csv`). Note which GPU you used.

GPU: _____

| Kernel | N=1024 GFLOP/s | N=4096 GFLOP/s | % of cuBLAS (4096) | Speedup vs naive |
|---|---|---|---|---|
| cuBLAS | | | 100% | |
| naive | | | | 1× |
| coalesced | | | | |
| tiled32 | | | | |
| blocktile1d | | | | |

## What I learned

_Write 3–5 bullets here after profiling: what bottleneck you saw in Nsight, what you changed, and what happened._

## Ideas for next steps

- 2D block tiling (each thread computes a TM×TN sub-tile)
- Vectorized `float4` loads
- Double buffering shared memory to overlap loads and compute
- Warp tiling
