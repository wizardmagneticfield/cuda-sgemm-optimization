#!/usr/bin/env bash
# Profile each custom kernel at N=4096 with Nsight Compute.
# Reports are written to profiles/<kernel>.ncu-rep (open with ncu-ui).
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=./sgemm
SIZE=${1:-4096}
mkdir -p profiles

declare -A KNAME=([naive]=sgemm_naive [coalesced]=sgemm_coalesced [tiled]=regex:sgemm_smem_tiled)
for k in naive coalesced tiled; do
  echo "== Profiling $k (N=$SIZE) =="
  ncu --set full \
      --kernel-name "${KNAME[$k]}" \
      --launch-skip 1 --launch-count 1 \
      -o "profiles/$k" -f \
      "$BIN" "$k" "$SIZE" > /dev/null
  # Key metrics to the terminal: DRAM throughput, L1/shared traffic, achieved
  # occupancy, FMA pipe utilisation, and global-load sectors per request
  # (4 = perfectly coalesced 128B warp access, 32 = fully uncoalesced).
  ncu --import "profiles/$k.ncu-rep" --page raw --csv \
      --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
done
