#!/usr/bin/env bash
# Profile each custom kernel with Nsight Compute.
# Usage: ./scripts/profile_ncu.sh [size]   (default 4096)
# Reports go to profiles/<kernel>.ncu-rep; open them with ncu-ui.
set -euo pipefail
cd "$(dirname "$0")/.."
SIZE=${1:-4096}
BIN=./sgemm
[ -x "$BIN" ] || make

declare -A FILTER=(
  [naive]=sgemm_naive
  [coalesced]=sgemm_coalesced
  [tiled32]=regex:sgemm_smem_tiled
  [blocktile1d_64x64x8_tm8]=regex:sgemm_blocktile_1d
)
mkdir -p profiles

for k in naive coalesced tiled32 blocktile1d_64x64x8_tm8; do
  echo "== $k (N=$SIZE) =="
  ncu --set full --kernel-name "${FILTER[$k]}" --launch-count 1 \
      -o "profiles/$k" -f "$BIN" --kernel "$k" "$SIZE" > /dev/null
  # Global-load sectors per request: ~4 means a warp reads one 128B line
  # (coalesced); ~32 means every thread hits its own sector (uncoalesced).
  ncu --import "profiles/$k.ncu-rep" --page raw --csv --metrics \
gpu__time_duration.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum
done
