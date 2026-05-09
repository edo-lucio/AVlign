#!/bin/bash
# Submit `fgw_alpha_sweep.sh` for several seeds — produces the multi-seed
# data needed for the across-seed SE that backs the H_FGW headline test
# (see fgw_validation/EXPERIMENT.md §6.3).
#
# Each seed runs the full α-grid; the output filenames already include
# the seed, so files don't collide:
#   fgw_validation/data/results/fgw_n{N}_a{ALPHA}_s{SEED}.json
#
# Usage:
#   bash jobs/fgw_multiseed_sweep.sh                       # n=200, seeds 0..4
#   bash jobs/fgw_multiseed_sweep.sh 200 0 1 2 3 4 5 6     # n=200, 7 seeds
#   bash jobs/fgw_multiseed_sweep.sh 500 0 1 2 3 4         # n=500, 5 seeds
#
# Compute footprint: ~1.5 h on the cluster for the default 5 seeds × 7 α ×
# 72 combos at n=200. Each (α, seed) launches one SLURM job.

set -e
set -o pipefail

N="${1:-200}"
shift || true

if [ "$#" -gt 0 ]; then
    SEEDS=("$@")
else
    SEEDS=(0 1 2 3 4)
fi

echo "[multiseed] n=$N  seeds=${SEEDS[*]}  (×7 default alphas → $((${#SEEDS[@]} * 7)) jobs)"

for S in "${SEEDS[@]}"; do
    bash jobs/fgw_alpha_sweep.sh "$N" "$S"
done

echo "[multiseed] all seeds submitted"
