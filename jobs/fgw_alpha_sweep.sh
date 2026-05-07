#!/bin/bash
# Submit one fgw_experiments SLURM job per alpha, sharing n and seed.
#
# Usage:
#   bash jobs/fgw_alpha_sweep.sh                            # n=200, seed=0, default alphas
#   bash jobs/fgw_alpha_sweep.sh 500 7                      # n=500, seed=7, default alphas
#   bash jobs/fgw_alpha_sweep.sh 200 0 0.0 0.3 0.7 1.0      # n=200, seed=0, custom alphas
#
# Each alpha launches an independent fgw_experiments.job in parallel on the
# cluster, writing to a distinct output file so they don't race:
#   fgw_validation/data/results/fgw_n{N}_a{ALPHA}_s{SEED}.json
# and a matching _plans/ directory next to it (--save_plans is on).
#
# Default alpha grid spans pure-Wasserstein (0.0) to pure-GW (1.0):
#   0.0  → only the caption bridge M matters
#   1.0  → only the inner geometries C_i, C_a matter
#   0.5  → POT default

set -e
set -o pipefail

N="${1:-200}"
SEED="${2:-0}"

if [ "$#" -gt 2 ]; then
    shift 2
    ALPHAS=("$@")
else
    ALPHAS=(0.0 0.1 0.25 0.5 0.75 0.9 1.0)
fi

echo "[sweep] n=$N  seed=$SEED  alphas=${ALPHAS[*]}"

for A in "${ALPHAS[@]}"; do
    sbatch jobs/fgw_experiments.job "$N" "$A" "$SEED"
done

echo "[sweep] submitted ${#ALPHAS[@]} job(s)"
