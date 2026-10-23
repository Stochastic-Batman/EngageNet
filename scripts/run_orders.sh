#!/usr/bin/env bash
# Train + score several fixed orders one after another (run_seeds.sh per order, tag ord_<order>_<noxi|noxij>).
#
# usage:  SEEDS="3 4 5" bash scripts/run_orders.sh <NoXi|NoXi+J> <gpu> <order> [<order> ...]
# Resumable like run_seeds.sh; scores are appended to logs/eval_ord_<order>_<corpus>.log next to the earlier seeds.

set -u
if [ $# -lt 3 ]; then echo "usage: $0 <NoXi|NoXi+J> <gpu> <order> [<order> ...]"; exit 1; fi
corpus=$1; gpu=$2; shift 2
cd "$(dirname "$0")/.." || exit 1
case $corpus in NoXi) c=noxi ;; NoXi+J) c=noxij ;; *) echo "unknown corpus $corpus"; exit 1 ;; esac

for X in "$@"; do
	bash scripts/run_seeds.sh "$corpus" "$gpu" "ord_${X}_$c" --shuffle-windows --active-modalities audio.egemapsv2 audio.w2vbert2_embeddings audio.xlm_roberta_embeddings openface2 openpose --fixed-order "$X"
done
echo "$(date +%T) all orders finished for $corpus: $*"
