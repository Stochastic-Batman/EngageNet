#!/usr/bin/env bash
# Order sensitivity of trained fixed-order models (inference only, no training).
#
# usage:  bash scripts/probe_orders.sh <NoXi|NoXi+J> <gpu> <order> [<order> ...]
# e.g.:   bash scripts/probe_orders.sh NoXi 0 blocked interleaved
#
# For each trained order X (models_ord_X_<corpus>_s<seed>, seeds from SEEDS, default "95 1 2") and each probe Y in
# PROBES (default: the four training orders plus rand2 rand3 rand4, minus X itself), the model scans the streams in
# order Y and its outputs are mapped back to the layout it was trained on, so only the scan order changes.
# Scores: logs/eval_probe_X_<corpus>_Y.log. Then dumps the A_bar statistics of seed 95 to logs/decay_ord_X_<corpus>.log.
# Resumable: every finished (seed, probe) leaves a done marker (see rescore.sh).

set -u
if [ $# -lt 3 ]; then echo "usage: $0 <NoXi|NoXi+J> <gpu> <order> [<order> ...]"; exit 1; fi
corpus=$1; gpu=$2; shift 2

cd "$(dirname "$0")/.." || exit 1
case $corpus in NoXi) c=noxi ;; NoXi+J) c=noxij ;; *) echo "unknown corpus $corpus"; exit 1 ;; esac
F=(--shuffle-windows --active-modalities audio.egemapsv2 audio.w2vbert2_embeddings audio.xlm_roberta_embeddings openface2 openpose)

for X in "$@"; do
	for Y in ${PROBES:-blocked interleaved rand0 rand1 rand2 rand3 rand4}; do
		[ "$Y" = "$X" ] && continue
		SEEDS="${SEEDS:-95 1 2}" bash scripts/rescore.sh "$corpus" "$gpu" "ord_${X}_$c" "probe_${X}_${c}_$Y" "${F[@]}" --fixed-order "$X" --probe-order "$Y"
	done
	if ! grep -q "mean A_bar" "logs/decay_ord_${X}_$c.log" 2>/dev/null; then
		echo "$(date +%T) decay ord_${X}_$c"
		( source EngageNet_venv/bin/activate; export XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION:-0.40} XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer="; python scripts/dump_decay.py --corpus "$corpus" --checkpoint-dir "$PWD/models_ord_${X}_${c}_s95" "${F[@]}" --fixed-order "$X" > "logs/decay_ord_${X}_$c.log" 2>&1 )
	fi
done
echo "$(date +%T) all probes finished for $corpus: $*"
