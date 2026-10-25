#!/usr/bin/env bash
# Inference-time window-length probe: re-run validation inference of already trained seeds with a different window
# length (and stride = half of it, tail covered), and score it. No training. Tells whether a model trained on short
# windows can use a longer context at inference (the SSM state simply runs longer).
#
# usage:  bash scripts/probe_window.sh <NoXi|NoXi+J> <gpu> <tag> <window_len> [flags of the original run ...]
# e.g.:   SEEDS="95 1 2" bash scripts/probe_window.sh NoXi+J 0 ftw60_noxij 3000 --shuffle-windows --fusion-temporal --active-modalities ...
#
# Pass the original run's flags, but no --window-len / --window-stride / --cover-tail (set here; later flags win anyway).
# Per seed: sweep_out/<tag>_w<len>_s<seed>/, logs/infer_<tag>_w<len>_s<seed>.log; scores appended to logs/eval_<tag>_w<len>.log.

set -u
if [ $# -lt 4 ]; then echo "usage: $0 <NoXi|NoXi+J> <gpu> <tag> <window_len> [flags]"; exit 1; fi
corpus=$1; gpu=$2; tag=$3; W=$4; shift 4

cd "$(dirname "$0")/.." || exit 1
source EngageNet_venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$gpu
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION:-0.40}
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer="

M=(--active-modalities audio.egemapsv2 audio.w2vbert2_embeddings openface2 openpose)
case $corpus in
	NoXi)   sub="[('NoXi','test-base')]" ;;
	NoXi+J) sub="[('NoXi+J','test')]" ;;
	*) echo "unknown corpus $corpus"; exit 1 ;;
esac
P="import sys; sys.path.insert(0,'src'); import inference as I; I.SUBMISSION_CORPORA=$sub; I.main()"

for s in ${SEEDS:-95 1 2}; do
	t=${tag}_s$s; o=${tag}_w${W}_s$s
	if [ ! -d "models_$t" ]; then echo "$(date +%T) skip $t (no models_$t)"; continue; fi
	echo "$(date +%T) infer $o"
	python -c "$P" --checkpoint-dir "$PWD/models_$t" --submission-dir "sweep_out/$o" --submission-split val "${M[@]}" "$@" --window-len "$W" --window-stride $((W / 2)) --cover-tail --infer-batch 1 > "logs/infer_$o.log" 2>&1 || { echo "$(date +%T) INFER FAILED $o"; continue; }
	python src/evaluate.py --submission-dir "sweep_out/$o" --data-root data --corpora "$corpus" --split val 2>&1 | grep "CCC =" | sed "s/^/$o /" >> "logs/eval_${tag}_w$W.log"
	echo "$(date +%T) done $o"
done
echo "$(date +%T) all seeds finished for ${tag}_w$W"
