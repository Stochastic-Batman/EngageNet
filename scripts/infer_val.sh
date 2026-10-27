#!/usr/bin/env bash
# Validation inference of already trained seeds with extra inference flags (e.g. --tta --tta-person), and its score.
# No training. The original run's predictions in sweep_out/<tag>_s<seed>/ stay untouched.
#
# usage:  bash scripts/infer_val.sh <NoXi|NoXi+J> <gpu> <tag> <suffix> [flags of the original run + extra flags ...]
# e.g.:   bash scripts/infer_val.sh NoXi 0 ftb4_noxi ttap --shuffle-windows --fusion-temporal ... --tta --tta-person
#
# Per seed: sweep_out/<tag>_<suffix>_s<seed>/, logs/infer_<tag>_<suffix>_s<seed>.log; scores appended to
# logs/eval_<tag>_<suffix>.log. Seeds default to "95 1 2" (override with SEEDS=...). A finished seed is skipped.

set -u
if [ $# -lt 4 ]; then echo "usage: $0 <NoXi|NoXi+J> <gpu> <tag> <suffix> [flags]"; exit 1; fi
corpus=$1; gpu=$2; tag=$3; suffix=$4; shift 4

cd "$(dirname "$0")/.." || exit 1
exec 9>"logs/${tag}_$suffix.lock"
if ! flock -n 9; then echo "$(date +%T) ${tag}_$suffix is already running - not starting a second copy"; exit 1; fi
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
	t=${tag}_s$s; o=${tag}_${suffix}_s$s
	if [ -f "logs/eval_$o.done" ]; then echo "$(date +%T) skip $o (done)"; continue; fi
	if [ ! -d "models_$t" ]; then echo "$(date +%T) skip $t (no models_$t)"; continue; fi
	echo "$(date +%T) infer $o"
	python -c "$P" --checkpoint-dir "$PWD/models_$t" --submission-dir "sweep_out/$o" --submission-split val "${M[@]}" "$@" > "logs/infer_$o.log" 2>&1 || { echo "$(date +%T) INFER FAILED $o"; continue; }
	python src/evaluate.py --submission-dir "sweep_out/$o" --data-root data --corpora "$corpus" --split val 2>&1 | grep "CCC =" | sed "s/^/$o /" >> "logs/eval_${tag}_$suffix.log"
	touch "logs/eval_$o.done"
	echo "$(date +%T) done $o"
done
echo "$(date +%T) all seeds finished for ${tag}_$suffix"
