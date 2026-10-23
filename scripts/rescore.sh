#!/usr/bin/env bash
# Re-run validation inference + scoring for already-trained seeds, without training.
#
# usage:  bash scripts/rescore.sh <NoXi|NoXi+J> <gpu> <model_tag> <out_tag> [extra flags for inference.py ...]
# e.g.:   bash scripts/rescore.sh NoXi 0 feat_noxi_t feat_noxi_t_besttau --shuffle-windows --active-modalities ...
#
# Reads models_<model_tag>_s<seed>/, writes sweep_out/<out_tag>_s<seed>/, appends scores to logs/eval_<out_tag>.log.
# Seeds default to "95 1 2 3 4 5" (override with SEEDS="..."). Finished seeds leave logs/eval_<out_tag>_s<seed>.done
# and are skipped on relaunch. The temperature comes from models_.../best_tau.json if present (see inference.resolve_tau).

set -u
if [ $# -lt 4 ]; then echo "usage: $0 <NoXi|NoXi+J> <gpu> <model_tag> <out_tag> [extra flags]"; exit 1; fi
corpus=$1; gpu=$2; mtag=$3; otag=$4; shift 4

cd "$(dirname "$0")/.." || exit 1

exec 9>"logs/$otag.lock"
if ! flock -n 9; then echo "$(date +%T) $otag is already running - not starting a second copy"; exit 1; fi
source EngageNet_venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$gpu
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION:-0.40}
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer="

case $corpus in
	NoXi)   sub="[('NoXi','test-base')]" ;;
	NoXi+J) sub="[('NoXi+J','test')]" ;;
	*) echo "unknown corpus $corpus"; exit 1 ;;
esac
P="import sys; sys.path.insert(0,'src'); import inference as I; I.SUBMISSION_CORPORA=$sub; I.main()"

for s in ${SEEDS:-95 1 2 3 4 5}; do
	t=${otag}_s$s
	if [ -f "logs/eval_$t.done" ]; then echo "$(date +%T) skip $t (done)"; continue; fi
	if [ ! -d "models_${mtag}_s$s" ]; then echo "$(date +%T) MISSING models_${mtag}_s$s"; continue; fi
	echo "$(date +%T) infer $t"
	python -c "$P" --corpus "$corpus" --checkpoint-dir "$PWD/models_${mtag}_s$s" --submission-dir "sweep_out/$t" --submission-split val "$@" > "logs/infer_$t.log" 2>&1 || { echo "$(date +%T) INFER FAILED $t"; continue; }
	grep -h "Inference temperature" "logs/infer_$t.log" | tail -n 1 | sed "s/^/$t /" >> "logs/eval_$otag.log"
	python src/evaluate.py --submission-dir "sweep_out/$t" --data-root data --corpora "$corpus" --split val 2>&1 | grep "CCC =" | sed "s/^/$t /" >> "logs/eval_$otag.log"
	touch "logs/eval_$t.done"
	echo "$(date +%T) done $t"
done
echo "$(date +%T) all seeds finished for $otag"
