#!/usr/bin/env bash
# Test-split inference for every seed of one trained configuration, resumably.
#
# usage:  bash scripts/infer_test.sh <NoXi|NoXi+J> <gpu> <tag> [extra flags for inference.py ...]
# e.g.:   bash scripts/infer_test.sh NoXi+J 1 feat_noxij_t --active-modalities ...
#
# Reads models_<tag>_s<seed>/ and writes sweep_out/test_<tag>_s<seed>/ (NoXi: test-base and
# test-additional; NoXi+J: test). Seeds default to "95 1 2 3 4 5" (override with SEEDS=...).
# A finished seed leaves logs/test_<tag>_s<seed>.done and is skipped on a relaunch.

set -u
if [ $# -lt 3 ]; then echo "usage: $0 <NoXi|NoXi+J> <gpu> <tag> [extra flags]"; exit 1; fi
corpus=$1; gpu=$2; tag=$3; shift 3

cd "$(dirname "$0")/.." || exit 1
exec 9>"logs/test_$tag.lock"
if ! flock -n 9; then echo "$(date +%T) test_$tag is already running - not starting a second copy"; exit 1; fi

source EngageNet_venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$gpu
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION:-0.40}
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer="

case $corpus in
	NoXi)   sub="[('NoXi','test-base'),('NoXi','test-additional')]" ;;
	NoXi+J) sub="[('NoXi+J','test')]" ;;
	*) echo "unknown corpus $corpus"; exit 1 ;;
esac
P="import sys; sys.path.insert(0,'src'); import inference as I; I.SUBMISSION_CORPORA=$sub; I.main()"

for s in ${SEEDS:-95 1 2 3 4 5}; do
	t=${tag}_s$s
	if [ -f "logs/test_$t.done" ]; then echo "$(date +%T) skip $t (done)"; continue; fi
	if [ ! -d "models_$t" ]; then echo "$(date +%T) MISSING models_$t"; continue; fi
	echo "$(date +%T) test $t"
	python -c "$P" --checkpoint-dir "$PWD/models_$t" --submission-dir "sweep_out/test_$t" "$@" > "logs/test_$t.log" 2>&1 || { echo "$(date +%T) TEST FAILED $t"; continue; }
	touch "logs/test_$t.done"
	echo "$(date +%T) done $t"
done
echo "$(date +%T) all seeds finished for test_$tag"
