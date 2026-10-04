#!/usr/bin/env bash
# Train + score one configuration over several seeds, resumably.
#
# usage:  bash scripts/run_seeds.sh <NoXi|NoXi+J> <gpu> <tag> [extra flags for train.py and inference.py ...]
# e.g.:   bash scripts/run_seeds.sh NoXi 0 shuf_noxi --shuffle-windows
#
# Seeds default to "95 1 2" (override: SEEDS="3 4" bash scripts/run_seeds.sh ...).
# Per seed: models_<tag>_s<seed>/, logs/train_|infer_<tag>_s<seed>.log, sweep_out/<tag>_s<seed>/.
# Scores are appended to logs/eval_<tag>.log; a finished seed leaves logs/eval_<tag>_s<seed>.done
# and is skipped when the script is started again, so an interrupted run can simply be relaunched.
# Extra flags go to both train.py and inference.py (same config parser), so window settings match.

set -u
if [ $# -lt 3 ]; then echo "usage: $0 <NoXi|NoXi+J> <gpu> <tag> [extra flags]"; exit 1; fi
corpus=$1; gpu=$2; tag=$3; shift 3

cd "$(dirname "$0")/.." || exit 1

# One driver per tag: a second launch with the same tag exits instead of training the same seeds in parallel
exec 9>"logs/$tag.lock"
if ! flock -n 9; then echo "$(date +%T) $tag is already running - not starting a second copy"; exit 1; fi
source EngageNet_venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$gpu
# Cap each process (default would let one process grow to 75% of the card) so two runs fit on one GPU,
# and skip kernel autotuning, whose temporary buffers caused the out-of-memory crashes.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION:-0.45}
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_autotune_level=0"

M=(--active-modalities audio.egemapsv2 audio.w2vbert2_embeddings openface2 openpose)
BASE=(--lr 2e-4 --patience 15 --lambda-ccc 1.0)
case $corpus in
	NoXi)   sub="[('NoXi','test-base')]" ;;
	NoXi+J) sub="[('NoXi+J','test')]" ;;
	*) echo "unknown corpus $corpus"; exit 1 ;;
esac
P="import sys; sys.path.insert(0,'src'); import inference as I; I.SUBMISSION_CORPORA=$sub; I.main()"

for s in ${SEEDS:-95 1 2}; do
	t=${tag}_s$s
	if [ -f "logs/eval_$t.done" ]; then echo "$(date +%T) skip $t (done)"; continue; fi
	echo "$(date +%T) train $t"
	python src/train.py --corpus "$corpus" --checkpoint-dir "$PWD/models_$t" "${M[@]}" "${BASE[@]}" --seed "$s" "$@" > "logs/train_$t.log" 2>&1 || { echo "$(date +%T) TRAIN FAILED $t"; continue; }
	echo "$(date +%T) infer $t"
	python -c "$P" --checkpoint-dir "$PWD/models_$t" --submission-dir "sweep_out/$t" --submission-split val "${M[@]}" "$@" > "logs/infer_$t.log" 2>&1 || { echo "$(date +%T) INFER FAILED $t"; continue; }
	python src/evaluate.py --submission-dir "sweep_out/$t" --data-root data --corpora "$corpus" --split val 2>&1 | grep "CCC =\|CDD_G" | sed "s/^/$t /" >> "logs/eval_$tag.log"
	touch "logs/eval_$t.done"
	echo "$(date +%T) done $t"
done
echo "$(date +%T) all seeds finished for $tag"
