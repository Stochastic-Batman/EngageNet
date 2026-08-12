# EngageNet

This repository contains the codebase for the **Multi-Modal Social Behaviour Analysis** project at the **Georgian Artificial Intelligence Association (GAIA)**. Under the supervision of **Dr. Philipp Müller** and PhD candidate **Beso Mikaberidze**, our team is exploring behavioral personalization using high-fidelity multi-modal datasets.

## Project Overview

The project focuses on predicting and analyzing social behavior - specifically **engagement** and **personalization** - in dyadic and group interactions. We utilize the datasets provided by the [MultiMediate Challenge](https://multimediate-challenge.org/), which involve expert-novice dynamics across diverse cultural and linguistic backgrounds.

## Datasets

We use three corpora from the MultiMediate series. All three share the same file structure and can be read identically in code.

### Corpora

| Corpus | Sessions | Locations | Notes |
|---|---|---|---|
| **MPIIGroupInteraction** | - | - | Group dynamics; used for eye contact detection and bodily behaviour recognition |
| **NoXi** | 1–84 | Paris (1–25), Nottingham (26–65), Augsburg (66–84) | Large-scale multilingual expert-novice dyadic interactions in FR/EN/DE |
| **NoXi+J** | 85–150 | Ishikawa, Japan (85–150) | East-Asian extension of NoXi with Japanese and Chinese speakers; critical for cross-cultural personalization analysis |

All sessions are screen-mediated dyadic interactions between an **expert** and a **novice** discussing a topic. Expert and novice are recorded independently (separate webcams, separate audio), giving two fully independent participant streams per session.

### Splits

Each corpus is divided into splits. **Train and Val contain engagement annotations; Test does not** (withheld for the MultiMediate challenge leaderboard).

| Corpus | Train | Val | Test | Additional Test |
|---|---|---|---|---|
| NoXi | annotations | annotations | - | - |
| NoXi+J | annotations | annotations | - | N/A |

NoXi+J split example (session numbers):

```
train/  086 101–120 133–142
val/    121–126 143–146
test/   127–132 147–150
```

### Per-Session File Structure

Each session folder contains the following files for both `expert` and `novice`:

```
{role}.engagement.annotation.csv               # 25Hz frame-wise engagement score in [0,1], one float per line - train/val only
{role}.audio.transcript.annotation.csv         # Speech transcript: start_time;end_time;content;confidence
{role}.age.annotation.csv                      # Encoded age category: format start;end;category_id;conf - decode via NoXi_MetaData.xlsx
{role}.gender.annotation.csv                   # Encoded gender category: format start;end;category_id;conf - decode via NoXi_MetaData.xlsx
language.annotation.csv                        # Session language: format start;end;language_name;conf (e.g. "Japanese")


# Pre-extracted features - each is a .stream/.stream~ pair (see below)
{role}.audio.egemapsv2.stream(~)               # eGeMaps v2 hand-crafted acoustic features (dim=88)
{role}.audio.w2vbert2_embeddings.stream(~)     # W2v-BERT 2.0 speech embeddings (dim=1024)
{role}.audio.xlm_roberta_embeddings.stream(~)  # XLM-RoBERTa text embeddings (dim=768)
{role}.openface2.stream(~)                     # OpenFace2: facial landmarks, AUs, head pose, gaze
{role}.openpose.stream(~)                      # OpenPose body + 2x hands keypoints (dim=139: x,y,conf)
{role}.clip.stream(~)                          # CLIP visual embeddings (dim=512)
{role}.dino.stream(~)                          # DINO self-supervised ViT embeddings
{role}.imagebind.stream(~)                     # ImageBind multimodal embeddings
{role}.swin.stream(~)                          # Swin Transformer video embeddings
{role}.videomae.stream(~)                      # VideoMAE temporal video embeddings

{role}.audio.wav                               # Raw audio
{role}.video.mp4                               # Raw video
```

### The `.stream` / `.stream~` File Pair

Every feature stream comes as two files that always go together:
- **`.stream`** - an XML metadata header (SSI format) describing the binary content: sample rate (`sr`), number of dimensions (`dim`), data type (`type`), and byte offsets for each chunk.
- **`.stream~`** - the raw binary blob of feature data, meaningless without its header.

Example `.stream` header:
```xml
<stream ssi-v="2">
    <info ftype="BINARY" sr="25.0" dim="88" byte="4" type="FLOAT" delim=" " />
    <meta type="" />
    <chunk from="0.000" to="600.000" byte="0" num="15000" />
</stream>
```

### Input Normalisation

The raw streams live on wildly different scales - eGeMAPS reaches +-2.6e5 and OpenFace2 ±3e4, while W2v-BERT sits in +-4. Feeding those in unscaled makes activations grow by orders of magnitude through the network and training becomes useless.

`src/normalization.py` standardises every channel to zero mean / unit variance. Statistics are computed **once over the train split only** (never val/test, to avoid leakage), then cached to `data/norm_stats_{corpus}.npz` and reused. The cache is created automatically on first run; delete it to force recomputation after changing the active modality set.

### Data Access

To obtain the data, you must follow the official MultiMediate procedures:

1. Visit the [MultiMediate Dataset Page](https://multimediate-challenge.org/Dataset/).
2. Download and sign the relevant End User License Agreement (EULA).
3. Contact the dataset owners to request access:
   - **MPIIGroupInteraction:** Victor Oei (`victor.oei@vis.uni-stuttgart.de`)
   - **NoXi:** Daksitha Withanage Don (`noxi@hcai.eu`)
   - **NoXi+J:** Marius Funk (`noxi+j@hcai.eu`)

### Features & Streams

The datasets provide the following multi-modal streams:

* **Audio:** eGeMaps v2 (openSMILE), W2v-BERT 2.0 (1024-dim), and XLM-RoBERTa (768-dim) embeddings.
* **Video:** OpenFace2 landmarks, OpenPose (139-dim: body [x, y, confidence] + 2x hands [x, y]), and CLIP (512-dim) visual embeddings.
* **Text:** Audio transcripts with start/end timestamps and confidence scores.
* **Annotations:** Continuous 25Hz frame-wise engagement scores (for Train/Val sets).

## Environment Setup

This project uses **Python 3.14**. Every pinned dependency (`jax`, `numpy`, `pandas`, `scipy`, `flax`) requires Python **3.11 or newer**, so system Pythons older than that will not work.

We use [`uv`](https://docs.astral.sh/uv/) because it installs its own standalone Python into your home directory - no `sudo`, no system package manager, and no dependency on `ensurepip`/`python3-venv` being present. This matters on shared machines (e.g. the MICM cluster) where you do not have admin rights. Also, it is written in Rust and is much faster than its alternatives.

### 1. Install `uv` (once per machine)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH" 
uv --version
```

### 2. Create the Virtual Environment

```bash
uv python install 3.14                          # downloads CPython 3.14 into ~/.local/share/uv
uv venv --python 3.14 EngageNet_venv
source EngageNet_venv/bin/activate              # Linux/macOS
.\EngageNet_venv\Scripts\activate               # Windows
python --version                                # expect Python 3.14.x
```

### 3. Install Dependencies

```bash
uv pip install -r requirements.txt
```

Verify the GPU is visible (on machines that have one):

```bash
python -c "import jax; print(jax.__version__); print(jax.devices())"
```

Expect `[CudaDevice(id=0), ...]`. If it prints `[CpuDevice(id=0)]`, JAX fell back to CPU.

> **CPU-only machines:** `requirements.txt` pins `jax[cuda12]`, which pulls ~3 GB of NVIDIA wheels you will never use on a laptop without an NVIDIA GPU. For a local environment used only for editing, linting, and import checks, install the CPU build instead.

> **Note for Windows Users:** To use the **NOVA** tool for manual session annotation and visualization, please refer to the [NOVA repository](https://github.com/hcmlab/nova).

### Shared GPU Etiquette

On multi-user machines, pin yourself to one GPU and disable JAX's default 75% memory preallocation so others can still use the card:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=1
```

XLA prints alarming `bfc_allocator ... ran out of memory` warnings during autotuning. These are rejected candidate algorithms, **not** failures - if the script prints its results, everything worked.

## Repository Structure

> All scripts are run **from the repository root**. Both `src/` and `scripts/` rely on paths relative to the working directory (`sys.path.insert(0, "src")` and `data/...`), so `cd scripts && python find_nan.py` will fail.

```
├── data/                                     # Dataset root (gitignored)
│   ├── NoXi/
│   ├── NoXi+J/
│   ├── MPII/
│   └── norm_stats_{corpus}.npz               # Cached per-channel normalisation statistics
├── models/                                   # Checkpoints written by training (gitignored)
├── submissions/                              # Generated submission CSVs + results.json (gitignored)
├── Dockerfile.train/Dockerfile.inference    # Dockerfiles for training and inference
├── EngageNet_venv/                          # Python virtual environment
├── requirements.txt                         # For Docker
├── scripts/                                 # Standalone dev/debug tools - not imported by src/
│   ├── bench_step.py                        # Time train_step on synthetic batches (GPU cost vs data-loading cost)
│   ├── check_collapse.py                    # Compare prediction spread vs target spread on val
│   ├── check_data.py                        # Scan raw streams + labels for NaN/inf and value scales
│   ├── find_nan.py                          # Run one real batch and report where NaN/inf first appears
│   └── profile_load.py                      # Time disk read vs windowing for a single session
├── src/
│   ├── aggregator.py                        # Overlap-add window predictions into session time series
│   ├── beta_head.py                         # Beta regression heads (multimodal + per-modality)
│   ├── bimamba.py                           # BiMamba block + IntraModalBiMamba wrapper
│   ├── config.py                            # Centralised hyperparameters and paths
│   ├── data_loader.py                       # Batch-yielding generator over the dataset
│   ├── dataset.py                           # Lazy, memory-efficient window iterator
│   ├── evaluate.py                          # Score submission CSVs against ground-truth (CCC, CDD)
│   ├── inference.py                         # Multi-corpus TTA inference + submission CSV output
│   ├── init_encoder.py                      # Per-modality shallow 1-D CNN encoder
│   ├── inter_modal.py                       # Gumbel-Sinkhorn ordering + cross-modal BiMamba
│   ├── metrics.py                           # CCC, CDD_G, CDD_L metric implementations
│   ├── modality_frontend.py                 # Runs all InitEncoders + channel projections
│   ├── model.py                             # Full EngageNet wiring all modules together
│   ├── normalization.py                     # Per-channel input standardisation (train-split stats, cached)
│   ├── read_data.py                         # Low-level SSI stream / annotation readers
│   ├── ssm.py                               # Pure JAX selective state space scan primitive
│   ├── train.py                             # Training loop (Optax + orbax checkpointing)
│   └── tta.py                               # Test-time adaptation loop
├── tests/
│   ├── test_frontend.py                     # Smoke-test: end-to-end forward pass on CPU
│   ├── test_inter_modal.py                  # Smoke-test: Gumbel-Sinkhorn + cross-modal BiMamba
│   └── test_model.py                        # Smoke-test of the entire model
├── engineering_roadmap.md
└── README.md
```

## Testing

Run the smoke-test to verify the full pipeline (data loading -> encoding -> projection) works on CPU:

```bash
# All 10 feature streams (default)
python tests/test_frontend.py

# Only the 4 core engagement-relevant streams (faster, less memory)
python tests/test_frontend.py --core

# Specify a different split
python tests/test_frontend.py val
python tests/test_frontend.py val --core

# All 3 tests
python tests/test_frontend.py && python tests/test_inter_modal.py && python tests/test_model.py
```

The `--core` flag restricts processing to `eGeMaps v2`, `W2v-BERT 2.0`, `OpenFace2`, and `OpenPose` - the streams most relevant to engagement prediction. See `CORE_MODALITIES` in `src/config.py`.

## Development Scripts

Diagnostic tools in `scripts/`, all run from the repository root. They import from `src/` and read from `data/`, so they need an activated environment and the dataset in place.

```bash
# Where does time go loading one session? (disk read vs windowing)
python scripts/profile_load.py data/NoXi+J/train/086

# How fast is a training step, independent of disk I/O?
python scripts/bench_step.py

# Are the raw streams and labels clean? (NaN/inf, value ranges)
python scripts/check_data.py

# Run one real batch and find where NaN/inf first appears
python scripts/find_nan.py

# Did the model collapse to a constant? (prediction spread vs target spread)
python scripts/check_collapse.py
```

`check_collapse.py` loads `models/best`. A healthy model has a prediction std close to the target std; a collapsed model predicts nearly the same value everywhere, which scores near-zero CCC regardless of how confident it looks.

## Training

```bash
# Default settings (all 10 streams, 50 epochs, checkpoints every 10 epochs)
python src/train.py

# Override any hyperparameter via CLI
python src/train.py --active-modalities audio.egemapsv2 audio.w2vbert2_embeddings openface2 openpose \
                    --n-epochs 100 --lr 5e-4 --batch-size 16

# Full list of options
python src/train.py --help
```

Checkpoints are saved to `models/EngageNet_{epoch}` every `--checkpoint-every` epochs (default: 10). Training includes CCC-based validation every epoch with early stopping (patience=10). The best checkpoint is saved to `models/best/`.

> **Warning:** the checkpoint directory is fixed, so a new run **overwrites `models/best`** as soon as it beats its own first epoch. Back up any run you care about before starting another:
> ```bash
> cp -r models models_backup_$(date +%Y%m%d)
> ```

Long runs should go under `tmux` or `screen` so they survive a dropped SSH session:

```bash
tmux new -s train
# ... start training, then detach with Ctrl-b then d
tmux attach -t train
```

## Test-Time Adaptation

TTA is applied at inference on unlabelled test data. It selectively fine-tunes surgical layers (BatchNorm, InitEncoder conv1, first Dense in cross-modal BiMamba) using samples where the multimodal prediction is confident but individual modality heads disagree. No ground-truth labels are needed - the multimodal predictive mean serves as a pseudo-target.

Key functions in `src/tta.py`:
- `sample_filter` - selects high-value adaptation samples via percentile thresholds on multimodal vs. unimodal uncertainty
- `tta_loss` - mutual information sharing (KL divergence) + pseudo-label supervision
- `surgical_mask` - freezes all parameters except the designated surgical layers
- `tta_step` - single adaptation step with masked gradients

## Evaluation

Score predictions against ground-truth (works on val split where labels exist):

```bash
python src/evaluate.py --submission-dir submissions/ --data-root data/ --corpora NoXi NoXi+J --split val
```

Outputs per-corpus CCC, Combined CCC, and saves `submissions/results.json`.

Alongside CCC, the challenge requires two fairness metrics - **CDD_G** (gender) and **CDD_L** (language) - implemented in `src/metrics.py`. Both need the relevant subgroups to be present in the split being scored: CDD_L requires more than one language, and CDD_G requires more than one gender code. The NoXi+J val split satisfies both (Japanese sessions 121-126, Chinese sessions 143-146; both gender codes present).

## Docker

The repo provides two Dockerfiles for the same codebase - one for training, one for inference. Both share `requirements.txt` and the `src/` directory, but differ in what data they expect and what script they run.


The inference container expects a `models/` directory with at least one checkpoint produced by training. It automatically loads the checkpoint with the highest epoch number.

```bash
# Build
docker build -f Dockerfile.train -t engagenet-train .
docker build -f Dockerfile.inference -t engagenet-inference .

# Train (GPU required)
docker run --gpus all engagenet-train

# Train with custom params
docker run --gpus all engagenet-train python3.14 src/train.py --n-epochs 100 --lr 5e-4

# Inference with TTA (after training)
docker run --gpus all engagenet-inference
```

If `data/` is too large to copy into the image, mount it at runtime instead:

```bash
docker run --gpus all -v /path/to/data:/app/data engagenet-train
docker run --gpus all -v /path/to/data:/app/data -v /path/to/models:/app/models engagenet-inference
```


## Contributors

- **Research Team:** Lado Turmanidze, Luka Javakhisvhili, Mariam Gadelia, Keso Chikhladze
- **Supervisors:** Dr. Philipp Müller, Beso Mikaberidze
- **Institution:** Georgian Artificial Intelligence Association (GAIA)
