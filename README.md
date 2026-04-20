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
{role}.engagement.annotation.csv               # 25Hz frame-wise engagement score in [0,1], one float per line — train/val only
{role}.audio.transcript.annotation.csv         # Speech transcript: start_time;end_time;content;confidence
{role}.age.annotation.csv                      # Encoded age category: format start;end;category_id;conf — decode via NoXi_MetaData.xlsx
{role}.gender.annotation.csv                   # Encoded gender category: format start;end;category_id;conf — decode via NoXi_MetaData.xlsx
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

This project uses **Python 3.14**. Use the following instructions to set up the dedicated virtual environment and install necessary dependencies.

### 1. Create the Virtual Environment

```bash
python3.14 -m venv MultiModalSocialAnalysisBehaviour_venv
source MultiModalSocialAnalysisBehaviour_venv/bin/activate  # Linux/macOS
.\MultiModalSocialAnalysisBehaviour_venv\Scripts\activate   # Windows
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install numpy pandas matplotlib scipy tqdm torch torchvision torchaudio transformers
```

> **Note for Windows Users:** To use the **NOVA** tool for manual session annotation and visualization, please refer to the [NOVA repository](https://github.com/hcmlab/nova).

## Repository Structure

```
├── data/                                     # Dataset root (gitignored)
│   ├── NoXi/
│   ├── NoXi+J/
│   └── MPII/
├── MultiModalSocialAnalysisBehaviour_venv/  # Python virtual environment
├── src/
│   └── read_data.py                         # Reading Data
└── README.md
```

## Contributors

- **Research Team:** Lado Turmanidze, Luka Javakhisvhili, Mariam Gadelia, Keso Chikhladze
- **Supervisors:** Dr. Philipp Müller, Beso Mikaberidze
- **Institution:** Georgian Artificial Intelligence Association (GAIA)
