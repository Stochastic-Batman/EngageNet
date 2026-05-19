# Engineering Roadmap

This markdown file describes the coding decisions we made after having an initial architecture, currently described in `documentation/EngageNet Draft.tex`. The architecture defines a multimodal BiMamba network with per-modality initial encoders, intra-modal and inter-modal BiMamba modules, Beta regression heads, test-time adaptation, and a differentiable Gumbel-Sinkhorn modality ordering mechanism. Before any of that can be built, we need a solid data pipeline and the first transformation layer - the `InitEncoder_i` that produces hidden representations `h_i` from raw feature streams. That is the scope of this first implementation phase.

## Overall Coding Roadmap

The NoXi+J dataset alone is 88.2 GiB of pre-extracted feature streams across 50 sessions, with 10 feature modalities per role (expert and novice) per session. Neither my laptop (HP Elitebook) nor potentially the data centre machines can hold all of this in RAM at once. So the very first engineering constraint is: **nothing loads the full dataset into memory**. Every piece of the pipeline must stream from disk, process one session (or one window within a session) at a time, and discard it before moving on.

The second constraint is hardware portability. I test on my local machine which has no GPU - only CPU. Training happens on a remote data centre node with NVIDIA GPUs accessed over SSH. The code must run identically on both. All randomness uses `jax.random` PRNGKeys rather than NumPy's `np.random` generators so that the random state is explicit, reproducible, and compatible with JAX's functional paradigm (no hidden global state).

## Phase 1: Data Pipeline and InitEncoder Frontend (Implemented)

### `src/config.py`

A single `@dataclass` that centralises every hyperparameter and path so nothing is hardcoded elsewhere. This is the only file you need to edit to change the corpus, window size, or per-modality encoder widths.

**`_DEFAULT_MODALITY_SPECS`** - a dictionary mapping each feature name to a 4-tuple `(C_i, C_i', kernel_size, stride)`. The `C_i` values (input channels) come directly from the dataset documentation and the `.stream` XML headers: for example, eGeMaps v2 outputs 88-dimensional acoustic descriptors, W2v-BERT 2.0 produces 1024-dimensional speech embeddings, OpenPose encodes body + 2× hand keypoints as 139 values (x, y, confidence per joint), and OpenFace2 packs 714 dimensions covering Action Units (facial muscle activation intensities), facial landmarks (2D/3D coordinates of 68 face points), gaze direction vectors, and head pose (rotation + translation). The `C_i'` values (encoder output width) are set to 64 for the two lowest-dimensional modalities (eGeMaps at 88 and OpenPose at 139) and 128 for everything else - this keeps the shallow encoder from being a bottleneck on small inputs while avoiding unnecessary width on already-compact features. Kernel size 5 and stride 1 are conservative defaults: kernel 5 gives a receptive field of 5 frames (200 ms at 25 Hz), which is enough to smooth frame-level noise without losing temporal resolution, and stride 1 preserves the original sequence length so `L' = L`.

**`target_sr: float = 25.0`** - the target sample rate in Hertz. All streams are resampled to this rate so that every modality shares the same temporal axis. 25 Hz was chosen because the engagement annotations are natively at 25 Hz (one label per video frame at 25 fps), so resampling to 25 Hz makes the labels and features align without any interpolation of the supervision signal.

**`window_len: int = 250`** - 250 frames at 25 Hz = 10 seconds. This is a common segment length in affective computing: long enough to capture meaningful engagement dynamics (a shift from engaged to disengaged typically takes several seconds) but short enough to fit comfortably in memory as a single batch element.

**`window_stride: int = 125`** - 50% overlap between consecutive windows. This doubles the effective number of training samples without loading more data from disk.

**`shared_dim: int = 128`** - the uniform projection dimension `C'` from Order-Sensitivity Problem Section of the draft. After each `InitEncoder_i` produces `h_i \in R^{L' × C_i'}`, a learned linear projection maps it to `R^{L' × C'}`. This is required by the Gumbel-Sinkhorn permutation mechanism in the inter-modal fusion module: you can only permute and soft-mix modality representations if they all live in the same vector space. 128 was chosen as a balance - large enough to not lose information from the 128-wide encoder outputs, small enough that the concatenation `M × C' = 20 × 128 = 2560` doesn't blow up the inter-modal BiMamba state size.

**`active_modalities: Optional[List[str]]`** - controls which pre-extracted feature streams are actually loaded and processed. Defaults to `None` (all 10 streams). Set to `CORE_MODALITIES` to use only the 4 streams most relevant to engagement: eGeMaps v2 (prosodic features), W2v-BERT 2.0 (learned speech representations), OpenFace2 (facial AUs, landmarks, gaze, head pose), and OpenPose (body posture and hand gestures). The remaining 6 streams (XLM-RoBERTa, CLIP, DINOv2, ImageBind, Swin, VideoMAE) are either redundant with the core 4 or encode generic visual semantics that don't vary meaningfully across engagement levels. This filter propagates everywhere: the dataset skips loading unused `.stream` files from disk, and the `ModalityFrontend` only creates encoders for active streams. Note that this only affects the pre-extracted feature streams - annotation data (engagement labels, transcripts, age, gender, language) is always loaded regardless, since it comes from separate CSV files outside the stream pipeline.


### `src/dataset.py`

The lazy, memory-efficient data pipeline. The central class `EngageNetDataset` wraps a corpus split directory (e.g. `data/NoXi+J/train/`) and exposes an `iter_windows()` generator. Internally it iterates over session directories one at a time, calls `read_data.load_session()` to pull all streams for that session from disk, resamples everything to the target sample rate, clips to the shortest common temporal length, slices into fixed-length windows, and yields each window as a dictionary of NumPy arrays. Once all windows from a session are yielded, the session data falls out of scope and gets garbage-collected before the next session is loaded. This way, peak memory usage is proportional to one session (~1.5-2 GiB) rather than the full corpus (88+ GiB).

**Resampling** uses `scipy.interpolate.interp1d` with linear interpolation. This is simple and fast; for 25 Hz signals being resampled to 25 Hz it is a no-op.

**`_pad_or_trim`** ensures every window has exactly `window_len` frames. If the session's tail is shorter (i.e. the last window runs past the end of the data), we zero-pad. The `*arr.shape[1:]` unpacking in the padding constructor preserves all non-temporal dimensions (i.e. channels) so the zero-padding shape matches regardless of how many channels the modality has.

**Shuffling** uses a `jax.random.PRNGKey` to generate a permutation of session indices via `jax.random.permutation`. This keeps all randomness in the JAX PRNG system - explicit, splittable, and reproducible - rather than relying on NumPy's global random state.

**Output format** - each window's streams are transposed to channels-first `(C_i, L)` because the `InitEncoder` expects this layout (matching the paper's notation `x_i \in R^{C_i × L}`).

### `src/data_loader.py`

A thin wrapper over `EngageNetDataset.iter_windows()` that accumulates windows into batches of size `cnfg.batch_size`, converts them to JAX arrays via `jnp.array(np.stack(...))`, and yields batched dictionaries. The `_collate` function stacks all numeric arrays along a new leading batch axis while keeping the `"session"` key as a plain Python list of strings (metadata, not a tensor). The last batch of an epoch may be smaller than `batch_size` - we yield it anyway rather than dropping data.

### `src/init_encoder.py`

The `InitEncoder` Flax module implements the shallow per-modality feature extractor described in Section 1.2.1 of the draft: `h_i = InitEncoder_i(x_i) \in R^{L' × C_i'}`.

Internally it is a single `nn.Conv` (1-D, kernel and stride from config) -> `nn.BatchNorm` -> SiLU activation. The convolution uses `padding="SAME"` so that with the default stride of 1, the output length `L'` equals the input length `L`. The dimension transposition from the paper's input layout `(C_i, L)` to the output layout `(L', C_i')` is handled inside the module: the input `(B, C_i, L)` is transposed to `(B, L, C_i)` (Flax Conv expects channels-last), convolved to `(B, L', C_i')`, and returned as-is - already time-first.

BatchNorm is included because the draft's Section 2.4 (Surgical Fine-Tuning) specifically lists "BatchNorm layers" and "First convolution in InitEncoder" as the surgical layers updated during test-time adaptation. Having BatchNorm here makes TTA straightforward later: just unfreeze `batch_norm1` and `conv1` in the InitEncoder while keeping everything else frozen.

SiLU is the activation used throughout the Mamba architecture and is specified in the draft's BiMamba equations.

### `src/modality_frontend.py`

The top-level `ModalityFrontend` Flax module that owns all `InitEncoder` instances (one per feature type) and `nn.Dense` projection layers (one per feature type). For each feature type, it runs the same encoder on both the expert and novice inputs - weight sharing across roles. This is the right call because expert and novice streams for the same feature (e.g. both `expert.clip` and `novice.clip`) are extracted by the same pretrained model from the same type of input (video frames), so the shallow encoder should learn the same low-level patterns for both.

After each encoder produces `h_i \in R^{B × L' × C_i'}`, a per-feature `nn.Dense` projects it to the shared dimension: `h_i \in R^{B × L' × C'}`. This is the uniform channel projection `f_i` from Order-Sensitivity Section of the draft, required so that the Gumbel-Sinkhorn permutation can treat all modality representations as interchangeable objects in the same vector space.

The output is a dictionary `{"{role}.{feat}": h_i}` with all hidden representations ready for the BiMamba modules.

### `tests/test_frontend.py`

A smoke-test script that loads one real window from the first training session, wraps it in a batch-of-1, initialises the `ModalityFrontend` on CPU with a JAX PRNGKey, runs a forward pass, and prints all input and output shapes plus the total parameter count. This confirms end-to-end correctness (data loading -> resampling -> windowing -> encoding -> projection) without needing a GPU or the full dataset.

## We are done with raw data handling, time for the fun architecture implementation!

From this point forward, every modality is a uniform `(B, L', C')` = `(B, 250, 128)` tensor. The `ModalityFrontend` has absorbed all per-modality heterogeneity (different input dimensions, different sample rates, different data types). I never need to think about raw `C_i` values again - I just work with the `h_i` dictionary.

### Phase 2: Intra-modal BiMamba

For each modality key in the `h_i` dictionary, apply the BiMamba block described in Section 1.2.1 of the draft:

1. **Gating**: `g_i = SiLU(W_g @ h_i + b_g)` - a learned linear projection + SiLU to highlight emotion-relevant information.
2. **Forward SSM**: `Conv1D -> SiLU -> SelectiveSSM` on `h_i`, then Hadamard product with `g_i`.
3. **Backward SSM**: same as forward but on `Rev_t(h_i)`, then reverse the output back.
4. **Residual merge**: `u_i = h_i + W_o @ mean(forward, reversed_backward) + b_o`.

The output `u_i` has the same shape `(B, L', C')` for every modality. This is where the Manifold-Constrained Hyper-Connections (mHC) improvement from the draft could replace the standard residual connection.

The Selective SSM itself needs an implementation of the Mamba scan - either a custom JAX `lax.associative_scan` or a port of an existing Mamba kernel. On CPU this will be slow but functionally correct for testing; on GPU the parallel scan should be efficient.

### Phase 3: Inter-modal BiMamba (with Gumbel-Sinkhorn ordering)

1. **Concatenate** all `u_i` along the channel axis: `U = u_1 \circ u_2 \circ ... \circ u_M \in R^{L' x MC'}`.
2. **Gumbel-Sinkhorn meta-network**: compute the score matrix `Z` from temporal mean-pooled `u_i` via query-key projections, perturb with Gumbel noise, run Sinkhorn normalisation to get a doubly-stochastic permutation matrix `P`, and apply the soft reordering.
3. **Transpose** the reordered concatenation so the fused channel axis becomes the sequence dimension: `m_tilde \in R^{MC' x L'}`.
4. **Apply another BiMamba block** (same architecture as intra-modal, but now modelling cross-modal dependencies along the `MC'` axis).

The output `H \in R^{MC' x L'}` captures bidirectional inter-modal context.

### Phase 4: Beta regression head

1. **Temporal pooling** over `L'` (e.g. mean-pool or learned attention pooling) to get a fixed-size representation.
2. **Two-head output**: two linear projections producing raw logits `z_alpha` and `z_beta`, then `alpha = softplus(z_alpha) + 1`, `beta = softplus(z_beta) + 1` to parameterise a unimodal Beta distribution.
3. **Training loss**: Beta negative log-likelihood `L_NLL`.
4. **Predictive mean**: `mu = alpha / (alpha + beta)`.
5. **Predictive variance** (uncertainty proxy): `alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1))`.

Need both a multimodal head (fed by inter-modal BiMamba output) and per-modality unimodal heads (fed by each `u_i` independently) for the TTA selection filter.

### Phase 5: Training loop

1. Standard Flax training state with Optax optimiser.
2. Iterate over `data_loader.iter_batches()` - already lazy and memory-efficient.
3. Forward pass through `ModalityFrontend -> Intra-modal BiMamba -> Inter-modal BiMamba -> Beta heads`.
4. Beta NLL loss + any regularisation.
5. Checkpoint saving via `orbax-checkpoint` (already installed).

### Phase 6: Test-Time Adaptation (TTA)

1. **Uncertainty-based sample selection**: compute `U_multi` and `U_i` per sample, apply the adaptive percentile thresholds from the draft.
2. **TTA loss**: `L_TTA = L_mis + lambda * sum(NLL_Beta)` - KL divergence between unimodal and multimodal Beta distributions plus pseudo-label supervision.
3. **Surgical fine-tuning**: only update BatchNorm layers, `conv1` in InitEncoder, and the first FC layer in inter-modal BiMamba. Freeze everything else.
4. **Online adaptation**: process test frames sequentially, maintaining the moving window of recent uncertainties for adaptive thresholds.
