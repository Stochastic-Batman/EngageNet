# Engineering Roadmap

This markdown file describes the coding decisions we made after having an initial architecture, currently described in `documentation/EngageNet Draft.tex`. The architecture defines a multimodal BiMamba network with per-modality initial encoders, intra-modal and inter-modal BiMamba modules, Beta regression heads, test-time adaptation, and a differentiable Gumbel-Sinkhorn modality ordering mechanism.

## Implemented

### `src/config.py`

A single `@dataclass` that centralises every hyperparameter and path so nothing is hardcoded elsewhere. This is the only file you need to edit to change the corpus, window size, or per-modality encoder widths.

**`_DEFAULT_MODALITY_SPECS`** - a dictionary mapping each feature name to a 4-tuple `(C_i, C_i', kernel_size, stride)`. The `C_i` values (input channels) come directly from the dataset documentation and the `.stream` XML headers: for example, eGeMaps v2 outputs 88-dimensional acoustic descriptors, W2v-BERT 2.0 produces 1024-dimensional speech embeddings, OpenPose encodes body + 2x hand keypoints as 139 values (x, y, confidence per joint), and OpenFace2 packs 714 dimensions covering Action Units (facial muscle activation intensities), facial landmarks (2D/3D coordinates of 68 face points), gaze direction vectors, and head pose (rotation + translation). The `C_i'` values (encoder output width) are set to 64 for the two lowest-dimensional modalities (eGeMaps at 88 and OpenPose at 139) and 128 for everything else - this keeps the shallow encoder from being a bottleneck on small inputs while avoiding unnecessary width on already-compact features. Kernel size 5 and stride 1 are conservative defaults: kernel 5 gives a receptive field of 5 frames (200 ms at 25 Hz), which is enough to smooth frame-level noise without losing temporal resolution, and stride 1 preserves the original sequence length so output length equals input length.

**`target_sr: float = 25.0`** - the target sample rate in Hertz. All streams are resampled to this rate so that every modality shares the same temporal axis. 25 Hz was chosen because the engagement annotations are natively at 25 Hz (one label per video frame at 25 fps), so resampling to 25 Hz makes the labels and features align without any interpolation of the supervision signal.

**`window_len: int = 250`** - 250 frames at 25 Hz = 10 seconds. This is a common segment length in affective computing: long enough to capture meaningful engagement dynamics (a shift from engaged to disengaged typically takes several seconds) but short enough to fit comfortably in memory as a single batch element.

**`window_stride: int = 125`** - 50% overlap between consecutive windows. This doubles the effective number of training samples without loading more data from disk.

**`shared_dim: int = 128`** - the uniform projection dimension `C'` from the Order-Sensitivity Problem section of the draft. After each `InitEncoder_i` produces a hidden representation of shape `(L' by C_i')`, a learned linear projection maps it to shape `(L' by C')`. This is required by the Gumbel-Sinkhorn permutation mechanism in the inter-modal fusion module: you can only permute and soft-mix modality representations if they all live in the same vector space. 128 was chosen as a balance - large enough to not lose information from the 128-wide encoder outputs, small enough that the full concatenation across 20 modality-role pairs (20 times 128 = 2560 channels) does not blow up the inter-modal BiMamba state size.

**`active_modalities: Optional[list[str]]`** - controls which pre-extracted feature streams are actually loaded and processed. Defaults to `None` (all 10 streams). Set to `CORE_MODALITIES` to use only the 4 streams most relevant to engagement: eGeMaps v2 (prosodic features), W2v-BERT 2.0 (learned speech representations), OpenFace2 (facial AUs, landmarks, gaze, head pose), and OpenPose (body posture and hand gestures). The remaining 6 streams (XLM-RoBERTa, CLIP, DINOv2, ImageBind, Swin, VideoMAE) are either redundant with the core 4 or encode generic visual semantics that don't vary meaningfully across engagement levels. This filter propagates everywhere: the dataset skips loading unused `.stream` files from disk, and the `ModalityFrontend` only creates encoders for active streams. Note that this only affects the pre-extracted feature streams - annotation data (engagement labels, transcripts, age, gender, language) is always loaded regardless, since it comes from separate CSV files outside the stream pipeline.


### `src/dataset.py`

The lazy, memory-efficient data pipeline. The central class `EngageNetDataset` wraps a corpus split directory (e.g. `data/NoXi+J/train/`) and exposes an `iter_windows()` generator. Internally it iterates over session directories one at a time, calls `read_data.load_session()` to pull all streams for that session from disk, resamples everything to the target sample rate, clips to the shortest common temporal length, slices into fixed-length windows, and yields each window as a dictionary of NumPy arrays. Once all windows from a session are yielded, the session data falls out of scope and gets garbage-collected before the next session is loaded. This way, peak memory usage is proportional to one session (~1.5-2 GiB) rather than the full corpus (88+ GiB).

**Resampling** uses `scipy.interpolate.interp1d` with linear interpolation. This is simple and fast; for 25 Hz signals being resampled to 25 Hz it is a no-op.

**`_pad_or_trim`** ensures every window has exactly `window_len` frames. If the session's tail is shorter (i.e. the last window runs past the end of the data), we zero-pad. The `*arr.shape[1:]` unpacking in the padding constructor preserves all non-temporal dimensions (i.e. channels) so the zero-padding shape matches regardless of how many channels the modality has.

**Shuffling** uses a `jax.random.PRNGKey` to generate a permutation of session indices via `jax.random.permutation`. This keeps all randomness in the JAX PRNG system - explicit, splittable, and reproducible - rather than relying on NumPy's global random state.

**Output format** - each window's streams are transposed to channels-first shape `(C_i, L)` because the `InitEncoder` expects this layout (matching the paper's notation for input `x_i`).

### `src/data_loader.py`

A thin wrapper over `EngageNetDataset.iter_windows()` that accumulates windows into batches of size `cnfg.batch_size`, converts them to JAX arrays via `jnp.array(np.stack(...))`, and yields batched dictionaries. The `_collate` function stacks all numeric arrays along a new leading batch axis while keeping the `"session"` key as a plain Python list of strings (metadata, not a tensor). The last batch of an epoch may be smaller than `batch_size` - we yield it anyway rather than dropping data.

### `src/init_encoder.py`

The `InitEncoder` Flax module implements the shallow per-modality feature extractor described in Section 1.2.1 of the draft. It maps an input of shape `(C_i, L)` to a hidden representation of shape `(L', C_i')`.

Internally it is a single `nn.Conv` (1-D, kernel and stride from config) -> `nn.BatchNorm` -> SiLU activation. The convolution uses `padding="SAME"` so that with the default stride of 1, the output length equals the input length. The dimension transposition from channels-first input `(B, C_i, L)` to time-first output `(B, L', C_i')` is handled inside the module (Flax Conv expects channels-last).

BatchNorm is included because the draft's Section 2.4 (Surgical Fine-Tuning) specifically lists "BatchNorm layers" and "First convolution in InitEncoder" as the surgical layers updated during test-time adaptation. Having BatchNorm here makes TTA straightforward later: just unfreeze `batch_norm1` and `conv1` in the InitEncoder while keeping everything else frozen.

SiLU is the activation used throughout the Mamba architecture and is specified in the draft's BiMamba equations.

### `src/modality_frontend.py`

The top-level `ModalityFrontend` Flax module that owns all `InitEncoder` instances (one per feature type) and `nn.Dense` projection layers (one per feature type). For each feature type, it runs the same encoder on both the expert and novice inputs - weight sharing across roles. This is the right call because expert and novice streams for the same feature (e.g. both `expert.clip` and `novice.clip`) are extracted by the same pretrained model from the same type of input (video frames), so the shallow encoder should learn the same low-level patterns for both.

After each encoder produces a hidden representation of shape `(B, L', C_i')`, a per-feature `nn.Dense` projects it to the shared dimension, giving shape `(B, L', C')`. This is the uniform channel projection required so that the Gumbel-Sinkhorn permutation can treat all modality representations as interchangeable objects in the same vector space.

The output is a dictionary keyed by `"{role}.{feat}"` with all hidden representations ready for the BiMamba modules.

### `src/ssm.py`

A pure JAX module (no Flax, no `nn.Module`) implementing the selective state space scan that sits at the core of every Mamba block. `bimamba.py` owns all weight matrices and computes the input-dependent parameters; `ssm.py` is the low-level primitive it calls after those projections.

**`discretize(delta, A, B)`** converts continuous-time parameters to discrete-time ones. It computes the discretized transition matrix `A_bar` by exponentiating the element-wise product of `delta` and `A`, and the discretized input matrix `B_bar` as the element-wise product of `delta` and `B`. Both are broadcast to shape `(B, L, D, N)` via reshaping.

**`_scan_fn(left, right)`** is the associative combiner passed to `lax.associative_scan`. Each time step defines an affine map: "given the previous hidden state, produce the next one by scaling with `A_bar_t` and adding the input contribution `b_t = B_bar_t * u_t`." Two consecutive affine maps combine by multiplying their scaling factors and folding the earlier input contribution through the later scaling. The scan accumulates these combinations across all time steps so all hidden states are computed in `O(log L)` parallel steps rather than `O(L)` sequential steps. Since the initial state is zero, the accumulated input contribution at each position is directly the hidden state at that time step, which is why only the second element of the scan output is unpacked.

**`ssm(u, delta, A, B, C)`** is the main public function. It calls `discretize`, precomputes the input contribution at each step, transposes for the scan, calls `lax.associative_scan`, transposes back, and produces the output sequence by taking the element-wise product of the output projection `C` and the hidden states and summing over the state dimension.

### `src/bimamba.py`

**`BiMambaBlock`** implements the three steps from Section 1.2.1 of the draft. It takes a hidden representation of shape `(B, L', D)` and returns one of the same shape.

Step 1 is gating: a learned SiLU projection produces a gate tensor of the same shape that is later applied via element-wise multiplication to suppress noise and highlight relevant features.

Step 2 is a shared linear projection applied to the input before splitting into two branches.

Step 3 is the forward SSM branch: a depthwise Conv1D (each channel processed independently) with SiLU captures local context, then the three input-dependent selection parameters are computed - `delta` (timescale, via `softplus(s_delta + Linear(x))`), `B` (what to write to the state), and `C` (what to read from the state). `A` is a learned parameter stored as its negated log so that the discretized transition values always land in `(0, 1)` for stable state evolution. `ssm` from `ssm.py` is called with these parameters, and the result is multiplied element-wise with the gate.

Step 4 is the backward SSM branch: the shared projection is time-reversed before the depthwise conv, the same structure is applied with a separate set of weights, and the output is time-reversed back to the original order before gating.

Step 5 is the residual merge: the forward and backward outputs are averaged, projected with a linear layer, and added to the original input as a skip connection.

**`IntraModalBiMamba`** wraps a `BiMambaBlock` per feature type and applies it independently to every key in the hiddens dict. Weight sharing mirrors `ModalityFrontend`: expert and novice streams for the same feature type share one `BiMambaBlock`. Output shape is `(B, L', C')` per modality, identical to the input.

### `tests/test_frontend.py`

A smoke-test script that loads one real window from the first training session, wraps it in a batch-of-1, initialises the `ModalityFrontend` on CPU with a JAX PRNGKey, runs a forward pass, and prints all input and output shapes plus the total parameter count. This confirms end-to-end correctness (data loading -> resampling -> windowing -> encoding -> projection) without needing a GPU or the full dataset.


## TODO

### `src/inter_modal.py`

Two distinct things in one file. First, the Gumbel-Sinkhorn meta-network: each modality representation is summarised by temporal mean-pooling, then asymmetric query and key projections produce an `M by M` score matrix where entry `(i, j)` scores the fitness of placing modality `j` at sequence position `i`. Independent Gumbel noise scaled by temperature `tau` is added to enable stochastic exploration of orderings during training. Applying iterative Sinkhorn-Knopp normalisation to the elementwise exponential of the perturbed scores yields a doubly-stochastic matrix `P` (rows and columns both sum to one), and the soft reordering is the `P`-weighted convex combination of the modality representations. Second, the reordered representations are concatenated and transposed so the fused channel axis becomes the sequence axis, and a `BiMambaBlock` is applied across modalities. Expose `tau` so the training loop can anneal it with the exponential decay schedule from the draft.

Add `tests/test_inter_modal.py`: verify `P` is doubly stochastic (row and column sums close to 1), and verify the equivariance guarantee by permuting the input modalities and checking the output representation is unchanged.

### `src/beta_head.py`

A `BetaHead` Flax module: mean-pool over the time axis, then two `nn.Dense` layers producing raw logits for alpha and beta, followed by `softplus + 1` to get valid shape parameters both greater than one (ensuring a unimodal distribution). Add static methods for `predictive_mean` (alpha divided by alpha plus beta), `predictive_variance` (the TTA uncertainty proxy: alpha times beta divided by the square of their sum times their sum plus one), and `nll_loss` (Beta negative log-likelihood). Then write `MultiHeadBeta` that instantiates one `BetaHead` per modality key (the unimodal heads fed directly by each per-modality output of `IntraModalBiMamba`) plus one for the fused inter-modal output - all needed by the TTA sample selection filter.

### `src/model.py`

The full `EngageNet` Flax module that wires everything together in order: `ModalityFrontend` -> `IntraModalBiMamba` -> `InterModalBiMamba` -> `MultiHeadBeta`. Returns a dict with the multimodal prediction (alpha and beta for the fused output) and all unimodal predictions (alpha and beta per modality). The unimodal outputs are needed at train time for the TTA loss but are cheap to compute here since the unimodal heads share the already-computed per-modality representations.

Add `tests/test_model.py`: full forward pass smoke test analogous to the existing `test_frontend.py`.

### `src/train.py`

Standard Flax + Optax training loop. Create a `TrainState` with an AdamW optimiser. The loss is Beta negative log-likelihood on the multimodal head. Iterate via `data_loader.iter_batches()`, call `jax.jit`-compiled train and eval steps, log metrics, and checkpoint with `orbax-checkpoint`. Handle temperature annealing for Gumbel-Sinkhorn: decay `tau` each epoch starting from an initial exploration temperature down to a near-deterministic floor via exponential decay.

### `src/tta.py`

Implement the TTA loop. The two-level sample filter keeps samples where the multimodal uncertainty is low (the fused prediction is confident) but the weighted average of unimodal uncertainties is high (individual modalities disagree), using adaptive percentile thresholds maintained over a moving window of the most recent `K` frames. The mutual information sharing loss is the sum of closed-form KL divergences between each unimodal Beta distribution and the multimodal Beta distribution. The total TTA loss combines this with a pseudo-label supervision term that treats the multimodal predictive mean as a target for each unimodal head, weighted by a hyperparameter `lambda`. Surgical fine-tuning: a helper that takes a `TrainState` and returns a masked gradient update that only touches BatchNorm parameters, `conv1` in each `InitEncoder`, and the first linear layer in the inter-modal BiMamba - everything else gets a zero gradient.
