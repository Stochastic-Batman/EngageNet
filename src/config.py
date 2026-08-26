import argparse

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Per-modality metadata
# (input_channels C_i, output_channels C_i', kernel_size, stride)
# C_i values come from the dataset documentation / .stream headers.
# C_i' are design choices (shallow encoder width).

_DEFAULT_MODALITY_SPECS: dict[str, tuple[int, int, int, int]] = {
    "audio.egemapsv2":              (88,   64,  5, 1),
    "audio.w2vbert2_embeddings":    (1024, 128, 5, 1),
    "audio.xlm_roberta_embeddings": (768,  128, 5, 1),
    "openface2":                    (714,  128, 5, 1),   # Action Units + facial landmarks + gaze direction + head pose
    "openpose":                     (139,  64,  5, 1),
    "clip":                         (512,  128, 5, 1),
    "dino":                         (2304, 128, 5, 1),
    "imagebind":                    (1024, 128, 5, 1),
    "swin":                         (768,  128, 5, 1),
    "videomae":                     (1408, 128, 5, 1),
}

# The 4 streams most relevant to engagement prediction
CORE_MODALITIES: list[str] = ["audio.egemapsv2", "audio.w2vbert2_embeddings", "openface2", "openpose"]


@dataclass
class EngageNetConfig:
    """A single hadrcoded class to rule them all"""

    # Paths
    data_root: Path = Path(__file__).resolve().parent.parent / "data"
    corpus: str = "NoXi+J"

    # Temporal
    target_sr: float = 25.0        # resample every stream to this sample rate (Hz)
    window_len: int = 250          # frames per window  (= 10 s @ 25 Hz)
    window_stride: int = 125       # hop between windows (= 5 s, 50 % overlap)

    # Modality specs  (name -> (C_i, C_i', kernel, stride))
    # This is the full catalogue; use active_modalities to select a subset.
    modality_specs: dict[str, tuple[int, int, int, int]] = field(default_factory=lambda: dict(_DEFAULT_MODALITY_SPECS))

    # Which modalities to actually use. None = all, or pass a list of keys.
    # Example: active_modalities=CORE_MODALITIES for the lean 4-stream setup.
    active_modalities: Optional[list[str]] = None

    normalize_inputs: bool = True

    # Shared projection dim for inter-modal fusion
    shared_dim: int = 128          # C'

    # Training
    batch_size: int = 8
    seed: int = 95
    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 50
    tau_init: float = 1.0
    tau_min: float = 0.1
    tau_decay: float = 0.95
    checkpoint_every: int = 10
    checkpoint_dir: Path = Path(__file__).resolve().parent.parent / "models"
    eval_every: int = 1
    patience: int = 10
    submission_dir: Path = Path(__file__).resolve().parent.parent / "submissions"

    # Training objective
    lambda_ccc: float = 1.0        # weight of the ordering (CCC) term
    lambda_uni: float = 0.5        # per-modality supervision, divided by M internally
    lambda_fair: float = 0.05      # fairness penalty weight
    beta_w: float = 0.5            # likelihood reweighting exponent, kappa^{-beta_w}
    cdd_bins: int = 10             # ground-truth quantile bins for CDD and its penalty
    
    # Override the split inference.py writes predictions for (None = each corpus's default).
    # Use "val" to produce predictions that evaluate.py can actually score.
    submission_split: Optional[str] = None

    # SSM / BiMamba
    ssm_state_dim: int = 16        # N - state dimension in selective SSM
    conv_kernel: int = 4           # D_C -  depthwise conv kernel in BiMamba
    dt_min: float = 1e-3           # smallest delta at initialisation
    dt_max: float = 1e-1           # largest delta at initialisation
    cross_modal_conv_kernel: int = 2   # D_C for the fusion block (scan axis is M, not L')

    # Gumbel-Sinkhorn
    gs_dim: int = 64               # query/key projection dim for score matrix
    gs_iters: int = 20             # Sinkhorn normalisation iterations

    # Beta head
    beta_hidden: int = 128         # hidden layer width in BetaHead
    
    # The specs for only the active modalities (filters modality_specs by active_modalities)
    @property
    def active_specs(self) -> dict[str, tuple[int, int, int, int]]:
        if self.active_modalities is None:
            return self.modality_specs
        return {k: self.modality_specs[k] for k in self.active_modalities}

    # Full path to the active corpus directory (e.g. data/NoXi+J)
    @property
    def corpus_root(self) -> Path:
        return self.data_root / self.corpus

    # Full path to a specific split inside the corpus (e.g. data/NoXi+J/train)
    def split_dir(self, split: str) -> Path:
        return self.corpus_root / split

    # Flat list of active modality keys for iteration
    @property
    def modality_names(self) -> list[str]:
        return list(self.active_specs.keys())

    # Convenience accessors into the per-modality (C_i, C_i', kernel, stride) tuples
    def input_dim(self, mod: str) -> int:
        return self.active_specs[mod][0]

    def output_dim(self, mod: str) -> int:
        return self.active_specs[mod][1]

    def kernel_size(self, mod: str) -> int:
        return self.active_specs[mod][2]

    def stride(self, mod: str) -> int:
        return self.active_specs[mod][3]


    @classmethod
    def from_cli(cls) -> "EngageNetConfig":
        parser = argparse.ArgumentParser()
        parser.add_argument("--corpus", type=str, default="NoXi+J")
        parser.add_argument("--target-sr", type=float, default=25.0)
        parser.add_argument("--window-len", type=int, default=250)
        parser.add_argument("--window-stride", type=int, default=125)
        parser.add_argument("--active-modalities", nargs="*", default=None)
        parser.add_argument("--no-normalize-inputs", dest="normalize_inputs", action="store_false")
        parser.add_argument("--shared-dim", type=int, default=128)
        parser.add_argument("--batch-size", type=int, default=8)
        parser.add_argument("--seed", type=int, default=95)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight-decay", type=float, default=1e-4)
        parser.add_argument("--n-epochs", type=int, default=50)
        parser.add_argument("--tau-init", type=float, default=1.0)
        parser.add_argument("--tau-min", type=float, default=0.1)
        parser.add_argument("--tau-decay", type=float, default=0.95)
        parser.add_argument("--checkpoint-every", type=int, default=10)
        parser.add_argument("--eval-every", type=int, default=1)
        parser.add_argument("--patience", type=int, default=10)
        parser.add_argument("--submission-dir", type=str, default=None)
        parser.add_argument("--submission-split", type=str, default=None)
        parser.add_argument("--ssm-state-dim", type=int, default=16)
        parser.add_argument("--conv-kernel", type=int, default=4)
        parser.add_argument("--gs-dim", type=int, default=64)
        parser.add_argument("--gs-iters", type=int, default=10)
        parser.add_argument("--beta-hidden", type=int, default=128)
        parser.add_argument("--checkpoint-dir", type=str, default=None)
        parser.add_argument("--lambda-ccc", type=float, default=1.0)
        parser.add_argument("--lambda-uni", type=float, default=0.5)
        parser.add_argument("--lambda-fair", type=float, default=0.05)
        parser.add_argument("--beta-w", type=float, default=0.5)
        parser.add_argument("--cdd-bins", type=int, default=10)
        parser.add_argument("--dt-min", type=float, default=1e-3)
        parser.add_argument("--dt-max", type=float, default=1e-1)
        parser.add_argument("--cross-modal-conv-kernel", type=int, default=2)
        args = parser.parse_args()

        return cls(
            corpus=args.corpus,
            target_sr=args.target_sr,
            window_len=args.window_len,
            window_stride=args.window_stride,
            active_modalities=args.active_modalities,
            normalize_inputs=args.normalize_inputs,
            shared_dim=args.shared_dim,
            batch_size=args.batch_size,
            seed=args.seed,
            lr=args.lr,
            weight_decay=args.weight_decay,
            n_epochs=args.n_epochs,
            tau_init=args.tau_init,
            tau_min=args.tau_min,
            tau_decay=args.tau_decay,
            checkpoint_every=args.checkpoint_every,
            eval_every=args.eval_every,
            patience=args.patience,
            submission_dir=Path(args.submission_dir) if args.submission_dir else cls.submission_dir,
            submission_split=args.submission_split,
            ssm_state_dim=args.ssm_state_dim,
            conv_kernel=args.conv_kernel,
            gs_dim=args.gs_dim,
            gs_iters=args.gs_iters,
            beta_hidden=args.beta_hidden,
            checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else cls.checkpoint_dir,
            lambda_ccc=args.lambda_ccc,
            lambda_uni=args.lambda_uni,
            lambda_fair=args.lambda_fair,
            beta_w=args.beta_w,
            cdd_bins=args.cdd_bins,
            dt_min=args.dt_min,
            dt_max=args.dt_max,
            cross_modal_conv_kernel=args.cross_modal_conv_kernel,
        )
