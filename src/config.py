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

    # Shared projection dim for inter-modal fusion
    shared_dim: int = 128          # C'

    # InitEncoder activation function (currently hardcoded in the module, but could be made configurable here)
    encoder_act: str = "silu"

    # Training
    batch_size: int = 8
    seed: int = 42

    # Derived helpers

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
