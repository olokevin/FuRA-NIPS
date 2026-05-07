"""FuRA: Full-Rank Adaptation via lossless Block Tensor-Train factorization.

Shared layer implementations and utilities used across all experiment tracks
(commonsense SFT, RL, VLM, QFuRA).
"""

from .btt_layer import (
    BTTLayer,
    convert_linear_to_btt,
    configure_blocktt_trainability,
)
from .svd_layer import (
    SVDLayer,
    convert_linear_to_svd,
    configure_svd_trainability,
)

__all__ = [
    "BTTLayer",
    "SVDLayer",
    "convert_linear_to_btt",
    "convert_linear_to_svd",
    "configure_blocktt_trainability",
    "configure_svd_trainability",
]
