"""kv_evict: value-aware, block-wise KV-cache eviction for vLLM.

Stage 1 (this commit): pure, GPU-free, vLLM-free scoring primitives.
Later stages add the eviction policy (Stage 2) and the vLLM integration
(Stage 4+). Nothing here imports vLLM or touches a GPU.
"""

from .manager import BlockEvictionManager, EvictionConfig
from .scoring import DEFAULT_EPS, score_blocks

__all__ = [
    "score_blocks",
    "DEFAULT_EPS",
    "BlockEvictionManager",
    "EvictionConfig",
]
