"""Block-wise KV-eviction policy (Stage 2): isolated, CPU-only, no vLLM.

The manager decides *when* to evict and *which* block to evict, but never
touches a real KV cache -- that wiring is Stage 4. It is deliberately a pure
bookkeeping object so it can be unit-tested without a GPU or a model.

Two responsibilities:

  1. should_evict(seq_id, total_len): fire ONLY at a block boundary
     (a block just filled, i.e. total_len % block_size == 0) AND when the
     sequence is over its per-seq block budget. Never mid-block.

  2. pick_block(seq_id, total_len, scores): argmin over importance scores,
     EXCLUDING protected blocks -- the first ``num_sink_blocks`` (attention
     sinks) and the last ``num_local_blocks`` (recent/local window).

Block accounting per sequence:

    blocks_in_cache = ceil(total_len / block_size) - blocks_evicted_so_far

so each ``block_size`` new tokens adds one block, and each recorded eviction
removes one. Lengths are passed in explicitly (caller reads them fresh from
vLLM each step) rather than tracked statelessly, to avoid hidden-state drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections.abc import Sequence


@dataclass
class EvictionConfig:
    """All knobs are config-driven (project constraint #6)."""

    block_size: int
    kv_budget: int          # max KV blocks retained per sequence
    num_sink_blocks: int = 0   # first N blocks always kept (attention sinks)
    num_local_blocks: int = 0  # last N blocks always kept (recent window)
    metric: str = "vk_ratio"   # "vk_ratio" (||V||/||K||) or "value_attention"

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if self.kv_budget <= 0:
            raise ValueError(f"kv_budget must be > 0, got {self.kv_budget}")
        if self.num_sink_blocks < 0 or self.num_local_blocks < 0:
            raise ValueError("num_sink_blocks / num_local_blocks must be >= 0")
        # We must always have at least one evictable block at the moment
        # eviction fires (blocks_in_cache == kv_budget + 1). Evictable count
        # there is kv_budget + 1 - (sink + local) >= 1  <=>  budget >= sink+local.
        protected = self.num_sink_blocks + self.num_local_blocks
        if self.kv_budget < protected:
            raise ValueError(
                f"kv_budget ({self.kv_budget}) must be >= num_sink_blocks + "
                f"num_local_blocks ({protected}); otherwise no block is ever "
                f"evictable."
            )


@dataclass
class _SeqState:
    blocks_evicted: int = 0


class BlockEvictionManager:
    """Per-sequence block-eviction bookkeeping."""

    def __init__(self, config: EvictionConfig) -> None:
        self.config = config
        self._seqs: dict[int, _SeqState] = {}

    # -- lifecycle ---------------------------------------------------------
    def register_seq(self, seq_id: int) -> None:
        self._seqs[seq_id] = _SeqState()

    def deregister_seq(self, seq_id: int) -> None:
        self._seqs.pop(seq_id, None)

    def is_registered(self, seq_id: int) -> bool:
        return seq_id in self._seqs

    # -- accounting --------------------------------------------------------
    def num_blocks(self, seq_id: int, total_len: int) -> int:
        """Physical KV blocks currently held for the sequence."""
        st = self._require(seq_id)
        full = math.ceil(total_len / self.config.block_size)
        return full - st.blocks_evicted

    def protected_indices(self, num_blocks: int) -> set[int]:
        """Block indices that must never be evicted, given a block count."""
        c = self.config
        sinks = range(0, min(c.num_sink_blocks, num_blocks))
        local_start = max(0, num_blocks - c.num_local_blocks)
        local = range(local_start, num_blocks)
        return set(sinks) | set(local)

    # -- policy ------------------------------------------------------------
    def should_evict(self, seq_id: int, total_len: int) -> bool:
        """True iff at a completed-block boundary AND over budget."""
        self._require(seq_id)
        at_boundary = total_len > 0 and total_len % self.config.block_size == 0
        if not at_boundary:
            return False
        return self.num_blocks(seq_id, total_len) > self.config.kv_budget

    def pick_block(
        self, seq_id: int, total_len: int, scores: Sequence[float]
    ) -> int:
        """Return the index of the lowest-importance NON-protected block.

        ``scores[i]`` is the importance of the i-th currently-held block
        (lower = evict first). Length must equal ``num_blocks`` -- a mismatch
        means the caller's view of the cache desynced, so we raise rather than
        evict the wrong block.
        """
        n = self.num_blocks(seq_id, total_len)
        scores = _to_float_list(scores)
        if len(scores) != n:
            raise ValueError(
                f"scores length ({len(scores)}) != num_blocks ({n}) for "
                f"seq {seq_id}; refusing to pick (cache view desynced)."
            )
        protected = self.protected_indices(n)
        candidates = [i for i in range(n) if i not in protected]
        if not candidates:
            raise RuntimeError(
                f"no evictable block for seq {seq_id}: all {n} blocks are "
                f"protected (sink={self.config.num_sink_blocks}, "
                f"local={self.config.num_local_blocks})."
            )
        # argmin over candidates; ties resolve to the lowest index (stable).
        return min(candidates, key=lambda i: scores[i])

    def record_eviction(self, seq_id: int) -> int:
        """Tell the manager one physical block was freed. Returns new count
        is left to the caller via num_blocks(); here we just bump the counter.
        """
        st = self._require(seq_id)
        st.blocks_evicted += 1
        return st.blocks_evicted

    # -- internal ----------------------------------------------------------
    def _require(self, seq_id: int) -> _SeqState:
        st = self._seqs.get(seq_id)
        if st is None:
            raise KeyError(f"seq {seq_id} not registered")
        return st


def _to_float_list(scores: Sequence[float]) -> list[float]:
    # Accept torch tensors / numpy arrays / lists without importing torch.
    tolist = getattr(scores, "tolist", None)
    if callable(tolist):
        return [float(x) for x in tolist()]
    return [float(x) for x in scores]
