"""Sparse-gather compaction for KV eviction (Phase 1: skip evicted blocks).

Block-wise eviction currently nulls a middle block-table entry (points the
logical slot at the null block) and frees the physical block. The
FlashAttention READ still *gathers* that null block as zeros, so we pay the same
attention FLOPs and add a softmax "dilution" from the zero keys. This module
turns the null-marked worker block table into a COMPACTED block table + reduced
sequence lengths for the attention read, so the kernel skips evicted blocks
entirely: fewer FLOPs, real working-set reduction, and no zero-key dilution.

Correctness rests on three invariants, all guaranteed by the eviction policy
(see kv_evict.manager / kv_evict.integration):

  * Eviction only fires during DECODE (single-token query) at block boundaries,
    so gather ORDER is irrelevant to causal masking -- the decode query attends
    to every cached key regardless of physical layout.
  * Only WHOLE, already-full blocks are evicted (the partial last block is in
    the protected local window), so retained-token accounting is exact:
    reduced_seq_len = seq_len - (#evicted blocks) * block_size.
  * The LOGICAL sequence length is unchanged -- RoPE positions come from the
    model's `positions` tensor, not from the gather list -- so retained
    post-RoPE keys keep exact relative encodings. We only shorten the physical
    gather list, which is why this does NOT corrupt RoPE (unlike length
    compaction, which shifts future query positions).

Pure and NumPy-only, so it is unit-testable without a GPU or vLLM. The worker
block table on the vLLM side is already a NumPy array
(``input_batch.block_table[0].block_table.np``), so this maps directly.
"""

from __future__ import annotations

import numpy as np

# vLLM's reserved placeholder block; a null-marked entry means "evicted".
NULL_BLOCK_ID = 0


def compact_block_table(
    block_table,
    num_blocks_per_row,
    seq_lens,
    block_size,
    null_block_id: int = NULL_BLOCK_ID,
):
    """Compact null-marked (evicted) entries out of a worker block table.

    Args:
        block_table: int array ``[num_reqs, max_blocks]``. An entry equal to
            ``null_block_id`` within a row's logical length marks an evicted
            block. Padding columns (>= ``num_blocks_per_row[r]``) are ignored.
        num_blocks_per_row: int array ``[num_reqs]`` -- logical blocks held.
        seq_lens: int array ``[num_reqs]`` -- context length per request.
        block_size: tokens per KV block.
        null_block_id: id marking evicted / null blocks (default 0).

    Returns:
        ``(compact_bt, compact_num_blocks, compact_seq_lens)`` as NumPy arrays.
        ``compact_bt`` has the retained physical ids left-packed in their
        original logical order, with the tail filled with ``null_block_id``.
        A row with no evictions is returned unchanged (same ids, same length).

    The three outputs are exactly what the FlashAttention read needs: the
    kernel gathers ``ceil(compact_seq_lens[r] / block_size) == compact_num_blocks[r]``
    blocks from the front of ``compact_bt[r]`` -- i.e. only the retained keys.
    """
    bt = np.asarray(block_table)
    n = np.asarray(num_blocks_per_row)
    sl = np.asarray(seq_lens)
    if bt.ndim != 2:
        raise ValueError(f"block_table must be 2-D, got ndim={bt.ndim}")
    num_reqs, max_blocks = bt.shape
    if n.shape != (num_reqs,) or sl.shape != (num_reqs,):
        raise ValueError(
            "num_blocks_per_row and seq_lens must be 1-D of length num_reqs "
            f"({num_reqs}); got {n.shape} and {sl.shape}"
        )

    cols = np.arange(max_blocks)[None, :]
    valid = cols < n[:, None]                      # within logical length
    is_null = valid & (bt == null_block_id)        # evicted (within valid)
    is_retained = valid & ~is_null

    # Stable partition: retained ids to the front, original order preserved;
    # nulls + padding sort to the tail.
    key = np.where(is_retained, 0, 1)
    order = np.argsort(key, axis=1, kind="stable")
    compact = np.take_along_axis(bt, order, axis=1)

    retained = is_retained.sum(axis=1)             # blocks kept per row
    evicted = n - retained                         # whole blocks dropped
    # Blank the tail so nothing beyond the retained count is ever addressable.
    compact = np.where(cols < retained[:, None], compact, null_block_id)

    compact_seq = sl - evicted * block_size
    return (
        compact.astype(bt.dtype, copy=False),
        retained.astype(n.dtype, copy=False),
        compact_seq.astype(sl.dtype, copy=False),
    )


def compact_block_table_torch(
    block_table,
    seq_lens,
    block_size: int,
    query_lens=None,
    null_block_id: int = NULL_BLOCK_ID,
):
    """GPU/torch variant used inside the FlashAttention metadata builder.

    Operates directly on the attention-metadata tensors:
      * ``block_table``: int tensor ``[num_reqs, max_blocks]`` (the gather map;
        a 0 within a row's logical length marks an evicted block).
      * ``seq_lens``:    int tensor ``[num_reqs]`` (== FA ``seqused_k``).

    Returns ``(compact_block_table, compact_seq_lens)`` -- new tensors with
    evicted blocks removed and per-request key length reduced by
    ``#evicted * block_size``. ``max_seqlen_k`` does NOT need updating: it is
    only a buffer/tiling upper bound, and an over-estimate stays correct.

    Only rows with ``query_len == 1`` (pure decode) are compacted when
    ``query_lens`` is given -- multi-token queries (prefill / spec-decode) keep
    their full gather list so index-based causal masking stays valid. Rows
    without evictions are returned unchanged. No in-place mutation, so the
    caller's original tensors remain valid for other consumers.
    """
    import torch

    R, max_blocks = block_table.shape
    device = block_table.device
    n = torch.ceil(seq_lens.to(torch.float32) / block_size).to(torch.long)
    cols = torch.arange(max_blocks, device=device).unsqueeze(0)
    valid = cols < n.unsqueeze(1)
    is_null = valid & (block_table == null_block_id)
    is_retained = valid & ~is_null

    row_do = is_null.any(dim=1)
    if query_lens is not None:
        row_do = row_do & (query_lens == 1)
    if not bool(row_do.any()):
        # Fast path: nothing evicted yet -> hand back the originals untouched.
        return block_table, seq_lens

    key = torch.where(is_retained, 0, 1)
    order = torch.argsort(key, dim=1, stable=True)
    compact = torch.gather(block_table, 1, order)
    retained = is_retained.sum(dim=1)
    evicted = n - retained
    compact = torch.where(
        cols < retained.unsqueeze(1), compact, torch.tensor(null_block_id, dtype=block_table.dtype, device=device)
    )
    new_seq = seq_lens - (evicted * block_size).to(seq_lens.dtype)

    bt_out = torch.where(row_do.unsqueeze(1), compact, block_table)
    seq_out = torch.where(row_do, new_seq, seq_lens)
    return bt_out, seq_out
