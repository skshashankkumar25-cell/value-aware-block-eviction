"""vLLM-side integration glue for KV-eviction (Stages 4b/4c).

This module is imported *inside* the running vLLM worker process. It must not
import vLLM at module load time (it's called from within vLLM), only torch.

Stage 4b: worker-side per-block scoring from the live KV cache, plus a
read-only debug logger used to validate the slicing/layout on real data
WITHOUT mutating anything.
"""

from __future__ import annotations

import numpy as np
import torch

import os

from .manager import EvictionConfig
from .scoring import DEFAULT_EPS, score_blocks_value_attention

# vLLM's reserved placeholder block (block_pool.py: "placeholder block with
# block_id=0"). Pointing a block-table entry here frees the real block while
# reads return zeros (the cache is .zero_()'d at init and block 0 is never
# written). Under full attention this is a bounded zero-value attention sink --
# it does NOT shift positions, so RoPE stays exact for every retained token.
NULL_BLOCK_ID = 0


def score_request_blocks(
    kv_caches: list[torch.Tensor],
    block_ids,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Per-block importance for one request, aggregated over all layers.

    Each entry of ``kv_caches`` is a FlashAttention KV tensor of shape
    ``(num_blocks, 2, block_size, num_kv_heads, head_size)`` (NHD layout; dim 1
    index 0 = K, 1 = V). For physical blocks ``block_ids`` we compute

        score(b) = ||V_b||_2 / (||K_b||_2 + eps)

    where the norm is the L2 over every element of that block across ALL
    attention layers (i.e. the layers' contributions are concatenated). Lower
    score = evict first. Returns a 1-D float32 tensor of length len(block_ids),
    on the kv-cache device.
    """
    idx = torch.as_tensor(list(block_ids), dtype=torch.long)
    n = idx.numel()
    if n == 0:
        return torch.empty(0, dtype=torch.float32)

    k_terms: list[torch.Tensor] = []
    v_terms: list[torch.Tensor] = []
    for kc in kv_caches:
        # Only full-attention KV tensors (skip e.g. mamba state tensors).
        if kc is None or kc.ndim != 5 or kc.shape[1] != 2:
            continue
        i = idx.to(kc.device)
        k_blocks = kc[i, 0].reshape(n, -1).to(torch.float32)
        v_blocks = kc[i, 1].reshape(n, -1).to(torch.float32)
        k_terms.append((k_blocks * k_blocks).sum(dim=1))
        v_terms.append((v_blocks * v_blocks).sum(dim=1))

    if not k_terms:  # no attention KV caches found
        return torch.empty(0, dtype=torch.float32)
    sum_k = torch.stack(k_terms).sum(dim=0)
    sum_v = torch.stack(v_terms).sum(dim=0)
    return torch.sqrt(sum_v) / (torch.sqrt(sum_k) + eps)


def _row_block_ids(model_runner, req_index: int):
    """Physical block ids currently held by request row `req_index` (group 0)."""
    bt = model_runner.input_batch.block_table[0]  # group 0 BlockTable
    n = int(bt.num_blocks_per_row[req_index])
    if n <= 0:
        return []
    return bt.block_table.np[req_index, :n].tolist()


def debug_log_scores(model_runner) -> bool:
    """Stage 4b validation: score the first request that has >= 2 blocks and
    log the result. READ-ONLY -- mutates nothing. Returns True once it has
    logged (so the caller can stop invoking it).
    """
    ib = model_runner.input_batch
    num_reqs = ib.num_reqs if hasattr(ib, "num_reqs") else len(ib.req_ids)
    for ri in range(num_reqs):
        block_ids = _row_block_ids(model_runner, ri)
        if len(block_ids) < 2:
            continue
        scores = score_request_blocks(model_runner.kv_caches, block_ids)
        vals = [round(x, 4) for x in scores.tolist()]
        finite = bool(torch.isfinite(scores).all().item())
        msg = (
            f"[SKIVE 4b] req_index={ri} blocks={len(block_ids)} "
            f"finite={finite} scores={vals}"
        )
        print(msg, flush=True)
        try:
            from vllm.logger import init_logger

            init_logger(__name__).info(msg)
        except Exception:
            pass
        return True
    return False


# --------------------------------------------------------------------------
# Stage 4c: actual eviction via null-block replacement (worker-side).
# --------------------------------------------------------------------------
def build_eviction_config(cache_config) -> EvictionConfig | None:
    """Construct an EvictionConfig from the vLLM cache_config, or None if
    eviction is disabled / misconfigured (in which case we never evict)."""
    if not getattr(cache_config, "kv_evict_enabled", False):
        return None
    budget = getattr(cache_config, "kv_evict_budget", None)
    if budget is None:
        return None
    try:
        return EvictionConfig(
            block_size=cache_config.block_size,
            kv_budget=int(budget),
            num_sink_blocks=int(getattr(cache_config, "kv_evict_num_sink_blocks", 0)),
            num_local_blocks=int(getattr(cache_config, "kv_evict_num_local_blocks", 0)),
            metric=os.environ.get("SKIVE_METRIC", "vk_ratio"),
        )
    except ValueError:
        return None


def score_request_blocks_va(model_runner, req_index, real_block_ids,
                            eps: float = DEFAULT_EPS):
    """SKIVE-style value x attention score for a request's real blocks,
    aggregated over layers. Returns a 1-D tensor (len == #real blocks) or None
    if the per-layer query wasn't captured (caller falls back to vk_ratio)."""
    try:
        import torch
        bs = int(model_runner.cache_config.block_size)
        q_row = int(model_runner.query_start_loc.np[req_index + 1]) - 1
        layers = [
            m for _, m in
            model_runner.compilation_config.static_forward_context.items()
            if type(m).__name__ in ("Attention", "MLAAttention")
        ]
        kv_caches = model_runner.kv_caches
        # Layer-stride scoring (Zipage): aggregate the value_attention score over
        # every stride-th layer instead of all layers -> ~stride x cheaper. The
        # relative block ranking is preserved, so eviction quality is ~unchanged.
        stride = max(1, int(os.environ.get("SKIVE_SCORE_LAYER_STRIDE", "1")))
        total = None
        for li in range(0, len(layers), stride):
            layer = layers[li]
            q = getattr(layer, "_skive_q", None)
            if q is None or li >= len(kv_caches):
                return None
            kc = kv_caches[li]
            if kc is None or kc.ndim != 5 or kc.shape[1] != 2:
                continue
            dev = kc.device
            idx = torch.as_tensor(real_block_ids, dtype=torch.long, device=dev)
            qr = q[q_row].to(dev)                       # [num_heads, head_size]
            kb = kc[idx, 0]                              # [nb, bs, Hkv, D]
            vb = kc[idx, 1]
            nb = kb.shape[0]
            k_all = kb.reshape(nb * bs, kb.shape[2], kb.shape[3])
            v_all = vb.reshape(nb * bs, vb.shape[2], vb.shape[3])
            nqpk = getattr(layer.impl, "num_queries_per_kv",
                           qr.shape[0] // kb.shape[2])
            scale = getattr(layer.impl, "scale", None)
            s = score_blocks_value_attention(qr, k_all, v_all, bs, nqpk, scale, eps)
            total = s if total is None else total + s
        return total
    except Exception:
        return None


def _ensure_null_zeroed(model_runner) -> None:
    """One-time defensive zero of the null block's KV across all layers."""
    if getattr(model_runner, "_skive_null_zeroed", False):
        return
    for kc in model_runner.kv_caches:
        if kc is not None and kc.ndim == 5 and kc.shape[1] == 2:
            kc[NULL_BLOCK_ID].zero_()
    model_runner._skive_null_zeroed = True


def _vk_score_cached(model_runner, req_id, real_pos, real_block_ids,
                     eps: float = DEFAULT_EPS):
    """vk_ratio scores for a request's real blocks, cached per (req, logical
    position).

    A block's K/V are frozen once the block is full, so its ``||V||/||K||``
    score never changes. We compute each block's score once and reuse it,
    turning the per-step scoring cost from O(all blocks) into O(newly-filled
    blocks) -- the dominant eviction-hot-path overhead. The cache entry stores
    the physical block id so a reused slot is re-scored rather than trusted.
    """
    cache = getattr(model_runner, "_skive_score_cache", None)
    if cache is None:
        cache = model_runner._skive_score_cache = {}
    rc = cache.setdefault(req_id, {})
    scores = [0.0] * len(real_block_ids)
    todo = []  # (k, logical_j, block_id) for uncached blocks
    for k, (j, bid) in enumerate(zip(real_pos, real_block_ids)):
        ent = rc.get(j)
        if ent is not None and ent[0] == bid:
            scores[k] = ent[1]
        else:
            todo.append((k, j, bid))
    if todo:
        new = score_request_blocks(
            model_runner.kv_caches, [bid for _, _, bid in todo], eps
        ).tolist()
        for (k, j, bid), s in zip(todo, new):
            rc[j] = (bid, s)
            scores[k] = s
    return scores


def _batch_prescore(model_runner, over_rows, ib, npbt, nrows,
                    eps: float = DEFAULT_EPS) -> None:
    """Populate the vk_ratio score cache for every cache-miss block across ALL
    over-budget rows in a SINGLE batched GPU call.

    This is the key eviction-hot-path optimization: scoring per request meant
    one GPU->CPU ``.tolist()`` sync per over-budget row every step (~6/step),
    and each sync stalls the async decode pipeline. Batching collapses that to
    one sync per step regardless of how many rows evict.
    """
    cache = getattr(model_runner, "_skive_score_cache", None)
    if cache is None:
        cache = model_runner._skive_score_cache = {}
    misses = []       # (req_id, logical_j, block_id) needing a score
    miss_bids = []
    for ri in over_rows.tolist():
        req_id = ib.req_ids[ri]
        if req_id is None:
            continue
        rc = cache.setdefault(req_id, {})
        n = int(nrows[ri])
        row = npbt[ri]
        for j in range(n):
            bid = int(row[j])
            if bid == NULL_BLOCK_ID:
                continue
            ent = rc.get(j)
            if ent is None or ent[0] != bid:
                misses.append((req_id, j, bid))
                miss_bids.append(bid)
    if miss_bids:
        s = score_request_blocks(model_runner.kv_caches, miss_bids, eps).tolist()
        for (req_id, j, bid), sc in zip(misses, s):
            cache[req_id][j] = (bid, sc)


def _batch_score_va(model_runner, over_rows, ib, npbt, nrows,
                    eps: float = DEFAULT_EPS):
    """Batched SKIVE value x attention scores for ALL over-budget rows at once.

    Replaces the per-request scoring loop (one q@K matmul chain + one GPU->CPU
    sync per request) with a few fused ops per sampled layer and a SINGLE sync
    per step. Rows are padded to the max block count; padded tokens are masked
    to -inf before the softmax, so results match the per-request path exactly.
    Honors SKIVE_SCORE_LAYER_STRIDE. Returns {req_index: [block scores]} or
    None (caller falls back to the per-request path / vk_ratio)."""
    try:
        import torch
        bs = int(model_runner.cache_config.block_size)
        stride = max(1, int(os.environ.get("SKIVE_SCORE_LAYER_STRIDE", "1")))
        layers = [
            m for _, m in
            model_runner.compilation_config.static_forward_context.items()
            if type(m).__name__ in ("Attention", "MLAAttention")
        ]
        kv_caches = model_runner.kv_caches
        rows = over_rows.tolist()
        bids_per_row = []
        for ri in rows:
            n = int(nrows[ri]); row = npbt[ri]
            bids_per_row.append([int(row[j]) for j in range(n)
                                 if int(row[j]) != NULL_BLOCK_ID])
        R = len(rows)
        nb_per_row = [len(b) for b in bids_per_row]
        max_nb = max(nb_per_row)
        idx = torch.zeros((R, max_nb), dtype=torch.long)
        for r, bids in enumerate(bids_per_row):
            idx[r, :len(bids)] = torch.as_tensor(bids, dtype=torch.long)
        q_rows = [int(model_runner.query_start_loc.np[ri + 1]) - 1 for ri in rows]

        total = None
        for li in range(0, len(layers), stride):
            layer = layers[li]
            q = getattr(layer, "_skive_q", None)
            if q is None or li >= len(kv_caches):
                return None
            kc = kv_caches[li]
            if kc is None or kc.ndim != 5 or kc.shape[1] != 2:
                continue
            dev = kc.device
            idx_d = idx.to(dev)
            qr = q[q_rows].to(dev).to(torch.float32)            # [R, Hq, D]
            kb = kc[idx_d.reshape(-1), 0].to(torch.float32)     # [R*max_nb, bs, Hkv, D]
            vb = kc[idx_d.reshape(-1), 1].to(torch.float32)
            Hkv, D = kb.shape[2], kb.shape[3]
            T = max_nb * bs
            k_all = kb.reshape(R, T, Hkv, D)
            v_all = vb.reshape(R, T, Hkv, D)
            Hq = qr.shape[1]
            g = getattr(layer.impl, "num_queries_per_kv", Hq // Hkv)
            k_h = k_all.permute(0, 2, 1, 3).repeat_interleave(g, dim=1)  # [R,Hq,T,D]
            v_h = v_all.permute(0, 2, 1, 3).repeat_interleave(g, dim=1)
            scale = getattr(layer.impl, "scale", None) or (1.0 / D ** 0.5)
            logits = torch.einsum("rhd,rhtd->rht", qr, k_h) * scale      # [R,Hq,T]
            tok_valid = (torch.arange(T, device=dev)[None, :] <
                         torch.as_tensor([n * bs for n in nb_per_row],
                                         device=dev)[:, None])           # [R,T]
            logits = logits.masked_fill(~tok_valid[:, None, :], float("-inf"))
            p = torch.softmax(logits, dim=2)
            v_norm = torch.linalg.vector_norm(v_h, ord=1, dim=3)          # [R,Hq,T]
            token_score = (p * v_norm).sum(dim=1)                          # [R,T]
            s = token_score.view(R, max_nb, bs).sum(dim=2)                 # [R,max_nb]
            total = s if total is None else total + s
        if total is None:
            return None
        flat = total.tolist()  # the ONE GPU->CPU sync for the whole step
        return {ri: flat[r][:nb_per_row[r]] for r, ri in enumerate(rows)}
    except Exception:
        return None


def evict_request_blocks(
    model_runner, cfg: EvictionConfig, req_index: int
) -> list[int]:
    """Null-replace the lowest-importance non-protected real blocks of one
    request until it is back within budget. Returns the logical row indices
    that were nulled (so the scheduler can free the same physical blocks).

    Positions/length are left UNCHANGED (RoPE stays exact); only the block
    table entry is repointed to the null block.
    """
    bt = model_runner.input_batch.block_table[0]  # group 0
    n = int(bt.num_blocks_per_row[req_index])
    if n == 0:
        return []
    row = bt.block_table.np[req_index]
    block_ids_full = row[:n].tolist()

    # Real (non-null) blocks, in logical order, with their row indices.
    real_pos = [i for i in range(n) if block_ids_full[i] != NULL_BLOCK_ID]
    num_real = len(real_pos)
    if num_real <= cfg.kv_budget:
        return []

    # Protected over the REAL ordering: first num_sink + last num_local.
    sink, local = cfg.num_sink_blocks, cfg.num_local_blocks
    protected_k = set(range(0, min(sink, num_real))) | set(
        range(max(0, num_real - local), num_real)
    )
    candidates_k = [k for k in range(num_real) if k not in protected_k]
    if not candidates_k:
        return []

    num_to_evict = min(num_real - cfg.kv_budget, len(candidates_k))

    # Score the real blocks. value_attention is query-dependent (recomputed);
    # vk_ratio is cached per block (frozen K/V) to avoid re-scoring every step.
    real_block_ids = [block_ids_full[real_pos[k]] for k in range(num_real)]
    req_id = model_runner.input_batch.req_ids[req_index]
    scores = None
    if os.environ.get("SKIVE_NOSCORE") == "1":  # diagnostic: skip GPU scoring
        scores = list(range(num_real))
    elif getattr(cfg, "metric", "vk_ratio") == "value_attention":
        # Prefer the whole-batch pre-scored result (one sync/step); fall back
        # to the per-request path if it's missing or desynced.
        vb = getattr(model_runner, "_skive_va_batch", None)
        if vb is not None and req_index in vb and len(vb[req_index]) == num_real:
            scores = vb[req_index]
        else:
            sv = score_request_blocks_va(model_runner, req_index, real_block_ids)
            if sv is not None:
                scores = sv.tolist()
    if scores is None:  # default proxy (cached), or fallback if query missing
        scores = _vk_score_cached(model_runner, req_id, real_pos, real_block_ids)

    # Lowest-importance candidates first (stable on ties via index).
    chosen = sorted(candidates_k, key=lambda k: (scores[k], k))[:num_to_evict]
    rc = getattr(model_runner, "_skive_score_cache", {}).get(req_id)
    nulled_rows = []
    for k in chosen:
        j = real_pos[k]
        row[j] = NULL_BLOCK_ID  # repoint worker table to null
        nulled_rows.append(j)
        if rc is not None:
            rc.pop(j, None)  # evicted block's cached score no longer needed
    return nulled_rows


def skive_post_step(model_runner) -> None:
    """Stable post-decode entrypoint called from the patched gpu_model_runner.

    Nulls evicted entries in the worker block table and records (req_id, j)
    pairs on ``model_runner._skive_pending_free`` for EngineCore to free on the
    scheduler side (4c-ii). Flag-gated upstream; a None config => no-op."""
    cfg = build_eviction_config(model_runner.cache_config)
    if cfg is None:
        return
    _ensure_null_zeroed(model_runner)
    _dbg = os.environ.get("SKIVE_TIME") == "1"
    if _dbg:
        import time as _t
        if os.environ.get("SKIVE_TIME_SYNC") == "1":
            torch.cuda.synchronize()  # drain GPU so timer measures pure CPU work
        _t0 = _t.perf_counter()

    ib = model_runner.input_batch
    num_reqs = ib.num_reqs if hasattr(ib, "num_reqs") else len(ib.req_ids)
    bt = ib.block_table[0]

    # Vectorized over-budget check for the whole batch in one NumPy op. This
    # replaces the per-request Python scan that ran every step: eviction only
    # actually fires when a block fills (~every block_size steps), so on most
    # steps this finds nothing over budget and we skip all per-request work.
    npbt = bt.block_table.np[:num_reqs]
    nrows = np.asarray(bt.num_blocks_per_row[:num_reqs])
    cols = np.arange(npbt.shape[1])[None, :]
    real_count = ((cols < nrows[:, None]) & (npbt != NULL_BLOCK_ID)).sum(1)
    over_rows = np.nonzero(real_count > cfg.kv_budget)[0]

    # Deferred eviction: block scores are static, so we don't need to pay the
    # GPU->CPU scoring sync every step. Only score+evict every SKIVE_EVICT_EVERY
    # steps, unless a row is >= MARGIN over budget (safety valve against pool
    # exhaustion). This cuts the per-step sync-stall frequency ~N-fold; the cost
    # is a few extra retained blocks briefly (peak ~budget + MARGIN).
    step = getattr(model_runner, "_skive_step", 0) + 1
    model_runner._skive_step = step
    if over_rows.size:
        excess = int((real_count[over_rows] - cfg.kv_budget).max())
        _every = int(os.environ.get("SKIVE_EVICT_EVERY", "8"))
        _margin = int(os.environ.get("SKIVE_EVICT_MARGIN", "4"))
        if step % _every != 0 and excess < _margin:
            over_rows = over_rows[:0]  # defer to a later step

    if _dbg:
        import time as _t
        _tc = _t.perf_counter()

    if os.environ.get("SKIVE_DBG2") == "1":
        _s = getattr(model_runner, "_skive_dbgstep", 0) + 1
        model_runner._skive_dbgstep = _s
        if _s <= 12 or _s % 50 == 0:
            print(f"[SKIVE DBG] step={_s} nreq={num_reqs} "
                  f"n_max={int(nrows[:num_reqs].max())} "
                  f"real_max={int(real_count.max())} real_min={int(real_count.min())} "
                  f"over={len(over_rows)} budget={cfg.kv_budget}", flush=True)

    # Batch-score every over-budget row in one GPU pass (one sync/step) before
    # the per-row eviction loop, which then runs entirely on CPU-held scores --
    # no per-row GPU sync. vk_ratio: cache-miss blocks only (scores static).
    # value_attention: fused whole-batch q@K scoring (query-dependent).
    model_runner._skive_va_batch = None
    if over_rows.size:
        if getattr(cfg, "metric", "vk_ratio") == "value_attention":
            model_runner._skive_va_batch = _batch_score_va(
                model_runner, over_rows, ib, npbt, nrows)
        else:
            _batch_prescore(model_runner, over_rows, ib, npbt, nrows)

    pending = getattr(model_runner, "_skive_pending_free", [])
    total = 0
    for ri in over_rows.tolist():
        req_id = ib.req_ids[ri]
        if req_id is None:
            continue
        for j in evict_request_blocks(model_runner, cfg, ri):
            pending.append((req_id, j))
            total += 1
    model_runner._skive_pending_free = pending

    # Drop score-cache entries for requests no longer in the batch (bounded).
    cache = getattr(model_runner, "_skive_score_cache", None)
    if cache:
        live = {rid for rid in ib.req_ids if rid is not None}
        for rid in [r for r in cache if r not in live]:
            del cache[rid]

    if _dbg:
        import time as _t
        _tend = _t.perf_counter()
        acc = getattr(model_runner, "_skive_t", 0.0) + (_tend - _t0)
        acc_chk = getattr(model_runner, "_skive_tchk", 0.0) + (_tc - _t0)
        acc_loop = getattr(model_runner, "_skive_tloop", 0.0) + (_tend - _tc)
        cnt = getattr(model_runner, "_skive_n", 0) + 1
        model_runner._skive_t, model_runner._skive_n = acc, cnt
        model_runner._skive_tchk, model_runner._skive_tloop = acc_chk, acc_loop
        fired = getattr(model_runner, "_skive_fired", 0) + (1 if total else 0)
        model_runner._skive_fired = fired
        if cnt % 200 == 0:
            print(f"[SKIVE TIME] post_step={1000 * acc / cnt:.3f} ms/step "
                  f"(check={1000 * acc_chk / cnt:.3f} loop={1000 * acc_loop / cnt:.3f}) "
                  f"fired {fired}/{cnt}", flush=True)
    if total:
        c = getattr(model_runner, "_skive_evicted_total", 0) + total
        model_runner._skive_evicted_total = c
        print(f"[SKIVE 4c] evicted {total} block(s) this step; cumulative={c}",
              flush=True)


def _skive_pop_pending(worker):
    """Worker-side: return and clear the pending (req_id, logical_index) frees.

    Used via collective_rpc from EngineCore. ``worker`` is the (wrapped) worker;
    attribute access delegates to the real worker, so .model_runner resolves."""
    mr = getattr(worker, "model_runner", None)
    if mr is None:
        return []
    pending = getattr(mr, "_skive_pending_free", None)
    if not pending:
        return []
    mr._skive_pending_free = []
    if os.environ.get("SKIVE_NO_RECLAIM") == "1":  # diagnostic: skip physical free
        return []
    return list(pending)


def skive_reclaim(kv_cache_manager, pending) -> int:
    """Scheduler-side: free the physical blocks the worker evicted, mirroring
    them to null_block in req_to_blocks (group 0). Returns #blocks freed.

    This mirrors vLLM's own sliding-window path (remove_skipped_blocks): null
    substitution + BlockPool.free_blocks. Prefix caching is OFF in this stage,
    so there are no shared/cached-block ref-count complications.
    """
    block_pool = kv_cache_manager.block_pool
    null_block = block_pool.null_block
    managers = kv_cache_manager.coordinator.single_type_managers
    if not managers:
        return 0
    mgr = managers[0]  # group 0 (single full-attention group); multi-group=Stage 5
    freed = 0
    for req_id, j in pending:
        blocks = mgr.req_to_blocks.get(req_id)
        if not blocks or j >= len(blocks):
            continue
        old = blocks[j]
        if getattr(old, "is_null", False):
            continue
        blocks[j] = null_block
        block_pool.free_blocks([old])
        freed += 1
    if freed:
        c = getattr(kv_cache_manager, "_skive_freed_total", 0) + freed
        kv_cache_manager._skive_freed_total = c
        print(f"[SKIVE 4c-ii] freed {freed} physical block(s); cumulative={c}; "
              f"pool_free={block_pool.get_num_free_blocks()}", flush=True)
    return freed
