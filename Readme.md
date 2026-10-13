 # Value-Aware Block KV-Cache Eviction for vLLM (`kv_evict`)

  Block-wise KV-cache eviction for long-context serving. During decode, each
  sequence's KV blocks are scored and the lowest-value blocks are evicted once a
  per-sequence budget is exceeded, freeing paged KV memory while protecting the
  attention-sink and recent-local blocks. Integrates with vLLM's V1 engine and
  PagedAttention. Pure Python, no CUDA rebuild.

  ## Files

  | File | Role |
  |------|------|
  | `scoring.py` | Block scoring: `value_attention` (value-aware), plus `knorm`, `keydiff` baselines |
  | `selection.py` | Picks blocks to evict given scores + protected regions |
  | `manager.py` | `BlockEvictionManager` / `EvictionConfig`: budget + sink/local protection |
  | `compaction.py` | Compacts the paged block table so evicted blocks are skipped on read |
  | `prefill_evict.py` | Optional eviction during prefill |
  | `integration.py` | vLLM glue: post-step scoring hook + physical-block reclaim |
  | `__init__.py` | Public API: `score_blocks`, `BlockEvictionManager`, `EvictionConfig` |

  ## Scoring (value_attention)

  For the current decode query, each retained token scores
  `sum_h softmax(q·k·scale) · ||v||_1`, summed over tokens in a block and over the
  scored layers. Highest-scoring blocks are kept; lowest are evicted.
  See `score_blocks_value_attention` in `scoring.py`.

  ## Integrate into vLLM

  Drop-in folder + 7 guarded hooks (all tagged `# SKIVE:`), Python-only:

  1. `cp -r kv_evict <vllm>/vllm/kv_evict`
  2. Apply the hooks (`git apply skive_integration.patch`) to:
     `config/cache.py`, `config/vllm.py`, `engine/arg_utils.py`,
     `v1/attention/backends/flash_attn.py`, `v1/worker/gpu_model_runner.py`,
     `v1/worker/gpu/model_runner.py`, `v1/engine/core.py`
  3. `VLLM_USE_PRECOMPILED=1 uv pip install -e .`

  ## Enable

  CLI:
  --kv-evict-enabled --kv-evict-budget 2048
  --kv-evict-num-sink-blocks 2 --kv-evict-num-local-blocks 6
  Env:
  | Var | Meaning | Default |
  |-----|---------|---------|
  | `SKIVE_METRIC` | `value_attention` \| `knorm` \| `keydiff` | `vk_ratio` |
  | `SKIVE_SPARSE_GATHER` | skip evicted blocks on attention read | `1` |
  | `SKIVE_SCORE_LAYER_STRIDE` | score every Nth layer | impl default |
  | `SKIVE_ALLOW_TP` | lift the TP guard for the validation harness | unset |

  `kv_evict_budget` = max KV blocks kept per sequence before eviction fires
  (`None` disables). Sink + local blocks are never evicted.

  ## Validate

  Run a long-context prompt with a budget below the context length so eviction
  fires, and compare quality/throughput against full-KV. `run_longbench.sh` drives
  the full quality-vs-budget sweep.