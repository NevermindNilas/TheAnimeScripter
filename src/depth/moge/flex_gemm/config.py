import os
from typing import Literal

USE_AUTOTUNE_CACHE = os.environ.get('FLEX_GEMM_USE_AUTOTUNE_CACHE', '1') == '1'
AUTOSAVE_AUTOTUNE_CACHE = os.environ.get('FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE', '1') == '1'

AUTOTUNE_MODE: Literal['adaptive', 'always', 'never'] = os.environ.get('FLEX_GEMM_AUTOTUNE_MODE', 'adaptive')
"""Autotune trigger policy. One of:

- ``'adaptive'`` (default): tune lazily — for each distinct autotune key (i.e.
  per shape/dtype/etc. signature), only after that key has been encountered
  at least ``AUTOTUNE_ADAPTIVE_THRESHOLD`` times do cache misses actually
  trigger benchmarking. Before that, the first config is used as a fallback
  (not cached). When tuning starts, a notice is printed to ``stderr``. This
  avoids the multi-minute silent stall on cold machines while still recovering
  optimal performance during real training/inference.
- ``'always'``: tune on every cache miss
- ``'never'``: never tune; always fall back to the first config (or cached
  result).
"""
if AUTOTUNE_MODE not in ('adaptive', 'always', 'never'):
    raise ValueError(f"Unknown AUTOTUNE_MODE: {AUTOTUNE_MODE!r}")

AUTOTUNE_ADAPTIVE_THRESHOLD = 100
"""Per autotune key (shape/dtype signature) call count threshold above which
``adaptive`` mode triggers real benchmarking on a cache miss."""

AUTOTUNE_STORE_META = os.environ.get('FLEX_GEMM_AUTOTUNE_STORE_META', '0') == '1'
"""Whether to record per-key timing metadata (top-K runner-up configs and
their measured ms) alongside the autotune cache. Useful for offline pruning
analysis but bloats ``autotune_cache.json`` significantly. Off by default."""

AUTOTUNE_CACHE_PATH = os.environ.get(
    'FLEX_GEMM_AUTOTUNE_CACHE_PATH',
    os.path.expanduser('~/.flex_gemm/autotune_cache.json')
)





# ---------------------------------------------------------------------------
# Sparse-convolution defaults
# ---------------------------------------------------------------------------






SPCONV_ALLOW_TF32: bool = True
"""Whether the Triton sparse-conv matmul kernels are allowed to use TF32 precision 
for much faster performance on Ampere and later GPUs.
Set to ``False`` to force IEEE single-precision accumulation."""
