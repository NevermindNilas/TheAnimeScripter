"""Sparse-convolution op namespace.

The runtime defaults (algorithm choice, CUDA hashmap ratios, serialization
mode) live in :mod:`flex_gemm.config`. This module only re-exposes a thin
backward-compatibility surface:

- :class:`Algorithm` — string-constant namespace, handy when callers want
  ``Algorithm.MASKED_IMPLICIT_GEMM_SPLITK`` instead of typing the literal.
- :func:`set_algorithm` — **deprecated**. Either pass ``algorithm=...`` to
  the op / nn-layer directly, or assign to
  ``flex_gemm.config.DEFAULT_SPCONV_ALGORITHM``.
"""
import warnings








from .submanifold_conv import submanifold_conv
