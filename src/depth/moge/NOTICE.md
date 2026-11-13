# MoGe-3 port

Provenance for `src/depth/moge/`.

- Architecture and weights: [microsoft/MoGe](https://github.com/microsoft/MoGe) at
  `74fbce0` (V3), MIT License, Copyright (c) Microsoft Corporation. The checkpoint
  is `Ruicheng/moge-3-vitl` on Hugging Face (MIT).
- `model.py` ports `moge/model/v3.py`, `v2.py` and `modules/conv_stack.py`. State
  dict keys are unchanged, so the upstream checkpoint loads directly. The DINOv2
  backbone is TAS's own `src/depth/dinov2.py` (Apache-2.0, Meta).
- `flex_gemm/` is [JeffreyXiang/FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM) at `b2fadb2`
  (MIT, Copyright (c) 2025 Jianfeng Xiang), Python only: the optional C++ `kernels/cuda` is dropped.
  Patched to compile on Triton 3.9: `hashmap.py` casts to a `const=True` pointer, and int32 sentinel
  comparisons in `kernels/triton` use `-1` instead of `0xffffffff`. `fast.py` drives it for CUDA and TensorRT.
  Stripped to what MoGe-3 inference runs: a traced CUDA and TensorRT frame at both
  qualities plus line coverage under fp32 and fp16 autocast, keeping every name the
  kept code references (Triton's dependency hashing reads them). 13,851 to about
  5,300 lines. Removed from upstream, to restore on a re-vendor:
  - whole modules: `nn/{pixel_shuffle,pixel_unshuffle,sparse_conv,submanifold_conv}`,
    `ops/pool/submanifold_pool`, `ops/sample/{grid_sample,pixel_shuffle,pixel_unshuffle}`,
    `ops/spconv/{sparse_conv,sparse_conv_transpose}`, `kernels/triton/sample/`, and the
    split-K kernels `kernels/triton/spconv/sparse_conv_{implicit,masked_implicit}_gemm_splitk`;
  - the `explicit_gemm` and split-K algorithms, `PersistentCacheAutoTuner` and
    `autotune` (only the split-K kernels used them), and the config default algorithm
    (callers pass one);
  - every autograd `backward` and the kernels only they launched (inference only);
  - the C++-extension and `_USE_PYTORCH_FOR_TEST` branches and their config;
  - `build_neighbor_cache`'s kernel_delta and caller-supplied-output-coords builders,
    the transposed strided build, output-shape inference, the fused-4D/ND output-coord
    and ND neighbor-map paths, the bilinear upsample, extrema segment-reduce, and
    `NeighborCache`'s map/segment conversions and bwd/valid-signal accessors (the
    builders kept here store the one representation each consumer reads);
  - unused 2d/4d aliases, torch reference implementations, hashmap lookup,
    coordinate serialize/sort/coalesce and `sparse_to_dense`.
  Removed paths raise `NotImplementedError` naming the removal; a removed spconv
  algorithm name raises `ValueError`. Also fixed:
  `neighbor_map_gray_code_sort` now returns on empty input instead of launching.
  Reproduce from upstream with `tools/moge3_flexgemm_strip/build_stripped.sh` (local,
  gitignored), after applying the Triton 3.9 patches.
- `sparse.py` reimplements the same ops in plain torch for everything else: MPS (Triton has no macOS
  wheels), ROCm (untested with Triton), the ONNX refiner, and any machine where Triton cannot compile
  (no C compiler, or no Python headers as in the Windows portable build). It is about 4x slower.
- `geometry.py` ports `recover_focal_shift` from `moge/utils/geometry_torch.py` and
  `geometry_numpy.py`, with SciPy's `least_squares` replaced by a 1-D damped Gauss-Newton
  using MINPACK's finite-difference step, so it matches SciPy to 6 decimals.

Deviations from upstream:

- Only the point and mask heads run. The normal and metric-scale heads stay in the
  checkpoint but are not built, because TAS writes relative depth.
- One frame per forward. Voxel binning and refinement are per image anyway.
- Under `--half` the DINOv2 encoder runs in fp16 (its Linear/Conv weights are stored
  fp16; LayerNorm, LayerScale and the position/class tokens stay fp32, so the output
  equals upstream's autocast), and the FlexGEMM refiner runs its two coarsest levels
  (256/512 channels) in fp16. The neck, heads, the refiner's first and last
  projections and voxel binning stay fp32. The MPS path is fp32.
