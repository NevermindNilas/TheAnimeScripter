# DUT — vendored model code

Inference-only port of [Annbless/DUTCode](https://github.com/Annbless/DUTCode)
(`e74472d`), the official implementation of "DUT: Learning Video Stabilization
by Simply Watching Unstable Videos" (Xu et al., 2020). Upstream license: MIT
(Copyright (c) Yufei Xu), see the upstream repository.

Modifications from upstream, all required to run inside TAS on cp314 +
torch 2.x with no new dependencies:

- `correlation.py`: the cupy JIT correlation kernel is replaced with a pure
  PyTorch cost-volume implementation matching the original kernel exactly
  (81 channels, displacement (dx, dy) = (c % 9 - 4, c // 9 - 4), channel-mean,
  zero padding).
- `projection.py`: `sklearn.cluster.KMeans(n_clusters=2)` is replaced with a
  deterministic numpy Lloyd's k-means (neither sklearn nor scipy is a TAS
  dependency); the majority-cluster relabeling that follows makes the label
  order irrelevant.
- `config.py`: the `easydict` config is inlined as module constants, restored
  to the values the pretrained weights were trained at (640x480; upstream
  changed them to a portrait demo's 448x960 in `e74472d` without retraining).
- `mesh_warp.py`: new. Evaluates the per-cell homography warp at the source
  video's native resolution by scaling the model-resolution mesh, instead of
  warping at 640x480 and resizing back like the upstream demo scripts.
- Training-only and fallback code paths (corner detector, KLT tracker, median
  propagation, StabNet, DIFRINT, descriptor patch clipping) are not ported.
- `torch.meshgrid` calls pass `indexing="ij"` (the pre-1.10 default the
  upstream code relied on implicitly).
- TAS-added, not from upstream: the `loadWeights()` methods on each net
  (upstream loaded checkpoints in `DUT.reload`), `Smoother.smoothPath` (the
  normalization upstream did inline in `DUT.inference`, with an optional
  caller-supplied min/max so chunked smoothing can normalize globally), and
  `image_utils.nms`'s `max_pool2d` window maximum (replaces the 25-slice
  concatenation; bit-identical because `nms` thresholds its input to >= 0
  first, making zero padding and max_pool's -inf padding agree).
- `rf_det.py`'s ten hand-unrolled scale blocks are collapsed into loops;
  module names are preserved so upstream checkpoints load unchanged.

Keep changes here minimal and documented. This tree keeps upstream's naming
and structure (not TAS camelCase) but is ruff-formatted and lint-included —
unlike the diff-clean vendored trees, it is a port and will not track
upstream.
