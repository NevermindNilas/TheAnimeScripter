# Vendored SAM 2

Source: https://github.com/facebookresearch/sam2 at commit
2b90b9f5ceec907a1c18123530e92e794ad901a4, Apache License 2.0 (see `LICENSE`).
Copyright (c) Meta Platforms, Inc. and affiliates.

Only what video tracking needs is kept: `modeling/`, `sam2_video_predictor.py` and
part of `utils/misc.py`.

Changes from upstream:

- Imports rewritten from `sam2.` to `src.sam.sam2.`.
- `modeling/backbones/hieradet.py`: the `iopath` pretrained backbone load is removed.
- `utils/misc.py`: video and image loading (PIL, decord, the async loader) is removed,
  and `get_connected_components` uses OpenCV in place of the compiled `_C` extension.
- `sam2_video_predictor.py`: `init_state` takes ready frames instead of a path, the
  tqdm bar is gone, and
  `from_pretrained` is removed.

`src/sam/build.py` replaces the hydra configuration.
