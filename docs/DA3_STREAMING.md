# DA3 depth-video streaming

Use `--depth_method video_small_v3` or `video_base_v3` on CUDA. Both reuse the
same weights/cache as `small_v3` and `base_v3`. No additional packages or weights
are required when those models are already installed.

```bash
python main.py --input input.mp4 --output depth.mp4 --depth_method video_small_v3 --depth_window 8 --depth_quality low
```

`--depth_window` is the chunk size (4, 8, 16 or 32; default 32). Adjacent chunks
share half their frames. Larger windows increase attention cost and VRAM use;
start with 4 or 8 at low quality. `--depth_batch` is forced to 1, as for other
temporal depth methods. The existing encoder, output bit depth and video range
options apply. Each frame uses its own disparity percentile stretch.

## What is implemented

Small/Base receive views as `[1, N, 3, H, W]`, enabling their trained attention
across views. The image backends use `[N, 1, 3, H, W]` for independent frames.
The new scheduler holds a bounded window of decoded RGB frames and depth maps,
aligns successive chunks by a robust depth-scale ratio on their repeated views,
and blends two predictions of each repeated frame. It never blends pixels from
different source frames. Every frame is emitted once, including partial chunks
and the final overlap. An inference failure marks the run failed and uses the
shared writer shutdown/reader drain path.

This is a TAS depth-video adaptation of the overlapping-chunk approach, **not a
port of the complete upstream DA3-Streaming 3D reconstruction pipeline**. It
does not estimate camera trajectories, align point clouds with Sim(3), or run
SALAD loop closure. It needs lookahead up to one chunk, rather than causal
single-frame inference. Scale drift and artifacts across scene cuts remain
possible; no upstream reconstruction accuracy or memory benchmark is claimed.

## Why not `video_large_v3`?

Our `large_v3` selects **DA3Mono-Large**, and `og_large_v3` selects
**DA3Metric-Large**. Both checkpoints are Apache-2.0 and remain available for
monocular video processing. Their backbone configuration has `alt_start: -1`,
disabling attention across views, and lacks the camera decoders needed by the
upstream streaming alignment. Chunking these models does not add learned
temporal context. **DA3-Large (any-view)** is a different checkpoint, licensed
CC BY-NC 4.0, and is excluded from this integration.

## Limbo checkpoint compatibility

An experimental CUDA test on `Raw.mp4` (78 frames, 504x280 input) confirmed
that both Limbo V1 and V2 checkpoints accept multi-view inputs and use their
DA3-Small cross-frame attention. Changing neighboring views while holding the
target image fixed changed its prediction. Both returned finite depth maps for
every frame with 8/16/32-frame overlapping windows.

With a 16-frame window, motion-compensated display-depth variation fell by
approximately 62% for each checkpoint compared with independent inference.
This is a temporal-consistency proxy on one short gameplay clip, not a depth
accuracy measurement or validation on anime. The existing `limbo` and
`limbo_v2` CLI modes remain independent-frame methods; this release only adds
`video_small_v3` and `video_base_v3`.

## License review (2026-09-12)

Reviewed upstream commit `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`:

- [DA3 code license](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/LICENSE): Apache-2.0.
- [DA3 model table](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/README.md): Small, Base, Mono-Large and Metric-Large are Apache-2.0; the Large/Giant any-view and Nested models are CC BY-NC 4.0.
- [Streaming script](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/da3_streaming/da3_streaming.py): overlapping chunks, with confidence and camera outputs required by its 3D alignment.
- [Mono-Large architecture](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src/depth_anything_3/configs/da3mono-large.yaml): monocular attention and DPT head.
- [Default weight downloader](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/da3_streaming/scripts/download_weights.sh): selects the non-commercial Nested checkpoint; not used here.
- [Pinned SALAD license](https://github.com/serizba/salad/blob/6aede13a3f6c25750bf7fde10209c06cb73060bb/LICENSE): GPL-3.0, not AGPL; not included here.

The new integration introduces no non-commercial or AGPL third-party dependency.
TAS itself retains its existing project license.
