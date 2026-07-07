"""
Streaming, per-frame-pair scene-change detection for the interpolation path.

Unlike ``src/autoclip`` (a whole-video prepass that writes cut timestamps to a
txt file), these detectors score a single ``(prev, curr)`` pair inside the main
frame loop and return a bool: True == hard cut. On a cut the interpolation loop
emits duplicated source frames instead of morphing across the cut, and resets
the interp driver's frame/feature cache via ``cacheFrameReset``.

Cheap tier (ssim/mse) reuses ``src/dedup``; the maxxvit tier reuses the 6-channel
classifier scorer shared with ``src/autoclip/autoclipMaxxvit.py``.
"""
