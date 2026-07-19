# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - YYYY-MM-DD
### Added
- DirectML and OpenVINO backends for DistilDRBA: `--interpolate_method distildrba-directml`, `distildrba-lite-directml`, `distildrba-openvino`, `distildrba-lite-openvino`. The exported graph has the same shape as the TensorRT one (img0/img1/img2/timestep in, one frame out, no recurrent state), so the driver only carries `img0` across calls; `temporalWindow = (0, 1)` gets it the next frame from the `FrameWindow`. The arch lives in `src/rifearches/DistilDRBA_directml.py` and decomposes every `grid_sample` into floor/gather/weighted-sum, since DirectML has no GridSample kernel and would otherwise fall the whole warp back to the CPU — the same reason `Rife_directml.py` exists. Verified against the eager arch on the DirectML EP to a max absolute difference of 2.5e-5 (fp32). Measured at 1080p fp16 on the DirectML EP: `distildrba-lite-directml` 25.9 interp-fps and `distildrba-directml` 13.3, against 10.8 / 12.2 / 13.6 for `rife4.25-lite-directml` / `rife4.25-directml` / `rife4.25-heavy-directml` — the lite variant is the fastest interpolation method on the DirectML path, not the slowest.
- DirectML and OpenVINO backends for Rife Elexor: `--interpolate_method rife_elexor-directml` / `rife_elexor-openvino`. `IFNet_elexor` joins the other hand-ported archs in `src/rifearches/Rife_directml.py` (decomposed `grid_sample`, `encode` inlined so the graph stays 3-in/1-out); no new weights, it reuses `rife_elexor.pth` and exports ONNX at runtime. Matches the eager arch to 1.6e-5 with ensemble on and off, and runs at 12.3 interp-fps at 1080p fp16 against 15.3 for `rife4.25-heavy-directml`. Elexor is the one model on this path that needs DirectML's graph fusion disabled (`ep.dml.disable_graph_fusion`, scoped to elexor so every other RIFE model keeps it): its flow ladder restarts after the second pass and carries the pre-restart flow forward as a long-range residual edge, and `DmlGraphFusionHelper` cannot build across that edge — it throws `RUNTIME_EXCEPTION: ... The parameter is incorrect` at session init. Bisected: the arch compiles with the edge removed, and a bare `flow + 0.0 * largeFlow` reproduces the failure on its own, while every op in the large-flow gate places on DML fine in isolation. So it is a fusion bug on the skip connection, not an unsupported op.

### Fixed
- Output frame rate was quantized to two decimals, drifting the timeline against the source. `getVideoMetadata()` stored `"FPS": round(fps, 2)`, so a `24000/1001 = 23.9760…` fps source became `23.98`; `main.py` reads that single value as `self.fps`, feeds it to `self.outputFPS`, and the writer emits `-r 23.98`. In After Effects a `restore+scale` re-render then imported at 23.98 instead of 23.976, and dropping it into a 23.976 comp cut a frame or two off the tail. The exact float is now stored in the metadata (`24000/1001` survives to the encoder's `-r`); rounding is kept only in the metadata log line and the progress bar, so no display output changes. Every other consumer of `metadata["FPS"]` (`totalFramesToProcess`, `fps * interpolateFactor`) already wanted the exact value.
- The upscale path crashed on any input whose dimensions were not divisible by 4 when the model is a RealCUGAN-family arch (`fallin_soft`, `fallin_strong`, `shufflecugan`, `aniscale2`) — on **every** backend. These archs run an internal two-level down/up cascade whose skip-connection `Add` only aligns when both spatial dims are mod-4; a `498x566` AE pre-render raised (on DirectML/OpenVINO) `Non-zero status code returned while running Add node. Name:'/Add_6' ... Attempting to broadcast an axis by a dimension other than 1. 274 by 275`, then failed again on the CPU fallback; the CUDA/MPS/TensorRT paths failed the same way (at warmup or the skip `Add`). The generic upscalers defined `calculatePadding` but never used it — only the `AnimeSR*` classes padded. Each generic upscaler (`UniversalDirectML`, `UniversalPytorch`, `UniversalPytorchMPS`, `UniversalTensorRT`) now detects the arch's required input multiple once at load — a resolution-independent probe: a throwaway CPU ONNX session for DML/TRT, a direct model forward for CUDA/MPS (`smallestValidMultiple` in `src/upscale/_shared.py`) — reflect-pads the input to it, and crops the surplus back off the output. TensorRT builds its engine for the padded shape. Fully-convolutional archs (compact/SPAN/superultracompact) probe to `1`, so padding stays zero and output is **bit-identical** (verified: SPAN on CUDA, max abs diff 0.0 vs the unpadded forward); a resolution that is already a multiple of the requirement also pads zero. The CUDA/MPS probe runs under `torch.no_grad()`, not `torch.inference_mode()`, so archs that lazily reparameterize on first forward (SPAN's `eval_conv`) do not cache an inference-mode tensor the grad-tracked warmup cannot reuse. Verified end-to-end at `498x566` → `996x1132`: `fallin_strong-directml` (no CPU fallback), `fallin_soft` (CUDA graph), and `fallin_strong-tensorrt` (engine built at `500x568`).
- `--preview` pinned a full CPU core, and the GIL with it, whenever the preview page was open before the first preview frame existed — which is exactly what happens when you click the URL TAS prints at startup and then wait through model init or a TensorRT engine build. `PreviewHTTPHandler._serveStream` seeded `lastSeq = -1` while `PreviewSink` starts at seq 0, so `waitNext` never parked on its condition and the "did the sequence advance" guard could never be satisfied; the handler spun. Measured on a 1080p `--dedup` run: 275 fps with a client attached from t=0 against 506 with none (-45%). The handler now seeds from the sink's initial seq and blocks. `tests/test_preview.py::test_streamHandlerDoesNotSpinBeforeFirstFrame` fails on the old code.
- `RifeDirectML` set `torch.set_default_dtype(torch.float16)` globally and never restored it, so every torch module constructed after a half-precision DirectML/OpenVINO RIFE in the same process silently came out fp16 — including later pipeline stages. It was also redundant: the archs take an explicit `dtype` for their buffers and the model is cast with `.half()` immediately after. Removed; output is bit-identical on `rife4.6`/`rife4.22-lite`/`rife4.25`/`rife_elexor`-directml.
- DistilDRBA on any input above 1080p padded to the wrong multiple. Both `DistilDRBACuda` and `DistilDRBATensorRT` hardcoded `padSize = 64`, but at `scale = 0.5` the pyramid's coarsest level is 1/32 and each `IFBlock.conv0` floors the spatial size by 4 before `lastconv` upsamples it back, so the padded frame has to stay mod-128. Padding to 64 left the pyramid a level short: `distildrba-lite` raised `Sizes of tensors must match except in dimension 1` and `distildrba` silently produced a misaligned flow (sometimes surviving with the right output shape, sometimes emitting the wrong width). This hit every >1080p resolution whose mod-64 size is not also mod-128 — 2560x1440 pads to 1472, and `1472 % 128 == 64`. 4K only escaped because 3840x2176 happens to be mod-128 on both axes. The padding multiple now follows the scale (64 at 1.0, 128 at 0.5); ≤1080p output is unchanged.

### Changed
- Upgraded PyTorch to 2.13.0 and torchvision to 0.28.0 across every `extra-requirements-*.txt` (from 2.12.1/2.11.0 and 0.27.1). Windows keeps `+cu132`, Linux keeps torch `+cu130` with torchvision `+cu132`, and the macOS/`*-lite` files stay on the plain PyPI wheels; cp314 wheels for all three suffixes are published on the PyTorch index and PyPI.
- `--stabilize` no longer recomputes ORB features for a frame it already analysed. `_analyzeFrames` feeds consecutive frames as `(prevGray, currGray)` and threads `prevGray = gray`, so frame N is this call's `currGray` and next call's `prevGray` (same object); the old `_estimateTransformORB` ran `detectAndCompute` on **both** every iteration, so every interior frame's resize + ORB extraction was computed twice. It now caches `(keypoints, descriptors)` for the frame just processed as `curr`, keyed on object identity, and reuses it as `prev` next iteration — each frame's `detectAndCompute` runs once. Bit-identical transforms/scores (ORB extraction is deterministic; the identity key means any miss simply recomputes, never serves stale features): isolated parity is exact over 119 real consecutive pairs, and the analysis pass is the whole cost of a stabilize run (the render pass is ~300 fps against ~38). Measured +33% end-to-end on a 300-frame 640×360 clip (65.7 → 87.6 fps, 8 interleaved pairs). This is the only ORB codepath that runs today — the SuperPoint path `_estimateTransform` prefers never initializes (its checkpoint at `SUPERPOINT_WEIGHTS_URL` uses `backbone.*.conv/bn` submodule names while the in-repo `SuperPoint` expects a `Sequential` `backbone.*.0/2` layout, so `load()` raises and every run falls back to ORB/LK).
- `--obj_detect` draws annotations straight onto the RGB frame instead of round-tripping it through BGR. `readBuffer` delivers RGB, but each detected frame did `cv2.cvtColor(RGB2BGR)` → draw → `cv2.cvtColor(BGR2RGB)` before writing — two full-frame colour conversions per detected frame (~1.6 ms at 1080p, ~6 ms at 4K) whose only purpose was cv2's BGR drawing convention. `draw_detections`/`draw_masks` now take a `palette`, and the callers pass a once-computed channel-reversed palette (`colors_rgb = colors[:, ::-1]`) so overlay pixels land byte-identical after the swap is removed; `addWeighted`/`rectangle`/`putText` are all per-channel so the reversal commutes exactly with the dropped conversion. Output is **bit-identical** (verified on a real 4K frame across both the annotated and `disableAnnotations` paths, and by `tests/test_object_detection_fastpaths.py`). Measured +10% end-to-end at 4K (`yolov9_small-directml`, 17.7 → 19.6 fps, 6 interleaved pairs).
- Startup metadata no longer shells out to `ffprobe`. `getVideoMetadata()` runs once per input in `VideoProcessor.__init__` (pure startup latency, before any decode); it used to spawn `ffprobe -show_format -show_streams`, and previously `-count_packets` on top of that (a full-container demux). It now reads everything from a decode-free `nelux.probe(path)` — nelux opens the container and reads stream info only (no decoder, no resolution-sized buffer, no threads) and returns the same libavformat metadata (width/height, exact `r_frame_rate`, `nb_frames`, `codec`, `pixel_format`, `color_primaries/transfer/range/space`, `duration`, `has_audio`) so no field is lost. (Falls back to a `VideoReader.get_properties()` read on `nelux` builds without `probe()`.) Verified field-for-field against ffprobe on h264/no-audio, ProRes 4:4:4/audio, and a BT.709 `color_range=tv` source (color tags read correctly, not `unknown`). `nelux.probe` measured **~0.6 ms at 360p, ~4 ms at 1080p, ~17 ms at 4K vs ffprobe's ~20–36 ms** (about 17 ms of every ffprobe call is pure process/DLL-load spawn that the in-process probe skips entirely; the old `-count_packets` form cost another ~160–220 ms on 400–700 MB inputs, seconds on multi-GB). It never opens a decoder, so it is also immune to the nvdec-construction deadlock on uncompressed/odd codecs. FFmpeg DLLs are already on the search path (`src/cli/startup.py`) before this runs. `nb_frames` drives the progress-bar total (self-corrected at stream end), with a `duration*fps` fallback for VFR/headerless sources. Requires `nelux>=0.15.1` (adds `probe()`; `>=0.15.0` still works via the `get_properties()` fallback); `ffprobe` is no longer invoked for metadata.
- `--preview` no longer goes through the encoder. ffmpeg used to carry a second output — `split=2` in a `filter_complex`, an `fps=2` throttle, an mjpeg encode, and a `pipe:1` stream a drain thread reassembled — so preview was billed to the encode whether or not anyone was watching, and the `filter_complex` branch quietly dropped the NVENC `hwupload_cuda` path and stripped `-vf` out of a `--custom_encoder`. A `PreviewSampler` (`src/server/previewSettings.py`) now samples the frames the writer already holds, and the ffmpeg command is byte-for-byte the same with preview on and off (verified: the output file is bit-identical). Two consequences: the sampler is **demand-gated**, so with the page closed the cost is one `wants()` call per frame (238 ns, 0.012% of a core at 500 fps) and nothing else; and the throttle is now **wall-clock**, not video-time, so a run at 20× realtime emits 2 preview frames/sec rather than 40 that nobody sees. The preview is still served at the output resolution — the JPEG encode runs on a worker thread and costs 0.6% of one core at 1080p, which is not worth degrading the picture to save. On a 1080p `--dedup` run: 519 fps baseline, 521 with `--preview` and the page closed, 505 with it open (was 506 / 488 / 275). Preview traffic over a run: 5.3 MB, down from 31.8 MB (same picture, emitted at wall rate instead of 20× it).
- `--preview` works with the `*_nelux` encoders. It was hard-disabled there (`NeluxWriteBuffer.enablePreview = False`) because nelux emits no preview stream — irrelevant now that preview is sampled from the frame rather than produced by the encoder.
- `importRifeArch`/`_RIFE_V1` lived as six verbatim copies (`rife.py`, `rife_directml.py`, `rife_tensorrt.py`, `rife_ncnn.py`, `distildrba.py`, `_shared.py`) — of which `rife_ncnn.py` and `distildrba.py` never called it and `_shared.py` was imported by nobody. It is now defined once in `src/interpolate/_shared.py` and imported by the three backends that use it, so adding an arch is one edit rather than five.
- The interpolation model tables in `README.md` and `PARAMETERS.MD` listed Rife Elexor as DirectML- and OpenVINO-capable long before it was. That is true as of this release, but the tables were asserting it while the CLI had no such choice.
- `test_frameWindow`'s DistilDRBA cross-domain-neighbour guard asserted a hard-coded `== 2` occurrences of the error string, so adding a third backend failed the test for the wrong reason. It now walks the AST and requires the guard on every `DistilDRBA*` class, which is what it was actually trying to check.

## [2.8.0] - 2026-07-11
### Added
- Apple Silicon MPS depth backends for the image-depth methods: v2 (`small_v2-mps`/`base_v2-mps`/`large_v2-mps`/`giant_v2-mps`), distill v2 (`distill_*_v2-mps`), og v2 (`og_*_v2-mps`), and v3 (`small_v3-mps`/`base_v3-mps`/`large_v3-mps`/`og_large_v3-mps`). These mirror the CUDA image-depth flow on macOS and support `--depth_batch N` for batched per-frame inference on MPS; `video_*` depth methods remain sequential and unchanged.
- `--mask` now applies to `--interpolate`, not just `--moblur`. Protected pixels — HUDs, hardsubs, watermarks, logos — are restored from the source frame preceding each generated frame instead of being warped by RIFE/GMFSS optical flow. A shared `src/masking.py` (`ProtectionMask`) backs both features, applied at the sink both interpolation orders push into, so every backend is covered without touching a backend class. Overhead ~0.97 ms per generated frame at 1080p, exactly zero when `--mask` is unset.
- `--scenechange` (opt-in): a per-frame-pair detector holds the frame at hard cuts instead of letting RIFE morph across them. `--scenechange_method` selects the cheap `ssim`/`mse` detectors or the `maxxvit` classifier; `--scenechange_sens` sets the threshold. Output frame count is unchanged.
- `--depth_batch N` (default 1): batch N frames per forward for the CUDA and TensorRT image depth methods (v2, distill, `og_*_v2`, v3). Amortizes kernel-launch overhead where the GPU is launch-bound — `small_v2` fp16 at `--depth_quality low` goes 175.6 → 257.3 FPS at N=4 (RTX 3090), while `high` is already compute-bound and gains nothing. `N=1` matches the previous path exactly. Not applied to the temporal `video_*` methods or the DirectML/OpenVINO backends. TensorRT batching needs a dynamic-batch ONNX (re-exported ones are hosted on TAS-Models-Host).
- `--segment_batch N` (default 1): the same for anime segmentation. CUDA resizes into a fixed 1024-box so its gain is resolution-independent (24.2 → 26.7 FPS end-to-end at N=4); TensorRT runs at native resolution, so it wins only where launch-bound (93.8 → 147.1 FPS at 640×360, flat at 720p). Forced to 1 on DirectML/OpenVINO, where batching measured slower.
- `--depth_window {32,16,8,4}` (default 32): temporal-attention window for `video_small_v2`/`video_large_v2`. 16 trims the attended context for ~+14% end-to-end with no added flicker (SSIM 0.997 vs the full window); 8 and 4 trade fidelity for speed. CUDA-only.
- `--depth_method large_v3`: Depth Anything 3 Mono Large (ByteDance, Apache-2.0), alongside `small_v3` and `base_v3`. The metric variant remains `og_large_v3`.
- `--depth_method video_small_v2-tensorrt`: TensorRT backend for temporal Video Depth Anything. It re-runs the full 32-frame window for every output frame, so it is **slower** than the CUDA streaming path (11.9 vs 29.7 FPS on a 300-frame 640×360 clip). Kept as an option; `video_small_v2 --half` stays the recommended default. Also fixes the self-overlapping `copy_` that made this path write only its first frame.
- NCNN (Vulkan) RIFE restored on cp314: `--interpolate_method rife*-ncnn` runs on any Vulkan GPU with no CUDA/TensorRT dependency, on Windows, Linux, and macOS (Apple Silicon).
- macOS now installs FFmpeg for you: when none is found on `PATH` (or under `/opt/homebrew/bin`, `/usr/local/bin`, `/usr/bin`), TAS runs `brew install ffmpeg` on first run instead of failing with an instruction to do it yourself. Homebrew must already be installed; TAS will not install Homebrew itself, and says so if it is missing.

### Changed
- `--depth_norm` now removes per-frame monocular scale/shift ambiguity without compositing historical depth pixels. A stronger motion-compensated history-fusion prototype was rejected after real footage exposed stamped scene textures and structural trails. For `video_*` methods, the flag now preserves Video Depth Anything's shared coordinate system with a slowly adapting scene-wide range instead of independently min/maxing every frame. On `Raw.mp4`, `video_small_v2 --half --depth_norm` lowers encoded flow-aligned error from 0.04137 to 0.01809 (−56.3%) while retaining edge energy, and produced the visually preferred `Raw-Depth-7.mp4`. Invalid DA3 depth is also excluded from normalization.
- The macOS portable build no longer bundles FFmpeg. `bundle_macos_ffmpeg()` used to copy Homebrew's `ffmpeg`/`ffprobe` and their entire dylib closure into `ffmpeg_shared/`, which made every macOS release asset a redistribution of GPL binaries — without the accompanying licence texts or offer of corresponding source. TAS now ships no FFmpeg bytes on any platform: Windows and Linux already downloaded it at runtime, and macOS installs it through Homebrew on the user's machine. Because the macOS nelux wheel links FFmpeg's shared libraries at absolute Homebrew paths, a Homebrew FFmpeg is what nelux needs anyway.
- **Breaking:** `--moblur_mask` is renamed to `--mask` and moved out of the Motion Blur group into its own Masking group, since it is no longer motion-blur specific. Scripts and presets passing `--moblur_mask` must be updated.
- The frame loop's `currentFrame`/`nextFrame` read pair is now a bounded ring buffer, `FrameWindow` (`src/io/frameWindow.py`). Drivers that need neighbouring frames declare `temporalWindow = (past, future)` as a class attribute, and the pipeline sizes the ring to the widest demand among its active stages — the lookahead now travels with the driver that needs it instead of living in a hardcoded method-name check in `main.py` (see *Fixed*, below, for what that check cost). A pipeline with no temporal driver reads no frame ahead at all, holding one fewer decoded frame (~12 MB at 1080p fp16). Throughput is unchanged: 135.7 vs 133.8 FPS mean over 4 interleaved rounds (900-frame 640×360 `shufflecugan`, RTX 3090), inside run-to-run spread. Output is bit-exact for every configuration that was already correct — `animesr` (CUDA), `distildrba`, non-temporal upscalers, and their combinations with `--restore`, `--dedup`, `--scenechange`, and `--interpolate_first`.
- A driver's neighbour frame is now guaranteed to be in the same domain as the frame it neighbours, and to belong to the same shot. The window has an *entry* stage (dedup, then restore) that every frame passes through before it can serve as anyone's context, so a temporal upscaler or `distildrba` is handed a restored/upscaled neighbour rather than the raw decoded one it used to get. Downstream upscaling in the interpolate-last order is memoized per frame, so a frame read by two stages is still upscaled exactly once, in frame order (keeping AnimeSR's recurrence intact). And under `--scenechange`, `distildrba`'s future frame is withheld across a hard cut — previously it pulled a frame from the *next* shot as motion context for the last generated frame before the cut. On a 37-frame clip with one cut, exactly one output frame changes (the interpolated frame straddling the cut boundary); everything else is bit-identical. Frames that dedup drops never enter the window at all, so no restore work is spent on them and no driver ever sees a duplicate of the frame it already holds.
- Removed `BuildBuffer.peek()` and the `threading.Event` that existed only to serve it; nothing had called it since `distildrba` switched to being handed its next frame. The `--depth video_*`, `--depth *-directml`, `--segment`, and `--obj_detect` process loops each carried a copy-pasted one-frame lookahead that they never read, pinning an extra decoded frame for nothing; they now read one frame at a time.
- Video depth methods (`video_*`, `og_video_*`) encode through the standard `WriteBuffer` — honoring `--encode_method`, `--bit_depth`, and `--custom_encoder` — instead of a hardcoded 8-bit `cv2.VideoWriter`. Depth fidelity against a lossless reference rises from PSNR 44.8 / VMAF 90.8 to PSNR 53.5 / VMAF 95.0 at ~57% smaller files; end-to-end FPS is unchanged, since encode was never the bottleneck.
- Bumped TensorRT `10.16.1.11` → `11.1.0.106`, moving from the hosted cp314 wheels to NVIDIA's current PyPI metapackage. Imports go through a compatibility helper so both the traditional `tensorrt` module and the 11.1 `tensorrt_bindings` layout work.
- `--preview` no longer round-trips through a `preview.jpg` on disk. The writer's ffmpeg now emits the throttled preview as an mjpeg stream on its stdout; a drain thread parses it and pushes the latest JPEG into an in-memory `PreviewSink`, and the preview server hands it to the browser over a `multipart/x-mixed-replace` stream (`/stream`) instead of a 1 Hz `<img>` poll of a file (`/image` is kept as a one-shot fallback). No disk file is written or deleted, torn-frame reads are gone, and the browser is pushed every emitted frame rather than sampling one per second. End-to-end pipeline throughput is unchanged — 720p `--interpolate` runs 371 FPS with preview on vs 382 off (median of 2, RTX 3090), inside the off-runs' own spread. In an isolated 1080p harness the mjpeg pipe matches the old file path on ffmpeg CPU, ffmpeg RAM, and encoder throughput, costing ~0.17 extra host core (the drain thread) and ~4 MB. The `pipe:1` stream **must** be drained unconditionally or ffmpeg blocks on a full OS pipe buffer and back-pressures the encoder, so the drain thread runs whether or not a browser is attached. Nelux encoders still emit no preview.


### Performance
- `--dedup_method ssim` (the default) runs its fp32 SSIM on the GPU when CUDA is available instead of shipping every decoded frame back to the CPU for the 224×224 Gaussian convolutions. The CPU path saturated ~16 cores while the GPU sat at 9% utilization; end-to-end dedup goes 327.9 → 1070.4 FPS (median of 3, 720p, RTX 3090) with bit-identical output (same fp32 math, verified: zero decision flips across 3416 frames on two clips, max SSIM-score delta 2.4e-7, and byte-identical encoded output vs the CPU path). CPU-only machines keep the exact previous path.
- fp16 fast path for `video_small_v2`/`video_large_v2`: DINOv2 attention through `F.scaled_dot_product_attention`, LayerScale folded into the projections, and a cached pos-embed interpolation. **+51% model forward** and 1.69–1.93× end-to-end. fp32 output is unchanged (SDPA is fp16-gated, the folds are algebraically exact); `og_video_*` stays as the fp32 baseline. The temporal motion-module attention was deliberately left off SDPA — it measured −10%.
- The FFmpeg-subprocess `WriteBuffer` writes each frame's tensor to ffmpeg stdin via the buffer protocol instead of `.numpy().tobytes()`, removing one full-frame host copy per frame (~24 MB at 4K). Bytes are identical.
- Startup hardware detection no longer shells out to WMI; compute capability is read from NVML directly. Non-NVIDIA machines that previously hit the WMI fallback drop ~194 ms → ~0.1 ms. Backend selection is unchanged.
- `BuildBuffer.processFrameToTorch` normalizes the fresh CUDA tensor in place and drops the ad-hoc per-frame pin. The 640×360 `rife4.6` 2× benchmark improves ~890 → ~994 FPS in fp16; pixel output verified unchanged.
- RIFE CUDA skips refilling the timestep tensor when its value has not changed, trimming steady-state overhead for 2× interpolation (~1.66 → ~1.59 ms per fp16 call at 640×360).
- The `rife4.25`, `rife4.25-heavy`, and `rife4.25-lite` CUDA/MPS paths padded every frame to a multiple of 128. Their coarsest IFBlock runs at `scale_list[0] = 16` and its `conv0` downsamples 4x more, so a multiple of **64** is sufficient at `scale == 1` — 1080p was padded to 1152 rows instead of 1088, spending ~6% of every forward pass on 64 rows of black. The multiple is now derived from the arch and the scale actually in use. Model forward at 1080p fp16 goes 10.86 → 10.26 ms (`rife4.25`) and 17.26 → 16.27 ms (`rife4.25-heavy`); end-to-end 2x on a 300-frame 1080p clip goes 161.7 → 170.2 FPS and 106.3 → 113.2 FPS (min-of-9 interleaved A/B, RTX 3090). Quality improves slightly rather than regressing: on the 412 anime triplets of ATD-12K (`Japan_v2_*`, drop-middle, paired bootstrap over per-triplet differences) `rife4.25` gains +0.062 dB PSNR (95% CI [+0.033, +0.096]) and +0.00024 SSIM, `rife4.25-heavy` +0.053 dB (95% CI [+0.031, +0.079]) and +0.00022 SSIM — the padded black border was costing accuracy at the coarse scales, not buying any. The 128 remains in force where it is genuinely required: above 1080p (`scale = 0.5`) and under `--dynamic_scale`, which re-picks scale per frame down to 0.5 and needs `4 x 16 / 0.5 = 128`. The `8 / scale` arches (`rife4.6`, `rife4.22`, …) keep their existing 64: dropping them to their architectural floor of 32 changes nothing at 1080p (which pads to 1088 under both) and measured −0.038 dB on `rife4.22`.


### Fixed
- Interpolation combined with upscaling built its model at the wrong resolution in one of the two pipeline orders, and every RIFE/DistilDRBA/GMFSS backend sizes fixed I/O buffers (and, on CUDA, a captured graph) at construction — so the mismatch was not survivable. `initializeModels` passed the interpolation factory the *post-upscale* dimensions unconditionally, but only GMFSS used them; RIFE and DistilDRBA used the *source* dimensions. The result was two mirror-image failures: interpolate-last (`--interpolate_first False`) with RIFE or DistilDRBA fed the input-sized model upscaled frames and died in the buffer copy (the run then hung on the wedged encoder); interpolate-first (the default) with GMFSS fed the output-sized model source frames and hit GMFSS's own shape assertion. The interpolation model is now built at the resolution it will actually receive — source dimensions in interpolate-first order, post-upscale dimensions in interpolate-last order — resolved once in `initializeModels` and passed to every backend uniformly. Interpolate-first without upscale and all upscale-free paths are byte-for-byte unchanged (the resolution handed over is identical); the two broken combinations now produce correct full-resolution output.
- `distildrba` no longer silently bilinearly resamples its future frame (`I2`) to match the current one. That resize only ever fired when the two were in different domains — an upscaled current frame against a source-resolution neighbour — and it papered over the mismatch by feeding the model a blurred, wrong-scale motion cue. The window now supplies the neighbour in the current frame's own domain, so the sizes always agree; a mismatch is a wiring bug and raises instead of degrading the output quietly.
- AnimeSR got no future frame on three of its four backends. AnimeSR is a recurrent 3-frame architecture: it consumes the previous, current, and next frames. `main.py` handed it the next frame only behind `upscaleMethod == "animesr"`, an exact string match, so `animesr-tensorrt`, `animesr-directml`, and `animesr-openvino` (the last two share `AnimeSRDirectML`) always received `nextFrame=None` and each silently substituted the current frame for it — running the model with `next == curr` and no forward temporal context at all. The window is now declared on the backend class, so every AnimeSR backend gets it. Output on those three backends changes, and moves toward the CUDA path that was always correct: measured against `animesr` (CUDA) on a 37-frame 640×360 clip, mean absolute per-pixel difference falls 0.249 → 0.134 for TensorRT and 0.247 → 0.135 for DirectML, with max deviation 18 → 7 of 255. The residual is TensorRT/DirectML-vs-PyTorch numerics. CUDA `animesr` output is bit-identical to before.
- `--dynamic_scale` did nothing on the fp16 CUDA path — the only path it is documented to support. `rife_fast._FastIFNet` stored `self.dynamicScale` and never read it: its `scale_list` was computed once in `__init__` and `forward` just indexed it, so the per-frame scale was never re-picked. Every non-fast arch (`IFNet_rife4*.py`) honoured the flag, so `--dynamic_scale` worked in fp32 and silently no-opped in fp16, while `RifeCuda` still disabled CUDA-graph capture on its behalf — users paid the cost and got none of the effect. The fast arch now rebuilds `scale_list` from the per-pair scale exactly as the reference arches do, and its output matches a statically-configured coarse scale bit for bit. Interpolation without the flag is untouched (171.2 vs 170.2 FPS, 2x 1080p, within run-to-run noise) and still CUDA-graphed.
- `dynamicScale()` cached a single global SSIM module and bound it to the device **and dtype** of the first caller. Its gaussian window is a registered buffer, so any later call with a different dtype died with `RuntimeError: expected scalar type Half but found Float`. This was latent only because the flag never reached the fp16 path (above). The module is now cached per `(device, dtype)`.
- GMFSS ran only the first of RIFE 4.6's four IFNet blocks. `src/gmfss/model/IFNet_HDv3.py` was re-added in October 2024 without the refinement loop's `else:` branch, so `flow` was non-`None` after the first iteration and blocks 1–3 were never reached — the loop still spun four times, recomputing identical warps. Two later cleanups, each correct given the last, then deleted the unreachable blocks and filtered `rife.pkl` down to `block0.*` on load. Because `fusionnet_union.pkl` was trained against the full four-block output, the union GridNet had been fed an off-distribution hint (a coarse scale-8 flow) ever since. All four blocks run again, matching upstream GMFSS-Fortuna. Scored against held-out ground truth on 40 motion-selected 1080p triplets (scene cuts excluded): PSNR 30.19 → 30.62 dB, SSIM 0.94536 → 0.94629, and the RIFE hint itself 27.05 → 27.99 dB. Cost is +2.6 ms per generated frame (RTX 3090, 1080p fp16) — −1.2% end-to-end at 2×, −2.2% at 4×, −3.4% at 8× — because `reuse()` (GMFlow) dominates a pair at ~166 of 212 ms. The 2.4M restored parameters were already shipped in `rife.pkl` and discarded on load.
- A dying encoder deadlocked the whole run instead of failing it. When FFmpeg (or the Nelux encoder) exited during startup or mid-stream — e.g. `--segment` into an `.mp4`, whose container rejects the RGBA `prores_ks` stream — the writer thread returned, nobody consumed the frame queue, and the producer blocked forever on `put()` to a full queue; the process had to be killed by hand. Both write buffers now drain the queue until the producer's close-sentinel on early-exit paths, so such runs finish with a clean non-zero failure instead of hanging. Successful runs are untouched (the drain only arms when the sentinel was never consumed).
- TAS exited `0` on every failure, so no shell, batch script, or CI job could tell a finished render from a dead one. Two causes: `VideoProcessor.start()` stored frame-loop exceptions on `self.processingError` and returned normally, so the batch loop's `except` never fired; and ten validation paths called bare `sys.exit()`, which exits 0 — including "No processing methods specified" and "Invalid JSON format in config file". Error paths now exit `1`, informational exits (`--help`, `--list_methods`, `--list_presets`) stay `0`, and argparse keeps `2`. A failed video is counted and forces a non-zero exit; the batch summary reports `N succeeded, M failed` instead of counting failures as successes. `KeyboardInterrupt` keeps its `os._exit(130)`.
- The After Effects panel was told a failed render had **completed**. `setCompleted` ran after the `processingError` branch in `start()`, so the panel then looked for an output file the starved encoder never wrote — the mechanism behind the recurring "Output file not found after processing" reports (issues #269, #236, #231). Failed or 0-byte runs now emit `setFailed`. Benchmark runs, which write no output by design, are exempt.
- An applied `--preset` silently overwrote flags typed on the command line: `--preset myquality --upscale_method span` ran the preset's method instead. The preset stored `vars(args)` — every flag — and reapplied all of it with an unconditional `setattr`, before the set of explicitly-provided options was even known. Precedence is now explicit CLI flag > preset > default. Saving a new preset is unchanged.
- Silent CPU fallback is now visible. DirectML/OpenVINO fallback notices used `logging.info`, but the root logger has no `StreamHandler` and writes to `TAS-Log.log` only, so the console said nothing while users ate an unexplained ~30× slowdown. Those notices now use `logWarning`. A new `src/infra/providerCheck.py:warnIfProviderMissing()` also catches ONNX Runtime accepting a requested GPU provider and then placing every node on CPU. Observability only — provider lists, device placement, and dtype are unchanged, and the inference-time handlers warn once rather than per frame.
- `--depth_batch N` with `N > 1` hung forever whenever the frame count was not an exact multiple of `N` — i.e. on almost every real video. `BuildBuffer` queues exactly one `None` end-of-stream sentinel; the batch loop consumed it on the short final batch and then blocked on a `read()` that could never return. All four batched depth loops now terminate on a short read instead of an empty one. `N=1` was never affected, which is why 240-frame (`240 % 4 == 0`) validation missed it.
- Startup disables per-user Python site packages before importing runtime dependencies, keeping bundled TAS dependencies ahead of user-installed packages that would otherwise shadow the packaged environment.
- Backend logging writes to a stable `TAS-Log.log` in overwrite mode again instead of creating PID-suffixed files per launch, which let Adobe render sessions accumulate hundreds of stale logs.
- TensorRT upscale engine creation now logs GPU/driver/VRAM diagnostics, retries a failed dynamic build once with a static input profile to reduce build-time VRAM pressure, and raises a clear build failure instead of crashing later with `'NoneType' object has no attribute 'num_io_tensors'`.
- TensorRT builds no longer abort on legacy Windows code pages (e.g. cp1252): the box-draw input-shape log raised `UnicodeEncodeError` mid-build, was caught as an optimization-profile failure, and resurfaced as the same `NoneType` error. Console streams are reconfigured to UTF-8 in `logAndPrint`, and the cosmetic shape print can no longer abort a build.
- Metadata probing validates the configured `ffprobe` binary before startup probing, instead of checking `ffmpeg` and failing later with a less useful subprocess error.

## TAS Adobe Edition
### New
- Media Browser: **folder navigation**. Subfolders now appear as tiles in the grid — folders that contain media lead the grid with a highlighted open-folder icon, empty ones trail — with Back/Forward/Up buttons and Explorer-style shortcuts (Alt+Left / Alt+Right / Backspace). Replaces the old "Include subfolders" recursive-scan checkbox.
- Media Browser: **previews for any codec.** Videos CEP's Chromium can't decode natively (HEVC, AV1, exotic containers/audio) now play through a live ffmpeg fragmented-MP4 pipe with zero temp files — h264/vp9 sources remux without re-encoding, and seeking restarts the stream at the target time.
- Media Browser: **Windows shell thumbnails.** Grid thumbnails now come from the OS shell thumbnail cache (the same thumbnails Explorer shows) via one long-lived hidden PowerShell worker — a cache read takes ~10-30 ms and writes nothing to disk. ffmpeg stays as the fallback (and the only path off Windows).
- Media Browser: the preview is a **lightbox** (click-outside or Escape to close, fade in/out) instead of a modal dialog. The grid thumbnail crossfades into the first video frame, an Import button sits inside the preview, and volume is a hover flyout with a vertical slider.
- **Scene-cut aware interpolation**: a new *Scene-cut Aware* toggle in Interpolate Settings holds the frame at hard cuts instead of letting RIFE morph across them (the output frame count is unchanged). Pick the detector (SSIM/MSE, cheap and model-free, or the MaxxVit classifier) and the sensitivity. Sponsor-gated detectors fall back to the CPU `ssim` detector for free-tier users.
- **The protection mask now applies to interpolation, not only motion blur.** The mask-layer picker is available in Interpolate Settings: opaque pixels on the picked layer — HUDs, hardsubs, watermarks, logos — are restored from the source frame instead of being warped by optical flow.
- **NCNN (Vulkan) interpolation is back**: a new *ANY VULKAN CAPABLE GPU* group in the interpolation model picker exposes the RIFE NCNN models. They run on any Vulkan GPU with no CUDA/TensorRT dependency and are free for everyone. This also fixes the built-in "Instant 4K 60fps AMD/Intel" preset, which selected `rife4.22-ncnn` — a model the picker had no entry for.
- **Depth Anything V3** models (Small, Base, Large, and the metric OG Large on CUDA; Small/Base on TensorRT and DirectML) and the **fp16 fast-path video depth** models (`video_small_v2` / `video_large_v2`, 1.7–1.9× faster end-to-end than the OG video models) are now selectable in Depth Map Settings, along with the TensorRT video-depth backend.
- **Depth and Segmentation batch size**: process several frames per model forward where the GPU is launch-bound, at a proportional VRAM cost. The control is hidden on the backends where the backend forces batching off (DirectML/OpenVINO, and the video depth methods).
- **Depth temporal window**: choose the attention window (32/16/8/4 frames) for the fp16 CUDA video depth models. 16 is meaningfully faster with no added flicker.
- Keyframe Graph Editor: an animated **preview bead** rides the easing curve (X advances linearly in time, Y follows the eased value) so the curve's acceleration/deceleration is visible at a glance. Toggled from the toolbar; loops with a short hold, and never redraws through React (canvas-only).
- Keyframe Graph Editor: **Copy as code** popover exports the current curve as CSS `cubic-bezier()`, CSS `linear()` (sampled, so it also represents overshoot/bounce a single bezier can't), or GSAP `CustomEase` — and a paste field loads a `cubic-bezier(...)` / `a,b,c,d` string back into the editor.
- Keyframe Graph Editor: **Platform** preset pack with real Material, iOS, Fluent 2, and Android platform easing curves.
- Keyframe Graph Editor: **Read** now names the closest matching preset in the toast (e.g. `(≈ Cubic Out)`) and auto-orients the graph to the selected keyframes' direction — descending pairs (e.g. 100 → 0) draw the graph flipped so it visually descends like AE's value graph. Orientation is view-only (the applied bezier is unchanged) and only re-evaluated on Read, never mid-drag.

### Changed
- **Requires the matching backend release.** The backend renamed `--moblur_mask` to `--mask`; the panel now sends the mask under the new key, so an older backend will ignore it and log an unknown-option warning.
- Backend log tracking follows the backend's new per-PID log files (`TAS-Log-<pid>.log`): the panel watches the install folder and binds to the newest matching log instead of a fixed `TAS-Log.log` (which older backends still get for compatibility).
- macOS installs now use the single `macos-mps` dependency profile — the backend has no `macos-lite` — and dependency install runs one `--download_requirements` pass instead of two chained ones.
- The bundled Python interpreter is now resolved per platform (`python.exe` at the runtime root on Windows, `bin/python3` on macOS/Linux) instead of assuming a root-level binary; process cleanup also matches `python3`, and the managed-install cleanup recognizes the unix prefix layout (`bin`/`lib`/`share`, `run.sh`, `requirements.txt`).
- The download-size tile shows per-platform sizes (~45 MB on Windows, ~70 MB on macOS) instead of a stale ~35 MB.
- Media Browser: quick access is a compact chip row (pinned + recent) above the breadcrumbs instead of a separate card, and the type filter is a segmented icon toggle instead of a picker.
- Media Browser: the grid remembers each folder's scroll position and last scan across tab switches, keeps an in-memory LRU thumbnail cache, sizes itself from the measured viewport (no more scrollbar flash on tab switch), and debounces thumbnail batches so scrolling doesn't stutter. The "Generating previews" indicator floats over the grid instead of shifting the layout.
- Run Chain is now an accent-filled button and Generate Preview a secondary button (both were monochrome action buttons), and sponsor-gated picker sections say "(Sponsors only)" instead of a lock emoji.
- Removed the phantom `rife-ncnn` picker entry (the model doesn't exist in the backend); the NCNN auto-reset fallback is `rife4.22-ncnn`.
- Account section badges (sponsor status, owned tiers) use Spectrum Badge components, and the deprecated `cta` button variant migrated to `accent` throughout.
- The bottom progress footer takes generic percent/status/details props, shows a full-width indeterminate sweep when work has started but no progress data exists yet, and animates its expand/collapse.
- The panel body uses an overlay scrollbar so the navbar no longer shifts sideways when switching between scrolling and non-scrolling tabs; the dead pre-rewrite toast CSS is gone.
- Motion Blur and Interpolate now share one mask-layer picker and one stored selection, so a mask picked in either place applies to whichever of the two runs.
- Keyframe Graph Editor: removed the manual **Flip Y** toggle (toolbar button and `Shift+V`) in favor of the automatic Read-driven orientation above; also removed the now-unused toggle-binding customization row from the shortcuts dialog. The bezier text input and Read/Apply are unchanged.
- Panel-written debug logs now consolidate into a `tas-log` folder inside the TAS install directory (next to the backend `TAS-Log.log`) for easier debugging: the crash log (`tas-log/crash.log`) and the ExtendScript dead-frame dedup breadcrumb (`tas-log/dedup_debug.log`) no longer scatter into the OS temp folder. Both fall back to the temp folder when the install directory doesn't exist yet or the CEP Node bridge is unavailable (e.g. Vite browser dev).
- Read and Apply buttons in the Keyframe Graph Editor now share equal width.
- Consolidate duplicated version comparison into shared `versionCompare.ts`.
- Rename `clearCompleted` → `clearFinished` (removes completed + failed).
- Performance (no behavior change): queue status counts and per-tick stats now compute in a single pass instead of several throwaway array allocations; the virtualized queue list uses a stable row renderer and `rowData` so react-window stops re-rendering all visible rows on every progress tick.
- Performance (no behavior change): the sponsor-status empty caps array is now module-scoped, restoring `ChainTab`'s `React.memo` (it was defeated by a new array identity every render, cascading through the chain callbacks).
- Performance (no behavior change): Media Browser sorting selects its comparator once instead of re-branching inside every comparison (~28% faster on large recursive listings, measured); Keyframe Graph Editor preset filtering merges its lists and lowercases the search query once per render; `QueueItemRow` is memoized with hoisted static styles.

### Fixed
- Cancelling the current queue item now marks it **cancelled** instead of **failed** — the kill's non-zero exit code was being read as a failure — and the rest of the queue continues.
- Media import failures now show the host's actual complaint: After Effects and Premiere return the first concrete failure reason (missing file, unsupported format, could not insert into the sequence) and the toast displays it instead of a generic "could not import".
- YouTube downloader: the URL field is a controlled input with a proper validation error message instead of shipping a placeholder URL as its default value, and the resolution dialog closes reliably (DialogContainer instead of a hidden-trigger hack).
- Keyframe Graph Editor preset tiles are keyboard-accessible (Enter/Space, focusable), and shortcut capture uses a real toggle button.
- **RIFE Ensemble and Dynamic Flow Scale were silently dropped from queued renders.** Both checkboxes were only wired into the direct chain command, never into the config the queue actually sends, so a queued job ran with neither. They now travel with the job. (The backend also fixed `--dynamic_scale` no-oping on the fp16 CUDA path, so the toggle now does something in both places.)
- Interpolating with a protection mask no longer bypasses the pre-render when "use layer in/out points" is on. The mask PNG is rendered in comp space, so processing source-dimension frames instead would misalign it against the input.
- "Open TAS folder" failures now show a red error toast instead of a green success toast (`generateToast` was passed `Positive` instead of `Negative`).
- "Open TAS folder" no longer throws `existsSync is not a function` when the CEP Node bridge is unavailable — `fs`/`child_process`/`os` members and an empty install path are now guarded, surfacing a clear message instead of the cryptic crash.
- Bit depth dropdown no longer falls back to "Select…" after applying a preset — default presets were saving `'8'` instead of `'8bit'`, which matched no picker key. Added store migrations to fix already-persisted values.
- `formatBytes` returns `"undefined"` for ≥1 TiB or negative values — extended unit array to PB, clamped index, guard negatives.
- `formatModelName` now handles embedded version tokens (`yolov9` → `YoloV9`); removed dead `.replace('V','V')` no-op.
- `extractBezierFromEase` no longer throws on empty ease arrays (returns `null`).
- Invalid trim ranges now fail during runtime validation when `--outpoint` is nonzero and not greater than `--inpoint`; the FFmpeg writer also exits cleanly if no frames reach the output queue.

## [2.7.10] - 2026-06-25
### Changed
- Bumped `nelux` `0.12.7` → `0.12.9`. 0.12.9 rebuilds the Windows wheels against FFmpeg master headers/libs and delay-loads the actual FFmpeg DLL names found at build time, so the nelux import/decode path no longer breaks when TAS downloads a newer shared FFmpeg build (e.g. `avcodec-63`) than the `avcodec-62` ABI the 0.12.7 wheel was built against; the runtime DLL diagnostic helper mirrors the same fallback and no longer reports a missing `avcodec-63.dll` when the compatible `avcodec-62.dll` set is present.

### Fixed
- `--decode_method nvdec` silently hung or crashed on sources NVDEC cannot handle (e.g. After Effects prerender AVIs with `codec=rawvideo, pix_fmt=bgr24`). nelux's NVDEC path only decodes H.264/HEVC/VP9/AV1 with YUV pix_fmts (NV12/P010/P016/YUV444); a raw/RGB source has no cuvid decoder, so `nelux.VideoReader(decode_accelerator="nvdec")` deadlocked/failed opaquely at construction and the CPU fallback in `BuildBuffer.__call__` never fired (constructor hang, not raise). `main.py` now ffprobes the input codec + pix_fmt and, when NVDEC is requested but the source is incompatible, downgrades to CPU decode with a clear `logging.warning`. Also normalized `args.decode_method` to lowercase at the source so the AE-bridge JSON's uppercase `"NVDEC"` matches the cache-invalidation guard in `src/io/ffmpegSettings.py` (`decodeMethod == "nvdec"` → `.lower()`); argparse `choices=["cpu","nvdec"]` enforced lowercase only on the CLI path, not on `--json`.
- Pipeline (`main.py` `process()`) and motion-blur (`src/motionBlur.py` `_processFrames`) deadlocked the `ThreadPoolExecutor` when `processFrame` raised mid-run (CUDA OOM, TRT engine mismatch, corrupt frame, etc.). The `finally` only closed the write buffer, leaving the read-thread blocked on `put()` to a full `decodeBuffer` — so `ThreadPoolExecutor.__exit__` waited forever and only a `KeyboardInterrupt` (`os._exit(130)`) could escape. Both loops now drain `readBuffer.decodeBuffer` in a `finally` block so the decoder's blocking `put()` returns and the reader thread reaches its sentinel-enqueue; motion-blur's loop body is also wrapped in `try/finally` (the success-only `writeBuffer.close()` at the end was insufficient).
- `--interpolate_method rife*-tensorrt` with `--half` corrupted every later tensor in the process. `RifeTensorRT.handleModel` called `torch.set_default_dtype(torch.float16)` and never restored it, so any subsequent `torch.zeros`/`torch.tensor`/`torch.full` (dedup SSIM init, restore model load, depth normalizer, writeBuffer frame conversion, …) silently became fp16. The line was redundant — every tensor creation in the file already passes an explicit `dtype=` — and is now removed.
- Removed the dead `--demo` code path (`main.py` `elif self.demo:` + the `self.demo` attribute). The branch imported `src.neluxDemo`, whose source was deleted (only a stale `cp313` `.pyc` remained, un-loadable on cp314); argparse had no `--demo` flag either, so the branch was unreachable and would have raised `ModuleNotFoundError` if ever hit.
- After a per-frame exception the success-only stats block in `VideoProcessor.start()` still reported a misleading `Total Execution Time / FPS` and `Output Size: 0.0 MB` because `process()` swallowed the exception in its own `try/except` and returned normally. `process()` now records the error on `self.processingError` and `start()` emits a single `Processing FAILED after <s>: <error>` line instead, so failed runs no longer look like they succeeded with a nonsense FPS.
- The FFmpeg-subprocess writer (`src/io/ffmpegSettings.py` `WriteBuffer.__call__`) waited only `3s` for `ffmpeg` to drain + finalize in its `finally` block and logged a noisy `Cleanup error: Command [...] timed out after 3 seconds` warning whenever the encoder was still flushing — especially common when the worker had died early and the pipe was starved. Bumped to `30s`, switched the timeout branch to `kill()` + a DEBUG log, and kept the genuine-error branch at WARNING.
- Concurrent TAS workers (e.g. AE bridge rendering multiple precomps) all opened the same `TAS-Log.log` with `filemode="w"` and interleaved at the line level, making each run's init block + `Processed/Encoded N frames` appear twice in the merged view. The log file is now per-run as `TAS-Log-<pid>.log` (path in the new `cs.LOG_PATH`), so concurrent runs no longer truncate + pollute each other.

## TAS Adobe Edition
### New
- Added an optional **Patreon** entitlement provider alongside GitHub Sponsors — links a Patreon account via an OAuth authorization-code flow (emulating device flow) with KV-backed polling sessions, patron-status caching, and a shared entitlement JWT.

### Changed
- Reworked the sponsor account UI: replaced the GitHub-specific account section with a provider-agnostic `AccountSection`, added `patreonSponsorService`, and made the sponsor store/hooks provider-aware.

## [2.7.9] - 2026-06-20
### Fixed
- Windows startup could fail with Nelux reporting `PyTorch must be imported before Nelux.` even when startup had already prepared the FFmpeg DLL search path. `src/io/ffmpegSettings.py` now imports `torch` before `nelux`, which matches Nelux's required import order and fixes the controlling startup failure; the FFmpeg DLL registration remains centralized in the bootstrap path.

## [2.7.8] - 2026-06-17
### Added
- Progress bar: leading spinner that turns into a `✔` on completion.
- Progress bar: live output filesize + bitrate column on the main pipeline bar; exact final size logged after encode (`Output Size: 29.5 MB`).
- **`--moblur_method`** — pick the motion-blur interpolation model + backend: `rife4.6` / `rife4.25` (CUDA) and their `-tensorrt` / `-directml` variants. Default `rife4.25` (rife4.6 is faster). Previously motion blur reused `--interpolate_method`.
- Added `smosr` 2x upscale model ([umzi2/SMoSR](https://github.com/umzi2/SMoSR), MIT) across CUDA, MPS, TensorRT, DirectML, and OpenVINO backends. Special thanks to UMZI.

### Changed
- Adopted **ruff** as the repo formatter + linter with a blocking CI lint gate (`ruff check` rules `E,F,I,UP,B` + `ruff format --check`); config in `pyproject.toml`, workflow `.github/workflows/lint.yaml`. Vendored trees are excluded to stay diff-clean vs upstream (#355).
- Bumped `barflow` `0.4.0` → `0.4.1`.
- Frame-processing loops (main pipeline, depth, object detection, segment) now terminate on the decoder's end-of-stream sentinel instead of polling queue state every frame; main loop uses a one-frame lookahead instead of queue peeking (#329). Thanks @KRRT7.
- Bumped `nelux` `0.12.4` → `0.12.7`.

### Fixed
- `--interpolate_method rife4.16-lite` crashed at model download (`ValueError: Model rife4.16-lite not found.`) — the method was wired through the CLI, model init and the arch table but had no `modelsMap` weight mapping. Mapping added (`rife416_lite.pth` / NCNN zips); new registry drift-guard tests (`tests/test_registryDrift.py`) now enforce the choices ↔ registry ↔ `modelsMap` sync so this class of gap fails tests instead of crashing users.
- `--encode_method lossless_nvenc` silently produced an FFmpeg command with no `-c:v` (the encoder defaulted per container) — `matchEncoder` only had an arm for the unreachable name `lossless_nvenc_h264`. Both names now resolve to the NVENC lossless flags and count as NVENC for hwupload.
- The final decoded frame of an `--inpoint`/`--outpoint` segment was silently dropped when the metadata frame count rounded down; sentinel-driven loops now emit every decoded frame (#329).
- DirectML/OpenVINO interpolation produced duplicate frames at factors above 2 — the ONNX `timestep` input was bound once and never refreshed per inserted frame. Now rebound each frame (#339).
- CI test job failed at collection: the registry drift-guard tests import `downloadModels`, which pulled in `barflow` (absent from the minimal CI env) at module level via `progressBarLogic`. The progress-bar import is now lazy inside `downloadAndLog`, so registry-only consumers (`modelsList`/`modelsMap`) import clean.

### Performance
- **AnimeSR (`MSRSWVSR`) CUDA inference path ~+7% faster** (RTX 3090, fp16, 360/720/1080p; ~+5–6% fp32) at **bit-identical output**. Removed a model↔input memory-layout mismatch (`channels_last` inputs fed to an NCHW model) that burned ~18% of CUDA time on per-conv layout conversions; dropped the no-op `res_scale == 1` multiply and the 9 `+ 0` multi-scale adds per forward. Peak reserved VRAM unchanged.
- **Motion blur is faster on CUDA** — only the frames inside the shutter window are interpolated. ~1.5× faster at the default shutter 180, up to ~3× at shutter 90. Output is unchanged; TensorRT/DirectML and shutter 360 are unaffected.
- **Upscale cold start ~30% faster** (launch → first frame on the CUDA/PyTorch path, e.g. `shufflecugan`; measured ~3.5s → ~2.5s at 720p on an RTX 3090). Output is byte-identical and all backends are unaffected. Five independent startup wins:
  - Made the `torchvision` imports in the vendored spandrel `LaMa`/`RestoreFormer` arches lazy (moved to call-site). The eager architecture registry imported `torchvision` (which drags in `torch._dynamo`, ~1s) at every startup even though the upscale model in use almost never touches those arches (~−740ms).
  - Removed two dead `from sympy import true` / `from sympy import false` imports (accidental auto-imports, never referenced) in the `SPANPlus`/`sudo_SPANPlus` arches that pulled in `sympy` during the registry build (~−290ms).
  - Reduced the pre-CUDA-graph warmup in `UniversalPytorch` from 5 to 3 iterations — `cudnn.benchmark` is off so the algorithm is fixed and 3 warmups suffice to prepare cudnn kernels/workspace before capture; the captured graph (and thus output) is identical (~−100ms).
  - Made `cv2` lazy in `ffmpegSettings` (only the OpenCV decode fallback and single-image PNG writer use it) (~−13ms), and read the Windows CPU name from `PROCESSOR_IDENTIFIER` instead of `platform.processor()`'s WMI shell-out in `checkSpecs` (~−25ms).
- **RIFE interpolation is faster on CUDA (fp16) and TensorRT.** CUDA: block inputs are downsampled per-component and concatenated at block resolution, image+feature warps sharing a flow are fused into one `grid_sample`, the dead feature warp after the final block is skipped, and the final blend/grid math use fused kernels — +9–15% FPS across the RIFE 4.x family (rife4.25 @1080p: 78→93 fps), output within fp16 noise of before. TensorRT: every `ConvTranspose2d` is repacked at load into a math-identical `Conv2d`+`PixelShuffle` pair, which TRT compiles to much faster tactics — +3% (rife4.6) to +12–18% (rife4.25; @1080p 90→110 fps). Quality verified against fp32 references (PSNR/SSIM/VMAF within measurement floor). Existing `.engine` caches rebuild automatically on first use.

## TAS Adobe Edition
### New
- The TAS chain now runs end-to-end inside **Premiere Pro** (increment 1): select a clip and its source media (with in/out points) is sent to the AI backend, then the processed result is imported onto a new video track above the source, time-aligned. It routes through the normal queue, so you get the same socket progress and queue UI as the After Effects path. After Effects-only UI (such as effect/preset scanning) is hidden when running in Premiere.
- Added an **MPS** entry to the Accelerator Guide — Metal Performance Shaders acceleration for Apple Silicon (M-series) GPUs, Mac only.
- Added two one-time supporter tiers: **Lifetime Supporter** ($99) and **Founder Pass** ($150).

### Changed
- Rewrote the toast notification system as a self-contained, After Effects 2020 / Chromium-74-safe implementation, replacing `@react-spectrum/toast` (now removed as a dependency) while keeping the same behavior. Toasts now appear bottom-center, stack, collapse into a deck of cards when there are several, pause on hover/focus, and animate their collapse/expand.
- Faster downloads: progress updates are now throttled so socket reads are no longer starved by per-chunk UI re-renders, which previously collapsed download throughput. The stall watchdog still tracks every chunk.
- Dependency install progress now refreshes per downloaded package instead of freezing on the last collected one, with more robust package-name parsing.
- The navigation bar now shrinks fluidly with panel width (icons, labels, and separators scale continuously) instead of stepping at fixed breakpoints, and drops separators entirely on very narrow panels.
- Chain pipeline rows no longer show an em-dash placeholder when a row has no status text.
- Shortened the "Edge Detection Enhancement" settings label across all locales.

### Fixed
- Fixed sponsor-status persistence recursing infinitely instead of sealing and caching the verified status.
- Sponsor avatars that fail to load now fall back to the GitHub logo instead of showing a broken image.
- AutoCut startup cleanup no longer errors in non-CEP runtimes (e.g. Vite browser dev) where Node `path`/`os`/`fs` methods are unavailable.


## [2.7.7] - 2026-06-01
## TAS Adobe Edition
### New
- Added AutoCAS to the NVIDIA GPU restore models (Gauge5) — a universal sharpening model. CUDA/sponsor-gated.

### Changed
- Replaced the standalone Sharpen pipeline checkbox with the AutoCAS restore model. The old `sharpen`/`sharpen_sens` options are removed; sharpening is now applied by selecting the AutoCAS restore model.

### Fixed
- Depth/pre-render no longer hangs at "initializing" forever when the project's default After Effects Output Module is set to an audio-only or otherwise invalid template (e.g. a leftover "Untitled 1"). Pre-renders now force a video output module — Lossless, falling back to ProRes/QuickTime then H.264 — so the file always contains a video stream, and fail with a clear error (instead of hanging) if no video template exists or no output file is produced.


## [2.7.6] - 2026-05-31
### Removed
- `--sharpen` / `--sharpen_sens` and any refereces to it.

### Fixed
- `--segment` (and any `prores`/transparent `.mov` output) crashed at encode start when the source had opus/vorbis audio — `WriteBuffer` stream-copied the audio into MOV, which FFmpeg rejects (`opus only supported in MP4`). `.mov` outputs now transcode audio to `aac` and subtitles to `mov_text` (MP4/M4V stay on copy — they accept opus natively).


## [2.7.5] - 2026-05-30

### Added
- ArtCNN luma upscale backend ([Artoriuz/ArtCNN](https://github.com/Artoriuz/ArtCNN)). ArtCNN doublers upscale the luma (Y) channel only, so the RGB frame is split to BT.709 Y'CbCr, the model runs on Y, chroma is bicubic-upscaled, and the planes are recombined. 8 models × 3 backends (`-tensorrt`, `-directml`, `-openvino`): `artcnn_c4f16{,_dn,_ds}`, `artcnn_c4f32{,_dn,_ds}`, `artcnn_r8f64`, `artcnn_r16f96` (DN = denoise/soften, DS = denoise/sharpen; only the C-series ship those variants). Fixed 2x, fp16/fp32.
- `deepdeband-f` restore method ([RaymondLZhou/deepDeband](https://github.com/RaymondLZhou/deepDeband), ICIP 2022).
- `autocas` restore method — AMD FidelityFX Contrast Adaptive Sharpening (PyTorch port, [NevermindNilas/AutoCAS](https://github.com/NevermindNilas/AutoCAS)) with per-frame automatic strength detection. Deterministic filter: no weights, nothing to download. Runs on CUDA or CPU as a frame-tensor restore stage (blur stats computed in fp32, so fp16 input is safe) and is chainable with other `--restore_method` models. Vendored as `src/extraArches/cas.py`, wrapped by `AutoCAS` in `src/unifiedRestore.py`.
- NVIDIA Maxine Video Super Resolution (RTX VSR) upscale backend via `nvidia-vfx` Python binding. New `--upscale_method` values: `maxine-bicubic`, `maxine-low`, `maxine-medium`, `maxine-high`, `maxine-ultra`, plus `maxine-highbitrate_{low,medium,high,ultra}` tuned for clean high-bitrate source. Runs on RTX Tensor cores; bundled TensorRT engines, no model download. Supports `--upscale_factor` 2/3/4.
- NVIDIA Maxine same-size enhancement restore backend via `nvidia-vfx`. New `--restore_method` values: `maxine-denoise_{low,medium,high,ultra}` and `maxine-deblur_{low,medium,high,ultra}`. Chainable with other restore methods.
- Added `nvidia-vfx==0.1.0.1` to `extra-requirements-windows.txt` and `extra-requirements-linux.txt` (lite variants unchanged). First install pulls ~490 MB (bundled model weights).
- Added `THIRD_PARTY_NOTICES.md` covering NVIDIA Video Effects SDK attribution requirement (AI Product-Specific Terms §1.7.1).
- `NeluxWriteBuffer` (the `*_nelux` encode methods) now muxes audio + subtitle streams from the source via `nelux.VideoEncoder.add_passthrough` instead of dropping them — `*_nelux` output was previously silent and subtitle-less because the path never spawned FFmpeg for muxing. Streams are registered before the first `encode_frame` (when the container header is written), gated/rebased to the `--ss`/`--to` (`inpoint`/`outpoint`) window to match the decode-side video trim, and gated on the `cs.AUDIO` global like the FFmpeg `WriteBuffer`. `allow_transcode=True` (nelux ≥ 0.12.0; feature-detected, copy-only fallback on older builds) re-encodes streams the container can't stream-copy (e.g. AAC→WebM → opus/webvtt) instead of skipping them, mirroring the `WriteBuffer` WebM path. Verified end-to-end: mkv(h264+aac+subrip) → mkv carries all three streams, and a 0.5–1.5s trim produces a 1.0s clip with audio/subs packet-gated to the window (`src/utils/ffmpegSettings.py` `NeluxWriteBuffer._setupPassthrough`). No FFmpeg subprocess is spawned on the `*_nelux` path.

- 
### Fixed
- `--segment` (and any `prores`/transparent `.mov` output) crashed at encode start when the source had opus/vorbis audio — `WriteBuffer` stream-copied the audio into MOV, which FFmpeg rejects (`opus only supported in MP4`). `.mov` outputs now transcode audio to `aac` and subtitles to `mov_text` (MP4/M4V stay on copy — they accept opus natively).
- `x264_nelux`/`x265_nelux`/`av1_nelux` were CLI-accepted but silently fell back to `h264_nvenc` (codec map only had the `nvenc_*_nelux` entries); now mapped to software codecs.
- `*_nelux` encoders ignored quality/preset and ran at nelux's ~4 Mbps default; each now maps to a `(codec, preset, cq)` mirroring `matchEncoder` (`av1_nelux` was also mis-routed to the slow `libaom-av1`).
- `NeluxWriteBuffer` crashed on non-CUDA hosts (unconditional `CudaStream.synchronize()`); stream usage now guarded.
- `--preview` + a `*_nelux` encoder crashed (`NeluxWriteBuffer` had no `previewPath`); preview is now skipped on the nelux path.
- Frame-processing errors deadlocked the run — the writer stop sentinel was only sent on the success path; moved into a `finally` block.
- Decode fallback duplicated frames: a mid-stream nelux decode failure re-decoded from frame 0 without draining the buffer; fallback now skipped once any frame is emitted.
- Truncated model downloads are now verified against `content-length` and retried (widened retry exceptions).
- `mse`/`mse-cuda` dedup was broken: `np.resize` mangled the buffer and the comparison was inverted; switched to `F.interpolate` + corrected direction, and added the missing `DedupMSECuda` class.
- FlowNetS dedup never detected duplicates: compared the signed optical-flow mean instead of motion magnitude.
- `--depth_method` no longer offers unwired values (e.g. `large_v3`, `giant_v3`, `og_video_small_v2-tensorrt`) that ran no model; added a raising default arm.
- Motion blur now fails loudly on an unsupported `interpolate_method` instead of producing empty output.
- Bare `--output filename.mp4` (no directory) crashed on `os.makedirs("")`; now resolves to an absolute path.
- DirectML/OpenVINO upscale + restore CPU fallback was dead code (caught only `UnicodeDecodeError`); now catches real provider errors.
- NCNN upscale: hardcoded `\` path separators broke non-Windows, and output was cast to `uint8` without clamping; fixed both.
- DistilDRBA TensorRT timestep race: the fill ran on the default stream while the graph replayed on another; fill now on the replay stream.
- Upscale `--compile_mode max` compiled with the default mode instead of max-autotune.
- Model optimizer no longer silently replaces ReLU with LeakyReLU, which corrupted activations.
- After Effects bridge: a newly-connected client got a stale/empty progress snapshot and the final payload could drop under rate-limiting; now emits the latest relayed payload, and dropped the no-op `cancel` capability.
- AutoClip (PySceneDetect) wrote a spurious final "cut" at EOF; the trailing boundary is now dropped.
- OpenVINO depth models failed to load due to a double-dash folder name.
- Depth (`OGDepthV3Cuda`) enqueued two output frames for near-empty-mask inputs, desyncing output length.
- Encoding crashed when the metadata file was absent (preview / single-image paths).
- Video metadata read `ColorSpace`/`PixelFormat` from swapped ffprobe fields.
- WebM input forced to `vp9` even when an AV1 encoder was selected; `validateEncoder` allowlist now includes the av1 encoders.
- AV1 encoders now set `SVT_LOG` correctly (was an exact-match check).
- RGBA PNG passthrough crashed on write (`COLOR_RGB2BGR` on a 4-channel image).
- Output naming overhaul: collisions get a `-1`/`-2` suffix instead of overwriting, Windows image sequences are no longer double-counted, directory inputs process in sorted order, and URL/sequence inputs pick the correct extension.
- YouTube downloader always showed the title as "Unknown" (typo'd info-dict key).

### Changed
- Upgraded to Python 3.14 (3.14.5 bundled). Dependencies are now cp314-specific — existing installs require a fresh dependency install to stay compatible.
- Encode-side RGB→YUV colorspace conversion (`WriteBuffer._buildFilterList`, BT.709 path) moved from `zscale` (zimg) to swscale `scale`. zimg runs a float pipeline costing ~3x the CPU of swscale's SIMD-integer path; isolated 1280×720 conversion benchmark (1200 frames, no codec) measured zscale `utime+stime` 6.77s vs swscale 2.14s. Because swscale only dithers on a bit-depth reduction, the chain routes through a 16-bit working format (`scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv444p16le,setparams=…`) so the downstream `-pix_fmt`/hwupload reduction still error-diffusion-dithers to the target depth — banding matches zscale exactly (luma flat-run identical). `setparams` now stamps the full color metadata (`colorspace`+`color_primaries`+`color_trc`+`range=tv`); the old `zscale=matrix=709` only tagged the matrix. Verified across all 14 `D:\testVideos` sources (bt601, rgb24, yuv420/422/444 8/10/12-bit, prores422/4444, rawvideo) × both encode input depths × output pixfmts (`yuv420p`, `yuv420p10le`, `yuv444p10le`, `p010le`) × 8 resolutions (4K…66×66): matrix-identical (luma mean Δ ≤0.42 of one 8-bit code), black level bit-exact, no banding regression. Peak white differs by +0.8 code (235→235.8, valid limited-range super-white, not clipping) — intrinsic to swscale's full→limited endpoint scaling, not flag-tunable. **BT.2020 stays on `zscale`** (`norm=bt2020` transfer handling and the `bt2020nc` matrix have no swscale equivalent). No behavior change for SD/601 sources (still default to BT.709 as before), grayscale, or transparent paths (colorspace filter skipped for the latter two).
- After a CUDA-profile dependency install, trim the ~470 MB of TensorRT runtime libs (`nvinfer_10`, `nvinfer_plugin_10`, `nvonnxparser_10`) bundled in the `nvidia-vfx` wheel that the Maxine VideoSuperRes path never loads — VSR runs entirely on NGX (`nvVFXVideoSuperRes → nvngxruntime → nvcuda`), verified at runtime (upscale/denoise/deblur all succeed without them). Trims the locally installed package only (no modified artifact redistributed). `src/utils/dependencyHandler.py` `pruneMaxineUnusedLibs()`.
- Bumped PyTorch and TorchVision to 2.12.0+cu132 and 0.27.0+cu132.
- Bumped Nelux to '0.12.2' (adds `VideoEncoder.add_passthrough` audio/subtitle muxing used by `NeluxWriteBuffer`; 0.12.2 fixes the cu132 wheel — 0.12.0 was built against torch 2.11/cu130 and 0.12.1 vendored a conflicting `c10_cuda`, both failing to import on a torch 2.12+cu132 box).
- Startup: GPU detection (`isCudaInit.py`) now runs `nvidia-smi` at most once per process. `detectNVidiaGPU` and `detectGPUArchitecture` were each spawning their own `nvidia-smi` (`-L` and `--query-gpu` respectively), and both are called from multiple `createParser` phases — up to 3–4 spawns per startup. Both now share one `@lru_cache`'d `--query-gpu=name,compute_cap` query (presence + arch from a single call); WMI/PCI fallbacks for driverless NVIDIA cards are unchanged. Dropped the now-dead `runNvsmicheck` and the brittle `-L` name-substring heuristic. When `compute_cap` is unavailable (driver too old to report it, or `nvidia-smi` absent while WMI/PCI still detect the card), `detectGPUArchitecture` now assumes a modern (Turing+) arch instead of returning `unknown` and downgrading the install to the lite/DirectML dependency profile (`src/utils/isCudaInit.py`).
- Startup: removed the `ffmpeg -version` subprocess (`logFfmpegVersion`) that ran on every launch purely to log the version string — informational only, never consumed. Deleted the function and both call sites (`argumentsChecker._handleDependencies`, `getFFMPEG`), plus the now-unused `subprocess` import in `getFFMPEG.py`.
- RIFE CUDA interpolation (`RifeCuda`) now captures the per-frame model forward in a CUDA graph and replays it, removing eager per-kernel launch overhead. The forward read fixed buffers (`I0`/`I1`/`_timestep_buffer`) already, so only the forward is captured; it is replayed on `normStream` in the exact spot the eager call ran, leaving every existing stream synchronize / cross-op race guard (decode-buffer + upscale interaction) untouched. RTX 3090, rife4.6 fp16, isolated min-frame latency: 720p 5.12→4.61 ms (**+11%**, ~196→216 fps), 1080p 9.70→9.16 ms (**+6%**), no VRAM change. Output is bit-identical to the eager path — verified by allclose (max Δ 0.0) on 2x/4x/timestep configs at 720p+1080p, and by hash-identical end-to-end output across {cpu, nvdec} decode × {interp-only, interp+upscale} (nvdec+upscale deterministic over 3 runs). Capture is gated to non-`staticStep` & non-`dynamicScale`, with an init self-check (graph replay vs eager forward) and try/except → eager fallback for any arch where capture can't be verified (`src/unifiedInterpolate.py` `RifeCuda._setupCudaGraph`). Other backends/classes unchanged.
- Stabilizer ORB transform estimation (`VideoStabilize._estimateTransformORB`, the SuperPoint fallback path) now extracts matched keypoint coordinates via a single `cv2.KeyPoint_convert` + numpy fancy-index instead of per-match Python `.pt` list-comprehensions, and does the Lowe ratio test in one pass that also collects match distances (dropping two extra iterations over the match list). Bit-identical to the previous extraction — same points in the same order feed `cv2.estimateAffinePartial2D`, so RANSAC behavior is unchanged (verified: 0 mismatches over 39 real frame pairs). ~2.3x faster on the extraction itself; cProfile self-time of `_estimateTransformORB` dropped ~17% (0.215→0.178 s over 238 calls on a 720p 10s clip). This is the only host-side hot Python loop the per-frame profile surfaced — the standard upscale/interpolate/restore/motion-blur paths are GPU-bound (negligible Python self-time), so this only affects ORB-fallback stabilization (`src/stabilize/stabilize.py`).
- DepthAnythingV2 CUDA fp16 fast path: SDPA (gated to fp16/bf16, fp32 stays bit-exact), positional-embedding cache keyed by `(W, H, dtype)`, `nn.quantized.FloatFunctional` skip-add wrappers dropped from DPT fusion blocks (also applied to Depth Anything V3 DPT), LayerScale gamma folded into preceding `attn.proj` / `mlp.fc2|w3` weights at `load_state_dict` time, and decoder-side resize via `BuildBuffer(resize=True, …)` to skip post-decode `F.interpolate`. End-to-end TAS workflow on RTX 3090, 720p input (1268×714, `--depth_quality high`), 5s clip:

  | model | base FPS | fast FPS | gain | base VRAM (alloc/resv) | fast VRAM (alloc/resv) | PSNR Y | SSIM Y |
  |---|---|---|---|---|---|---|---|
  | small_v2 | 24.78 | 45.31 | +83% | 768/1122 MB | 559/782 MB | 45.97 dB | 0.9961 |
  | base_v2 | 11.92 | 22.25 | +87% | 1438/2032 MB | 904/1280 MB | 42.99 dB | 0.9962 |
  | large_v2 | 4.63 | 8.36 | +81% | 2237/3002 MB | 1749/2428 MB | 44.49 dB | 0.9966 |

- RIFE CUDA fp16 fast path: load-time weight repack (ResConv beta fold, ConvTranspose→Conv2d+PixelShuffle) + cached backwarp grid. Same `.pth`, fp32 untouched, output within fp16 noise (PSNR ≥74 dB). RTX 3090 1080p fp16 FPS:

  | method | base FPS | fast FPS | gain | base VRAM | fast VRAM |
  |---|---|---|---|---|---|
  | rife4.6 | 99.0 | 109.2 | +10.3% | 2.7 GB | 2.4 GB |
  | rife4.15-lite | 88.8 | 98.8 | +11.3% | 2.9 GB | 2.6 GB |
  | rife4.17 | 66.0 | 71.0 | +7.6% | 2.8 GB | 2.7 GB |
  | rife4.18 | 66.1 | 71.2 | +7.7% | 3.2 GB | 2.7 GB |
  | rife4.20 | 64.5 | 69.1 | +7.2% | 3.2 GB | 2.8 GB |
  | rife4.21 | 52.7 | 54.9 | +4.3% | 3.1 GB | 3.0 GB |
  | rife4.22 | 53.5 | 55.5 | +3.8% | 3.1 GB | 3.0 GB |
  | rife4.22-lite | 79.5 | 89.3 | +12.4% | 2.9 GB | 2.7 GB |
  | rife4.25 | 73.5 | 78.1 | +6.2% | 3.0 GB | 2.8 GB |
  | rife4.25-lite | 75.4 | 80.7 | +7.1% | 3.1 GB | 2.9 GB |
  | rife4.25-heavy | 53.3 | 56.1 | +5.3% | 3.4 GB | 3.1 GB |

- GMFSS CUDA fast path: dead IFBlocks dropped, scale-0 transformer hoisted bidir, splat/interp/backwarp batched, nn.PReLU swap, ResConv beta-fold, fp16 warp, + prebuilt native CUDA softsplat kernel (`_gmfss_softsplat_ext.pyd`, Turing→Blackwell+PTX). No end-user nvcc/toolkit required. RTX 3090, 1080p fp16, 2x, 3s clip:

  | metric | baseline | torch-only opts | + native kernel |
  |---|---|---|---|
  | FPS | 4.02 | 4.92 (+22.4%) | 5.12 (+27.3%) |
  | Peak VRAM | 4058 MB | 4402 MB | 4169 MB |

  Quality vs baseline: PSNR 51.4 dB avg / SSIM 0.9989 / VMAF 96.3 (fp16-noise tier).

### Removed
- Dead-code sweep (no behavior change). Removed orphaned depth-arch modules `dpt_v1.py`, `blocks_v1.py`, `dinov2_v3.py`, and the stray top-level `dpt_temporal.py` / `blocks.py` / `transform.py` (superseded by the `dpt_v2` / `og_dpt_v2` / `video_depth_anything` lineages). Removed unreferenced symbols: the never-dispatched `DedupMSSSIMCuda` class (and the now-orphaned `MS_SSIM` / `ms_ssim` in `dedup/ssim.py`) and `ATRCuda`; all unused `frameReset()` methods across upscale/restore; `objectDetection.drawBoxesTorch`; `stabilize._processStreamingFrames`; `downloadModels.downloadTensorRTRTX`; the unused `logAndPrint` helpers (`setVerbose`/`setQuiet`/`logDebug`/`blue`/`magenta`/`lightBlue`/`rainbow`/`bold`); dead `CudaChecker` members (`mpsAvailable`/`acceleratorAvailable`/`disableCudaOptimizations`/`deviceName`/`allDeviceNames`); and several write-only attrs / unused params (`useOpenCV`, `batchSize`, `channels_first`, `optimizationLevel`, `args.isNvidiaGPU`, `args.requirements_file`).

## TAS Adobe Edition
### New
- Added NVIDIA Maxine RTX VSR upscaling and Maxine denoise/deblur restore models.
- Added deepDeband F (CUDA) and Intel OpenVINO restore models.
- Added a "Show Legacy Models" setting to hide deprecated upscale models.
- Remove Dead Frames now runs natively in After Effects, with resolution divisor and preserve-length options.
- Added a feedback form link in the Socials section.
- Added x264 (NeLux) and x265 (NeLux) CPU encoders — up to 50% faster with lower CPU usage (Beta).
- Added a Lock View toggle to the Graph Editor — freezes pan/zoom (and reset-on-double-click) so the mouse wheel scrolls the panel instead.

### Changed
- Faster UI: memoized hot components to cut unnecessary re-renders.
- Removed the compact model family from the upscale picker; defaults now use `shufflecugan-directml`.
- Background Removal now defaults to the free Anime DirectML model (works on all GPUs) instead of the sponsor-only Anime Cuda.
- Graph Editor canvas now caps its height and stretches horizontally (After Effects-style), with sharper high-DPI rendering.
- Merged the Graph Editor's separate shortcuts and settings popovers into one dialog with Reference and Customize tabs.

### Fixed
- Discord Rich Presence toggle now actually disables Rich Presence.
- Fixed a WebSocket and heartbeat leak when a run was cancelled or errored without emitting a completion event.
- Fixed a log-watcher retry timer leak on process exit, error, or cancellation.
- Fixed ExtendScript calls breaking when an argument was `undefined`.
- Fixed Precompose placing layers at the wrong start time.
- Auto-clip (Premiere) now validates the clip and results file before running and always releases the file handle.
- Hardened the sponsor-login worker against open-redirect and token leakage (opaque-origin allowlist matching, https/cep-only redirects) and stopped internal error details from leaking in responses.

## [2.7.0] - 2026-05-10
### Fixed
- DirectML black frames due to IOBinding issues.

## [2.7.0] - 2026-05-10
### Added
- New encoding options, x264_nelux, x265_nelux and av1_nelux, which utilize the Nelux encoder for improved performance.
- `--moblur_shutter_angle` (degrees, default `180`): cinema-style shutter window. `360` = full smear, `90` = crisp, `0` = passthrough.
- `--moblur_no_linear_blend`: opt-out of gamma-correct blending (default on).
- `--moblur_mask`: transparent PNG to protect regions from motion blur. Paint opaque dark pixels on a transparent background — those areas stay sharp (HUD/text), transparent areas blur.
- `--autoclip_method` with selectable scene-cut backends: `pyscenedetect`, `maxxvit-directml`, `maxxvit-tensorrt`, and `transnetv2`.
- Added MaxxVit- and TransNetV2-based autoclip support.
- Added A VERY! experimental WIP MacOS build.

### Changed
- Up to `300ms` start up times improvements by optimizing imports and lazy loading certain modules.
- Up to `2x` performance improvements to `--resize`.
- Major motion blur quality overhaul: gamma-correct (linear-light) blending plus symmetric shutter window centered on each source frame. Trails no longer lag motion, highlights retain punch, shadows stay clean.
- `--moblur_factor` default `2` → `8`, `--moblur_strength` default `equal` → `gaussian_sym`.
- Reduced peak VRAM during model initialization by ~`22%` (measured on `--depth_method base_v2`: 1004 MiB → 788 MiB peak reserved).
- Reduced FFmpeg decode/encode buffer queues from `64` to `32` frames, halving frame queue VRAM (e.g. ~3 GB saved at 4K fp16) without throughput impact.
- Autoclip dispatch now routes to backend-specific implementations, and `--autoclip_sens` is remapped per backend so higher values consistently mean more aggressive cut detection.
- Nelux `VideoReader` reuse now accounts for resize targets and can perform decoder-side resize directly, avoiding redundant post-decode interpolation when `--resize` is enabled.
- Bumped nelux to `0.10.1` for improved stability and performance. Decoding is now significantly faster and more efficient, especially for H264, HEVC and AV1 content of any pixel format. Nearly all pixel formats are now supported with no fallback to OpenCV.
- Bumped openvino and onnxruntime-openvino to the latest release.
- Removed the Linux `torchcodec` runtime dependency from requirements and project metadata.
- Collapsed duplicate `*-directml` / `*-openvino` match arms in depth dispatch (`small_v2`, `og_*_v2`, `*_v3`).
- Extracted PNG passthrough flow in `main.py` into `_runPngPassthrough` to keep the batch loop readable.
- Added `*.log` to `.gitignore` to stop tracking transient log artifacts.
- Replaced ~25 hand-rolled `if-not-exists → downloadModels else cached-path` blocks across upscale/interpolate/restore/depth/segment/dedup with a single `resolveWeightPath` helper in `src/utils/downloadModels.py`.
- Removed unused `pyproject.toml` (build/runtime are driven by `requirements*.txt`; the file was stale metadata only). Updated `build.py`, `install.ps1`, `BUILD.md`, and `Build-macOS.yaml` accordingly.
- Narrowed macOS CI trigger paths to `build.py`, `requirements.txt`, and `extra-requirements-macos.txt` so unrelated source/doc commits no longer rebuild.
- Added `build_macos` job to `Build-All-Platforms.yaml` so release builds now publish a `TAS-<ver>-macOS-arm64.zip` asset alongside the Windows zip.

### Fixed
- Fixed openvino related issues.
- Fixed broken DirectML/OpenVINO provider check in `UnifiedRestoreDirectML` (operator-precedence bug made the CPU fallback unreachable).
- Fixed wrong attribute reference in `DepthCuda` compile error path (`self.interpolateMethod` → `self.depth_method`).
- Fixed double-dash folder names for ONNX restore weight resolution (`-directml`/`-openvino` suffixes are now replaced as a unit).
- Fixed mirrored `replace("directml"/"openvino", "-onnx")` folder-name bug in `UniversalDirectML` and `AnimeSRDirectML` (same root cause as the restore fix).

## TAS Adobe Edition
### Added
- A FULLY redesigned auto clip workflow with its own separate interface, clip selection support, fast and accurate selection, multiple new backends and more. ( UX inspiration from https://github.com/crptk/AMVerge )
- Added mask option, linear blend, and shutter angle options to the motion blur settings.
- Added new autoclip methods, MaxxVit- and TransNetV2-based.
- Read button to the graph editor to read the current graph from the backend and display it in the UI.
- Import / Export buttons to the graph editor to export and share your presets.
- Added a flip x/y axis toggle to the graph editor.
- A major redesign to the chain tab with checkboxes, clear indication of what models are being used and a more intuitive layout.
- Added a real time import to AE option to the graph editor, this allows you to apply the changes in real time.
  
### Changed
- The keyframe graph editor is now bound by the X axis of the canvas.
- Updated dependencies.
- Reworked the Youtube download workflow, it will now try its best to comply with after effects innate codec and container limitations.
- Shaved up to 3s on the yt-dlp fetch and download process by optimizing the logic and parallelizing certain operations.
- Major performance improvements in start-up times.
- Further security improvements to the GitHub Sponsors workflow.

### Fixed
- Fixed an issue where the graph editor would not properly read the current graph from the backend.
- Fixed an issue with av1 / vp9 and webm containers when downloading from youtube.
- Changed some delinking related parameters in the Github Sponsors workflow to be less strict and more effective. Less false positives and less deconnection issues. 
 
## [2.6.0] - 2026-04-13
### Added
- Added `adore` 2x upscaler (shufflecugan architecture). Available across CUDA and TensorRT.
- Added motion blur support (`--moblur`). Supports `equal`, `gaussian_sym`, `pyramid`, `ascending`, and `descending` weighting using `--moblur_strength`, plus `--moblur_factor`. Works with all RIFE/GMFSS interpolation backends and includes automatic scene cut detection.
- Added `--depth_norm` to normalize depth maps over time for better quality with minimal performance impact.
- Terminal window/tab title now updates during processing to `★ Processing - <video name> [current/total]`, making batch progress visible at a glance from the taskbar.
- Added new upscaling methods: `adore` (shufflecugan architecture) with TensorRT and Cuda support.
- Major reworks for the dependency handler including --download_requirements which now allows for 2 variables `--windows-cuda` and `--windows-lite` to further specify the dependencies to be downloaded for windows users.
- Added a brand new progress bar with better performance and better visuals. Named `barflow`, this is a custom progress bar library that I've developed for TAS and will be open sourced in the future.

### Changed
- Reverted dependency handling back to pip-based requirements files for runtime installs, the portable build, and the Windows installer.
- Upgraded Pytorch to 2.11.0.
- Upgraded Torchvision to 0.26.0.
- Upgraded TensorRT to 10.16.0
- Upgraded Nelux to 0.8.10
- Improved `--help` / `--version` startup performance by ~57% (326ms → 141ms avg)
- Replaced the `colored` library with native ANSI escape codes for further improvements to startup time and reduced dependencies.
- Completely removed rich from the codebase.

### Fixed
- Color formatting related issues when encoding BT709 + RGB24.
- Fixed Error [22] - FFMPEG errors caused by certain memory allocations + mem cuda pinned issues.
- Fixed Error [32] - FFMPEG can't write to output file due to permission issues or non existing folder. ( in some cases, TAS-AdobeEdition fails to create the folder/s )
- Fixed `og_small_base_v2`.
  
## Adobe Edition
### Added
- Graph Editor is now out of BETA.
- Added Adore Cuda and Adore TensorRT
- Added a Legacy models toggle in the settings tab to show or hide superseeded models.
- Added depth normalization toggle to the depth settings.
- Remove dead frames now properly supports multiple layers at once.

### Changed
- Rewrote the Graph Editor from the ground up for better performance, stability and usability.
- Rewrote the About Page to be more informative and better looking.
- Rewrote the Queue Tab to be more informative and better looking.
- Minor UI adjustments.
- Up to 37% performance improvements for Remove Dead Frames.
- Updated all the dependencies.
- Further improvements to the GitHub Sponsors workflow and Security.

### Fixed
- Fixed many translations and broken UI elements.
- Fixed many contextual helpers looking weird due to the the new language system implemented.

## [2.5.10] - 2026-03-16
### Added
- Added a Windows PowerShell one-line installer (`iwr -useb https://raw.githubusercontent.com/NevermindNilas/TheAnimeScripter/main/install.ps1 | iex`) that installs in the current directory and supports optional PATH registration.
- Added `tas` and `theanimescripter` command aliases for CLI usage (`--help`, etc.) after install.
- Reworked dependency management and the build process, expect some issues here and there but it should be mostly smooth sailing.
- Added "Depth Anything V3" support with commands `og_large_v3` ( WIP ).

### Changed
- PNG output support will now be handled by pillow / opencv instead of ffmpeg, this should result in better performance and better output quality at a significantly faster speed. ( from 7s execution time down to 3s )

### Fixed
- Fixed an issue where the dependency handler would not properly handle certain edge cases when installing or uninstalling dependencies.
- Fixed an issue where the dependency handler would faulty trigger a dependency installation / check due to a false positive.
- (NELUX FIX) Fixed an issue where NVDEC caused Cuda Race conditions when the consumer was also utilizing the frames/tensors for pre-processing the frames.
- (NELUX FIX) Fixed an issue where OpenCV returned frames caused stuttering / freezing of the first X initial frames due to frame queing issues.

### Known issues
- NVENC Nelux encoders can have some stuttering / freezing in the first few frames due to the way the frames are queued and processed, this is a known issue that I'm actively working on fixing. ( Cuda Race condition, nelux needs to bep patched )
- The new dependency handler is still a work in progress and may have some edge cases that are not properly handled, if you encounter any issues with dependencies, please report them to me so I can fix them.

## Adobe Edition
### Added
- New depth Method, "OG DA3 V3 Large Cuda".

### Changed
- Minor UI adjustments.
- Minor Performance improvements.
- Tweaks to a few translations.

### Fixed
- Fixed an issue where the UI would faulty display a red box around Video Stabilization.
- Fixed a regression in the queue handler, mostly related to the progress bar not updating properly.

## [2.5.9] - 2026-02-22
### Added
- A progress status for tensorrt engine generation.
- PNG input support.
- Added Video Stabilization support with command "--stabilize". ( WIP )
- Added CUDA upscaling support for `figsr`. ( special thanks to UMZI for this model )

### Changed
- Replaced Torchcodec with OpenCV as a fallback decoder.
- Replaced Video Depth Anything OpenCV decoding with Celux for better performance and better output quality.
- Replaced cv2 based decoding to nelux based for yolov9-directml and yolov9-tensorrt for better performance for the same output quality.
- Bumped python to 3.13.12
- Improved the AEComms bridge between TAS and Adobe for better performance.

### Fixed
- Fixed unintended removal of spandrel.

## TAS Adobe Edition
### Added
- A new PREVIEW system to the chain tab which allows you to preview the output(s) before committing to the render.

### Changed
- Minor UI tweaks.

### Fixed
- About 250% less API calls to GitHub and Cloudflare to check for updates and sponsors status.
- Fixed frequent expiration of GitHub Sponsors status due to rate limits and other issues.  

## [2.5.8] - 2026-02-15
### New
- Added support for Video Depth Anything V2 models with commands: "og_video_small_v2", "og_video_base_v2" and "og_video_large_v2".
- Added support for Dehalo with commands: "dehalo", "dehalo-tensorrt", "dehalo-directml", "dehalo-openvino".
- Added DirectML and OpenVino support for ShuffleCugan, Fallin Soft and Fallin Strong.
- Added experimental nvenc support for encoding through nelux ( rebranded celux ), use commands: "nvenc_h264_nelux", "nvenc_hevc_nelux" & "nvenc_av1_nelux" ( WIP ).
- Added support for "yolov9_large-tensorrt" ( WIP )
- A new argument interpreter with better error handling and more flexibility.
- The github workflow will now auto add the changelogs to the release description when a new release is created.
  
### Changed
- Reverted to Pytorch 2.9.1 to fix some stability issues with certain systems.
- Updated some dependencies to their latest versions.
- Turned TAS into a fully portable aplication, all dependnecies are now included in the TAS-Portable folder and there is no more need for scaterring dependencies around the system.

### Fixed
- Certain users experienced instant crashes with Pytorch 2.10, for now I have reverted back to torch 2.9.1.

### Removed
- N/A

### Adobe Edition
#### New
- Added DeHalo -CUDA / -TensorRT / -DirectML / -OpenVino support
- Added ShuffleCugan, Fallin Soft and Fallin Strong -DirectML / -OpenVino support
- Added preview image support into the queue tab.
- Added Dark and Light mode support to the UI ( the light mode is take it or leave it :D )
- Added support for turning off discord rich presence.
- Added support for dynamic download and storage of the backend, the user can now choose where to store the backend.
- In the quick presets section, added a export and import feature to share presets with other users and or store them for later use.

#### Changed
- Major performance improvements.
- Few UI Tweaks to clean up space and improve organization.
- Minute changes to graph editor
- Increased security and privacy of the GitHub Sponsors workflow.
- Further security improvements.
- Updated some dependencies to their latest versions.
- UI improvements for the about page.
- Added a "New" tag for newly added models.
- Added a "Experimetntal" tag for experimental models / features.
- 
#### Fixed
- Fixed --sharpen not being parsed to the backend.
- Fixed --slow-mo not being parsed to the backend.
- Fixed a memory leak in the graph editor.
  
#### Removed
- Bloat from the graph editor UI.
- NCNN mentions from the UI.

## [2.5.7] - 2026-01-22
### New
- Added Torchcodec as a fallback decoder in case if celux fails to decode certain formats.
- Added DirectML and OpenVino support for Rife.
### Changed
- More progress updates will be present in the Adobe Edition progress bar.
- Surpressed many of the unnecessary logs.
- Bumped Pytorch to 2.10.0
- Bumped Celux to 0.8.5

### Fixed
- https://github.com/NevermindNilas/TheAnimeScripter/issues/238

### Removed
- Deprecated NCNN backends. 

### Adobe Edition
#### New
- Added AUTO updating capabilities to TAS-AdobeEdition ( so long as it has access to the preset folder )
- Added DirectML and OpenVino support for Rife.
- Added tab switch transitions.

#### Changed
- More info in the queue tab.
- Added more async and promise operations over the place for better performance and less UI blocking.

#### Fixed
- Potential fix for: https://github.com/NevermindNilas/TheAnimeScripter/issues/242

#### Removed
- All NCNN related code.
- Small or Large UI modes, the default will remain as is.
- Lossless or Quicktime Presets, Quicktime is now the default.

## [2.5.6] - 2026-01-17
### New
- Increased ENCODING performance by up to 10% for most CPU workloads and by UP to 20% for NVIDIA NVENC workloads.

SUC-TensorRT, 1080p, 3090

Before optimizations: ~70 FPS with NVENC
After memoryview: ~86 FPS
After hwupload_cuda: ~97 FPS

- Added new upscale methods, "gauss", "gauss-tensorrt", "gauss-directml", "gauss-openvino".
- Added new flag, `--decode_method` with options "CPU" and "NVDEC", this is a work in progress for the Celux project that I've taken over recently but performances are already better than the default. This is a WIP and quality and performance are still being actively improved.

Rife 4.6-TensorRT, 1080p, 3090, default encoding ( x264 vfast ).

Benchmark / Baseline: ~320 FPS
CPU Decode: ~250 FPS
NVENC Decode: ~315 FPS

### Changed
- Slight tweaks to the decoding and encoding pipelines.
- Improved the performance of .peek() and .get() methods
- Major improvements to AnimeSR performance.
- Bumped onnx to 1.20.1
- Up to 33% improvements to start-up times.
- Fixed PNG output related code.
- Better logging.
- Switched from using fully enclosed ffmpeg to shared ffmpeg binaries to work with Celux.

### Fixed
- Fixed a bug where if the default `_method` was used, it will not be picked up as an active option.

### Removed
- SceneChange detection from the main codebase, it will be reintroduced in the future with better performance and accuracy, stay tuned!

---

### Adobe Edition
#### New
- Added a very requested feature, PNG output support!
- Added tweaking opportunities to remove dead frames.
- A brand new navbar with smaller footprint, better animations and a more modern look, compatible with AE2020+. 
- Account information will now use your own GITHUB profile picture.
- Added NVDec based decoding methods ( WIP )

#### Changed
- Minor performance improvements to remove dead frames.
- Resize Input will now happen inside After Effects to lower overall processing time.
- Removed the control points from the keyframe graph editor.
- Removed Deduplication from the chain tab and replaced it with remove dead frames.

#### Fixed
- Potential fix for Global shortcuts not working on some systems.
- Fixed duplicated upscale text description in quick presets.
- Fixed an issue where the keyframes graph editor would go blank when resizing
- Partially fixed an issue where the UI froze when switchign to the keyframe graph editor.

## [2.5.5] - 2026-01-07
### New
- Added `socket-io` comms to the backend facilitating the development of the Adobe Edition.
- Added a `CPU` fallback for all DirectML / OpenVino accelerators.
- Added new Segment ( Background removal ) options, namely `anime-directml` and `anime-openvino`.

### Changed
- Changed the way the commands get parsed to the backend, you are no longer required to use `--upscale --upscale_method`, you can simply use `--upscale_method` ( the same applies to other options, like --interpolate or --depth )

### Fixed
- N/A

### Removed
- Flask

---

## Adobe Edition
### New
- Added a full `socket-io` comms system to the frontend, lowering overhead and improving performance.
- Added a new Sponsor Tab to the UI.
- Added support for DirectML and OpenVino accelerated Segmentation.
- Merged logs tab into a queue element log for each time you run a job.

### Changed
- Json related logic has been heavily reworked.
- Improved security and UX of the GitHub Sponsors workflow.

### Fixed
- Fixed an issue where Github Sponsors would run into a bad credentials error.
- Fixed an issue where YT-DLP was not working.
- Fixed some spacing related issues.
- Fixed some inconsistencies in the Picker sections where certain elements would not be displayed.

### Removed
- The Logs tab.

## [2.5.4] - 2026-01-03
### New
- Added `OpenVino` accelerator to the backend through onnxruntime ( highly experimental and only tested so far on my iGPU )
- Added `distildrba-tensorrt` interpolate method.

### Changed
- Bumped numpy 2.2.4 -> 2.2.6 and moved it to one of the default requirements.
- Slight rework to the directml backend

### Fixed
- N/A

---

## Adobe Edition
### New
- Added Discord Rich Presence support.
- Added new OpenVino Accelerators ( free for now )
- Added DistillDrba-Tensorrt interpolation method
- A brand new Youtube DL handler using ytdlp-nodejs, with both Video and Audio download support.

### Changed
- Better OAuth workflow handling incl. improved security.

### Fixed
- Fixed issues related to Youtube Downloading.


## [2.5.3] - 2025-12-26
### New
- Added new restore methods, based on anime4k, `linethinner-lite-cuda`, `linethinner-medium-cuda`, `linethinner-heavy-cuda`

### Changed
- Added a check for when `--upscale_factor` is below 2 or a non-integer.

---

## Adobe Edition
### New
- Added a copy button for the one time code that's being generated when loging into GitHub.
- Added restore methods, LineThinner Lite Cuda, LineThinner Medium Cuda, LineThinner Heavy Cuda.
- Added " Github Sponsors Only " option to all CUDA and TensorRT relevant features.

### Changed
- Changed the deactivation of the GitHub sponsors priviledges from 30min to 7days.
- Removed BuyMeACoffee from the Sponsors Tab.
- Removed Offline Mode from the UI

### Fixed
- Fixed an issue where some sponsors couldn't receive their rewards.
- Fixed an issue where --upscale_factor 1 was being parsed to the backend.
- Fixed an issue where --half was set to False despite fp16 being set to true.
- Fixed Saryn related names.

## [2.5.2] - 2025-12-24
### Hotfix
- Fixed wrong wheel for VMAF-Torch

## Adobe Edition
- Improved security for GitHub sponsors communication.


## [2.5.1] - 2025-12-24
### Hotfix
- Fixed broken links

## [2.5.0] - 2025-12-24
### IMPORTANT NOTES:
- N/A

### Added
- Added `distilldrba`, `distilldrba-lite`, `distilldrba-lite-tensorrt` interpolate methods.
- Added `vmaf` & `vmaf-cuda` dedup method.
- Added my very first AI Model, Saryn with options: `saryn`, `saryn-tensorrt` & `saryn-directml`.

### Changed
- Improved output handling and name generation.
- Removed `pytorch-lightning` from the requirements.
- All Depth OG models will now use CELUX for decoding for better performance and better output quality.

### Fixed
- Fixed an issue where if the progress bar contained non UTF-8 compliant text, it would result in a crash.
- Better text handling in the terminal.

## Adobe Edition
### New
- Added a brand new Queue System to Adobe Edition
- Added all the new upscale, interpolate and dedup methods.
- Added Alpha support for Layer inpoint and outpoint effectively bypassing the pre-render step.

### Changed
- Rewrote a significat amount of the code.
- Improved many elements of the UI.
  
### Fixed
- Errors with access violation 3221225477.

## [2.4.9] - 2025-12-01
### IMPORTANT NOTES:
- I have taken over the development of [Celux](https://github.com/NevermindNilas/CeLux), the primary decoding backend for TAS. In order to improve output quality I have had to migrate many of its core color conversion logic from `swscale` to `libyuv` as well as rework some of its internal pipinelines. This change was non-trivial and performance may be affected in most of the workflows for a significantly better I/O handling and quality. All notable changes will be in that repo. Production of TAS will see a major slowdown now that I have single handedly maintain 3 repositories ( TAS, TAS-AdobeEdition & Celux ) so please be patient with updates.

## The Anime Scripter
### Added
- Added `av1` decoding support through `libdav1d` in Celux.

### Changed
- A significant rework of the Dependency Handler.

### Fixed
- An issue where certain dependencies would not be properly uninstalled when using `--download_requirements`.

## Adobe Edition
### Added
- Graph Editor is now out of Early Alpha, with improved stability, responsiveness and usability.
- Graph Editor now has a `.flow` import option in order to easily migrate from other plugins.
- Added a way in which the nodes can stick to the walls while holding shift in the Graph Editor.
- Double clicking the nodes will reset them to their default position in the Graph Editor.
- Added a final fallback to the pre-render file seeking logic which will now look for the latest file in the TAS Pre-Renders folder if all else fails.
- Added a settings tab and moved all advanced settings there.
- Added a "What's New" disclosure to the About Tab which will show the latest changes in each version.

### Changed
- Minor UI tweaks.
- Merged the extra tab with the layers tab.

## [2.4.8] - 2025-11-15
### Added
- Added an auto-refresh feature to the `--preview` functionality.
- 3 new restore models: `hurrdeblur`, `hurrdeblur-tensorrt`, `hurrdeblur-directml` by Zarxrax
- Added restore methods chaining through `--restore_method model model` option.
- Added command `--json` to pass arguments through a .json file, this is primarily used for the Adobe Edition but can be used in CLI as well.

### Changed
- Added back flask and related dependencies.

### Fixed
- Issues with rife onnx exports.
- Possible cuda race condition when peeking future frames.
- Fixed an issue with fastlinedarken + fastlinedarken-tensorrt.

## Adobe Edition
### Added
- AnimeSR support.
- Hurrdeblur support.
- Default presets to the chain tab to ease the first time user experience.
- Added support for restore methods chaining through the UI.

### Changed
- Improved the Graph Tab performance and responsiveness.
- Removed some hardly used features from the graph editor.
- Added a search for preset option in the Graph Editor.
- Migrated to a --json approach for passing arguments from the Adobe Edition to TAS backend for better path safety.


## [2.4.7] - 2025-11-02
### Added
- Added `AnimeSR` support with `animesr` & `animesr-tensorrt` commands ( DirectML support may be added in the future altho' performance may be subpar )

### Changed
- UX improvements to the YTDLP logic.
- Completely rewrote the `--preview` functionality for better performance and lower overhead, it should now use up at most `0.5-1%` of the CPU rather than up to 5-10% previously.

### Fixed
- Issues with `--preview` not showing the actual output frame but instead the input frame.
- Temporarily reverted the `tensorrt` version in order to fix cuda-runtime issues.

### Removed
- Flask and related dependencies.



## Adobe Edition
### Added
- Added a `Reinstall TAS` button to the extra TAB.

### Changed
- Updated dependencies to their latest versions.

### Fixed
- N/A

## [2.4.6] - 2025-10-24
### Added
- As per community request, I've `Depth Anything V2 Giant` model support with commands:
- `giant_v2` & `og_giant_v2`, TRT and DirectML support may be added in the future depending on demand.

### Changed
- Removed some unused warnings.

### Fixed
- Missing dependency `onnxscript`.

## Adobe Edition
### Added
- `Depth Anything V2 Giant` model support in the UI.
- Quick presets options to the `Chain Tab` which also includes self made presets. 

### Changed
- Defaulted / Reverted back to `PreRender_XXXXXX` format for pre-renders.
- Some small UI tweaks.

## [2.4.5] - 2025-10-18
### Added
- FFMPEG build logging.
- A more robust dependency handler which should allow for a safe fallback for older NVIDIA GPUs ( GTX 1000 series and older )
- Added 6 new restore methods:
  - `deh264_real`, `deh264_real-tensorrt`, `deh264_real-directml` by @Phhofm
  - `deh264_span`, `deh264_span-tensorrt`, `deh264_span-directml` by @tntwise

### Changed
- Improved the robustness of the GPU detection logic.
- The arguments sections should only show the actually used arguments now.
- Bumped python to `v3.13.9`
- Bumped `torch` to `2.9.0`
- Migrated back from `.7z` to `.zip` for packaging in order to simplify the build process.

### Fixed
- Fixed a bug where certain unlucky users ( Non-NVidia ) would download the wrong dependencies.

## Adobe Edition

### Added
- Added `deh264_real` and `deh264_span` restore methods.
- Added a " Github Sponsors " button.
- Shortcuts options for the whole UI including settings in the `About` Tab.

### Changed
- Updated some descriptions in the UI for better clarity.
- Updated all of the dependencies to their latest versions.

### Fixed
- Potential fix for `.mov` files missing.

## [2.4.4] - 2025-09-18
### Added
- New upscaling methods: `fallin_soft`, `fallin_strong` with TensorRT and DirectML support by @renarchi

### Changed
- Bumped Python to `3.13.7`

## Adobe Edition

### New
- Added `Fallin Soft` and `Fallin Strong` to the Upscale Methods dropdown.
- Added `Trim to Work Area` button which automatically trims all selected layers ( incl comps ) to the duration of the work area.
- Added `Freeze on Current Frame` button which automatically freezes the layers ( incl comps ) to the desired frame.

### Changed
- Updated all dependencies to their latest versions.
- Made `Fallin Soft` as the default upscaling model.
- Minor UI tweaks.
- Removed Span from the upscale drodown.
- Removed the "NO" button from the Dependency Downloader of TAS as well as changing some logic where it will interchangeably use "Update" or "Download" based on the current state of the dependencies.

## [2.4.3.] - 2025-09-02
### Changed
- Reverted back to the old progress bar logic due to some issues with the new one.

## [2.4.2] - 2025-09-02
### Fixed
- Fixed an issue where ffmpeg would fail to download due to Charmap issues in Adobe Edition
- Theoretically fixed an issue for AMD Users where certain variables would not be defined.


## [2.4.1] - 2025-08-31
### Fixed
- An issue with appdata path seeking.

## [2.4.0] - 2025-08-31
### WARNING
- TAS is now using pytorch 2.8.0 + cuda 12.9, this means that older GPUs ( Maxwell, Pascal, and Volta ) with capabilities <= sm 70 have been fully deprecated and will no longer be supported. This means that TAS will no longer work on these GPUs, if you are using a GPU with sm 70 or lower, please stick to TAS 2.3.8 or earlier.
- With the migration to Celux, I strongly recommend deleting previous installations of TAS and reinstalling the latest version to avoid any potential issues with the new decoding backend.

### New
- Migrated, yet again, from Vapoursynth to `celux` for decoding, this should result in better performance and lower CPU usage.
- Added `--output_scale` parameter for custom output resolution scaling (e.g., `--output_scale 2560x1440`)
  - Supports any resolution in WIDTHxHEIGHT format (e.g., 2560x1440, 3840x2160, 1920x1080)
  - Uses bilinear scaling for minimal artifacts
  - Applied during FFmpeg encoding for optimal performance
  - Works with all processing methods (upscaling, interpolation, etc.)
- Added a lightweight dependency check that doesn’t import heavy libraries
  - Uses `importlib.metadata.version` to see if a package is installed
  - Uses `importlib.util.find_spec` to see if the module is discoverable
  - Prints a simple list like `Missing distributions detected: ['tensorrt']` before installing
- Added a small map for common package→module name differences
  - Example: `opencv-python` → `cv2`, `onnxruntime-gpu` → `onnxruntime`
- Added report of download speed to the progress bar.

#### Changed
- Improvements of up to `25%` in encoding bottlenecked workflows.
- Improvements of up to `10%` in Cuda/TensorRT workflows, partly due to the new torch/tensorrt updates.
- Dependency check flow is faster and simpler
  - Instead of importing `torch` just to “test it”, we now check presence via metadata
  - Only runs `pip install` when something is actually missing or mismatched
- Speed up `pip` calls
  - Added `--disable-pip-version-check` to `pip install`/`pip uninstall` to shave startup time
- Made the forced download helper more robust
  - Avoids referencing `isNvidia` when a specific requirements file is provided
- Slight modifications to imports to further tune start-up times.
- Upgraded to python `3.13.6`
- Up to `10%` faster GMFSS performance.
- Redesigned the progress bar.
- Added supression of the flask banner.

#### Fixed
- Fixed an issue with distill-directml depth methods where the output was not being properly handled.
- Fixed an issue with distill-directml depth methods where fp32 would crash.

## Adobe Edition
### New
- Added `Xpress 8k` compression method to the TAS-Portable folder resulting in up to a 50% reduction in final binary size, highly experimental and not recommended for everyone.
- An attempt to solve an issue where Pre-Renders would not work properly in the Adobe Edition due to different UI language.
- Added Resize Output option to the Adobe Edition, this will resize the output to the specified resolution, this is primarily used for upscaling workflows.
- Added back `ProRes` Encoding.
- Added breadcrumbs to the `Chain` tab to showcase each step the Input takes.

### Changed
- Improved the design of the `LogTab` and `AboutTab`

### Fixed
- Fixed an issue where the screenshot functionality would not parse 3rd party plugins properly.
- Fixed an issue where the ProRes necessary extension ( .mov ) wasn't properly added to the output file.

## [2.3.8] - 2025-08-03
#### Changed
- Updated some of the imports in the depth methods to be more efficient 

#### Fixed
- Missing `einops` dependency.

### Adobe Edition

#### Changed
- Update the UI, making it more responsive, compact and easier to use.
- Added UTF-8 encoding to the Adobe Edition to prevent issues with non-ASCII characters in paths ( WIP fix, not 100% yet ).
- Added an Error Toast when the composition width or height is non divisible by 2.

#### Fixed
- Fixed an issue where the progress bar would not update properly when using the Adobe Edition.

## [2.3.7] - 2025-07-30
#### Added
- Download update to the AdobeEdition progress bar.
  
#### Changed
- N/A

### Adobe Edition

#### Fixed
- Fixed an issue where the copy logs button would not work properly in the Adobe Edition.


## [2.3.6] - 2025-07-28
### AdobeEdition
#### Fixed
- String handling for output generation.
- Graph tab would crash the UI ( WIP fix, not 100%

## [2.3.5] - 2025-07-28
#### New
- Added the following depth methods:
  - `distill_small_v2-tensorrt`
  - `distill_base_v2-tensorrt`
  - `distill_large_v2-tensorrt`
  - `distill_small_v2-directml`
  - `distill_base_v2-directml`
  - `distill_large_v2-directml`
  - `og_small_v2-tensorrt`
  - `og_base_v2-tensorrt`
  - `og_large_v2-tensorrt`
  - `og_distill_small_v2-tensorrt`
  - `og_distill_base_v2-tensorrt`
  - Benchmark: Encoding + Decoding with `og_small_v2-tensorrt`, fp16, 1080p video, 13700k + 3090:
  Elapsed Time: 0:00:12 • ETA: 0:00:00 • FPS: 119.36 • Mem: 954.6MB • Frames: 1495/1495

#### Changed
- Fully rewrote `parameters.md` to reflect the new changes and to better explain the usage of TAS.
- Better lazy importing of the `ProgressBarLogic` in `main.py` have improved the startup time of TAS by ~100ms.

#### Fixed
- Issues with yt-dlp where it would add extra container extensions to the output file, causing a discrepancy between the expected and actual output file format.

## TAS Adobe Edition
### New
- Added resolution preview in upscale settings dialog - now you can see input dimensions (like 1920×1080) and output dimensions (3840×2160) before processing
- Added FPS preview in interpolation settings dialog - shows current composition FPS and what FPS you will get after interpolation (example: 24.000 FPS → 48.000 FPS for 2x)
- New functions getCompDimensions() and getCompFPS() in After Effects script to get composition information
Viewport zoom slider was added to `ToolBox`
- Added a brand new Graph Editor ( Alpha )
- Added a function to dynamically fetch the available space in C drive and display it in the download tas dependencies dialog.
- Migrated from `yarn/npm` to `bun` for package management, improving build performance and dependency management.

### Changed
- Made big performance improvements to main interface - now it renders much faster and uses less memory
- Converted many functions to use useCallback and useMemo for better React performance
- Moved progress bar display to separate component file for cleaner code organization
- Resolution and FPS previews automatically fetch data when you open the settings dialogs
- Slight refactor to the dependencies dialog to improve readability.
- Removed Motion and fade in/out effects in order to retain snappiness. 
- Migrated from using `exec` to `spawn` for process execution, improving performance and reliability.
- A complete overhaul to `logTab` to make it more performant and user-friendly.
- Updated the `executeProcess` function to handle errors more gracefully and provide better user feedback.

### Fixed
- Fixed issue where executeProcess function was used before it was declared (TypeScript error)
- Loading states now show properly when fetching composition information


## [2.3.2] - 2025-07-19
#### Fixed
- Fixed a misshap in the `requirements.txt` file where `Flask-Cors` was not included, causing issues with CORS handling in the Adobe Edition.

## [2.3.1] - 2025-07-18
#### Added
- Added support for more color formats / pix formats.

#### Changed
- Updated the `aeComms.py` to use a more efficient event stream logic.
- Deprecated 'socketio' in favor of a less headachey approach.

#### Fixed
- Issues such as: `Decoding error: Resize error: Resize error 3074: no path between colorspaces (2/2/2 => 0/2/2). May need to specify additional colorspace parameters` with https://github.com/NevermindNilas/TheAnimeScripter/issues/128

## TAS Adobe Edition
### Changed
- Better I/O handling if the pre-render input file is a video file.
- Removed `socket.io-client` dependency and replaced with flask.
- Massive refactor to the code and better usage of helpers.
- Improved error handling and logging throughout the application.
- Slightly improved logging throughout to `LogTab`.

### Fixed
- Fixed an issue where the naming scheme for pre-rendered files was not consistent.
- Fixed an issue wehre the progress bar was not updating correctly during processing.


## [2.3.0] - 2025-07-16
#### Added
- N/A

#### Changed
- Minor tweaks to TensorRT Handler.
- Minor tweaks to the progress bar logic.
- Turned `--ae` into a string argument rather than a boolean and it will now require a url to be parsed for more flexibility.
- Added some more logging to `aecomms.py` for better debugging.
- Removed `--realtime` argument and related logic.

#### Fixed
- Fixed a typo when using compile mode and upscale.
- Potential fix for deadlocks when installing dependencies.

## TAS Adobe Edition
### New
- Introduced `addPortToCommand` function to dynamically add the AE comms server URL to commands.
- Improved error handling and reconnection logic in the socket connection.
- Added a new `startExtraTabLogic` function to handle logic for extra tabs like depth map extraction and background removal.
- Added a new supporters pannel.

### Changed
- Updated WebSocket to dynamically adjust the port based on if a port is already in use or not.
- Updated the `main.tsx` file to include more detailed descriptions for depth map quality settings.
- Removed unused functions and imports to clean up the codebase.

## [2.2.9] - 2025-07-07
### Adobe Edition

#### Added 
- Autoscroll to the bottom for the log tab.
- Memoization and caching to the logging logic making it faster and less resource heavy ( probably min maxing but it should be significantly better now ).
- Added Report Issue Button to the Log Tab which copies the logs and opens the Github Issues page.
- Added support for shortened youtube links ( youtu.be )

#### Changed
- Better copy feature for the logs with proper newline handling.
- Added smooth autoscrolling in the logTab.
- Renamed application settings to advanced settings
- Renamed GTX (CUDA) to CUDA models

#### Fixed
- FFMPEG Encoding command becoming red.


## [2.2.9] - 2025-07-04
#### New
- Added new SocketIO based approach to the progress bar, this should result in a more responsive and faster parsing between TAS and AE.
- Switched to @TNTWise's spandrel fork and removed spandrel from the dependencies.
- Added a new option to `build.py` through argparse, `--develop` this is primarily used for development purposes and will not be documented.

#### Changed
- Up to `50%` faster inference speeds with `og_small_v2`, `og_base_v2`, `og_large_v2`, `og_distill_small_v2`, and `og_distill_base_v2` depth models with absolutely 0 quality loss.

### Adobe Edition

#### New
- Added a new `Progress Bar` which is now based on SocketIO and should be more responsive and faster.
- The status bar will now show the current status of TensorRT workflows, this is primarily used for the `engine generation` step for now but will be further expanded in the future.

#### Changed
- Switch animations from a sliding animation to a fade-in animation, also decreased the animation duration to 0.2s.
- Small tweaks to the Progress Bar.

#### Fixed
- Layer sorting was using the workarea start and end points rather than the actual layer start and end points.

#### Known Issues
- ProRes encoding is not currently working.
- AutoCutClip is not currently working.
- DPIR is not currently working.

## [2.2.8] - 2025-06-23
#### Added
- Added `triton-windows` as well as support for `torch.compile()` using command `--compile_mode` with options `["default", "max", "max-graphs"]`. Initial benchmarks show impressive gains for CUDA specific workflows. Tested with a 1080p video on a 13700k with a 3090 and `--benchmark` mode on and a 1 frame warm-up period. Results:
  - **SuperUltraCompact:** 80.80fps → 92.07fps - Mode: Max-Graphs
  - **Rife 4.6:** 140.48fps → 165.51fps - Mode: Max-Graphs
  - **Small Depth Anything V2:** 108.74fps → 131.19fps - Mode: Max-Graphs
  - **GMFSS:** 6.05fps → 9.51fps - Mode: Max
  - **OG Small Depth Anything V2:** 15.72fps → 16.02fps (likely OpenCV bottlenecked) - Mode: Max-Graphs
  - DO NOTE THAT THIS IS STILL UNDER DEVELOPMENT AND IT ONLY WORKS ON RTX GPUs.

#### Changed
- Added precision to the printed tensor shapes when building an engine.
- Updated the bugreport.md

#### Fixed
- Fixed an issue where Adobe Edition would not work due to missmatch in input pix formats.

## [2.2.7] - 2025-06-21
#### Added
- N/A

#### Changed
- Removed some unused debugging code from progressbarLogic.py

#### Fixed
- A mishap on my side when dependencies were being downloaded resulting in unnecessary checks, upgrades and caches. Please run a `.\python.exe -m pip cache purge` if you are upgrading from a previous version.

### Adobe Edition
#### New
- Thanks to the community behind Motion, I was able to add a fully custom animation styling to TAS. Tabs will now `swipe` from the left to right, the progress bar will have a cool little animation both when appearing and disappearing ( and more in the future )
- Added a selective `fp16` or `fp32` precision to the `Extra Tab`, primarily focused on helping users with lower end systems who were running into black frame outputs.

#### Changed
- Removed the preview image primarily because it wasn't really useful for much and it only added extra logic to worry about.
- Small tweaks around `useEffects`.
- Updated some vite-cep dependencies.

#### Fixed
- A potential deadlock when streaming the logs to the logsTab.


## [2.2.5] - 2025-06-20
#### Added
- A whole decoding backend migration to `Vapoursynth` including better I/O Support (WIP!!!)
- Added support for automatic deprecated dependencies removal, see more in `deprecated-requirements.txt` & `src.utils.DependencyHandler`
- Added `VideoTimeStamps` library by @moi15moi for accurate `--inpoint` and `--outpoint` handling.
- New depth methods, `distill_large_v2` and `og_distill_large_v2`

#### Changed
- Switched from yuv420p encoding to rgb24 in order to further minimize CPU usage across the board. This change should imply about a 5% lower CPU usage with minal loss to performance.
- Upgraded TensorRT to `10.12` , this brings a series of goodies with it, I am seeing an up to 10% performance lift in non encode / decode bottlenecked workflows as well as significantly faster engine generation times ( 30s -> 8s for SUC-TensorRT, 30s -> 20s for Rife4.6-TensorRT )
- Upgraded Pytorch to `2.7.1`
- Upgraded Torchvision to `0.22.1`
- More error handling around handle dependencies
- Migrated from `Basswood-AV` -> `Vapoursynth`, this should bring with it faster performance and better I/O handling. I am entertaining the idea of using a full Vapoursynth workflow ( through `vspipe` ) although personal benchmarks show up to `2x` lower performance in upscaling - cuda.
- Removed `explicit batch` from TRT and added `strongly typed`. 
- Upgraded the packaged python from `3.13.3` to `3.13.5`.
- Up to `50%` faster inference speeds with `rife-ncnn`.
- Updated build instructions.
- Updated the `slow_av1` encode-method.
- Optimized the logic behind the progress bar. Adobe updates should now be nearly instantaneous. 
- The progress bar will now also log `fps` and `eta` for AdobeEdition.

#### Fixed
- Fixed issues with GMFSS.
- Fixed issues with Scunet-TensorRT
- Fixed a bug where AMD / Intel GPU users would default to full dependencies esentially downloading more than they actually needed.
- Fixed issues with `--inpoint` and `--outpoint`
- Fixed issues regarding `bt709` & `yuv420p` formats, this further extends to anything compatible with `yuv420p` such as `rgb24` and `bgr24`. Special thanks to [@JET](https://github.com/Jaded-Encoding-Thaumaturgy) community for the help in debugging.

#### Known issues
- `--resize_method` is currently not implemented, due to the new decoding approach.
- I/O support is wonky until I will be able to test more color ranges and inputs, please report any issues.

### Adobe Edition
#### New
- Added dynamic output examples to the `Upscale` and `Depth` Settings.
- Added `distill large` to the depth extraction methods.
- Added a dynamic ETA and FPS to the progress bar.

#### Changed
- Renamed `NVIDIA GTX GPUs` sections to simply `NVIDIA GPUs`
- Added `CUDA` extension wherever approapiate.
- Simplified and memoized some logic.
- The progress bar will now properly update once every 1s cycle ( assuming an update did happen on the backend side )
- Better text formatting for the progress bar.
- TAS will now use `Chain` & `Depth` and `RmvBackground` as the generic keyword for each process output.
- Added logic to use the layer name when pre-rendering if only 1 layer is selected. 

#### Notes: 
- INTERNALIZATION HAS YET TO BE ADDED, non English characters may or may not break TAS' workflow.

#### Fixed
- Potentially fixed issues for AMD/Intel users when downloading the backend.

## [2.2.2] - 2025-05-30
### Adobe Edition
#### Fixed
- Double render issues with precomps
- WorkArea wasn't properly set if the work area was out of boundaries
- No warning if no layer is selected with ' Extract Depth Map ' and ' Remove Background '


## [2.2.1] - 2025-05-28
#### Added
- Restore method `gater3` | `gater3-directml`, special thanks to @umzi2 ( tensorrt and cuda under investigation )
- Added experimental `scunet-tensorrt` backend ( VERY VRAM HUNGRY! )

#### Changed
- Small tweaks to the upscale cuda pipeline and cuda model optimizer.
- Improvements to the restore cuda and trt pipelines.
- Changed the dependencyChecker logic simplifying it.

#### Fixed
- A regression to start-up times caused by overengineering the dependencyHandler.py

### Adobe Edition
#### New
- Added `Gater3 | Gater3DirectML | SCUNet-TensorRT` to the main UI
- Added `Remove Dead Frames` option to ToolBox, this works all on the AE side and is based on SSIM.

#### Changes
- Added a `debouncer` to the settings savings logic. 
- Created a helper function for the `contextualHelper(s)` in order to make the UI more consistent.
- Defaulted to a `8px` gap size basically everywhere. ( more work needed )
- Improved the robustness of the Automated backend download handler as well as improved its logging efficiency.

#### Fixed
- Fixed an issue where the automated download functionality would freeze upon completion locking the user out of TAS ( whoopsies )

## [2.2.0] - 2025-05-26
#### Added
- A fully redesigned and naive build method.
- Added `--download_requirements` file to download the requirements as a helper functionality for Adobe After Effects.
- Added `--cleanup` to uninstall unused dependencies or migrated ones ( e.g. PyAV -> BassWood-AV )
- Added `png` output support with `--encode_method png`

#### Changed
- With the recent changes to Pytorch and TensorRT and the ever increasing wheel sizes, I was forced to adapt a more dynamic approach to dependency handling, TAS will now dynamically download most of the libraries and only ship core needed libraries or very small ones, this lowers initial download sizes from 2.5GB ( compressed ) all the way to +-30MB ( compressed ), obviously this implies a major rework but down the line it will be easier to update TAS since dependencies don't change as often as the underlying code of TAS.
- Reworked `argumentsChecker.py` for better maintainabillity
- Upgraded pytorch to 2.7.0 adding BlackWell support.
- New usage method in the CLI will be `.\python main.py` based.
- Changed the detection method for `autoclip` to `AdaptiveDetector` for the higher performance and accuracy.
- Added a forced break even if `frame` is `None`.
- Switched from `pyav` to `basswood-av`
- Upgraded to TensorRT v10.10
- Logging all currently available torch.cuda devices.

#### Deprecations
- Fully removed `anipy-cli` from TAS.
- Temporarely removed `gputil` until I find a better solution.

#### Fixed
- `shufflecugan-ncnn` defaulting to non-existent `directml` backend when not available.
- Fixed an inversion issue regarding `--autoclip_sens`
- Fixed a cuda race condition when chaining upscale trt models with other cuda / trt workflows.
- Fixed a bug in gmfss
- Fixed an issue where Rife TensorRT would use double the necessary VRAM to generate an engine.
- Fixed an issue where Rife CUDA's timestep generation would overshoot and create artifacts.
- Fixed an issue where Fractional Interpolation wouldn't evenly distribute across the video, special thanks to @HolyWU for the base implementation idea. ( Hopefully I was able to translate it well enough )
- Fixed an issue with yt-dlp where it would not be able to find .mp4 queries.

### Adobe Edition
#### New
- The UI comes in 1 more new flavor with larger scale, specifically for 4k/5k monitors.

#### Changed
- Moved encoder / outputs related settings to `Application Settings` in the `Extra` Tab.
- Simplified the about tab
- Added a visual warning if the Youtube URL is not in the correct format.
- Disclosure panels have been changed to `Accordion` for better spacing and the autoclosing feature.
- Precompose will no longer create alerts and will instead use `Toasts`
- Simplified Toast generation in order to not stack the UI with toasts.
- Updated all react-spectrum dependencies.
- `Enhanced Error Handling:`
  - Added comprehensive try/catch blocks around critical file operations
  - Ensured all errors are properly logged and presented to the user
- `Improved Folder Management:`
  - Added proper parent directory validation and creation
  - More reliable cleanup of legacy TAS folders
  - Better handling of empty installation folders
- `Robust Version Checking:`
  - Better error handling for version detection failures
  - Improved version comparison logic
  - Added specific error messages for version check failures
- `Code Structure Improvements:`
  - Reorganized logical flow for better readability
  - Added more descriptive variable names
  - Used finally block to ensure initialization state is always updated
- `Enhanced Debugging Support:`
  - Added more detailed console logging
  - Better status reporting during dependency checks

#### Fixed
- Few minor bugfixes.

## [2.1.5] - 2025-04-14
### The Anime Scripter
#### Added
- Improved model download progress tracking with percentage-based milestones (20%, 40%, 60%, 80%, 100%) and file size reporting in logs

#### Changed
- Removed `torchmetrics` & replaced all SSIM relevant functions with a custom SSIM.
- A major redesign of the video decoding/encoding pipeline resulting in lower CPU usage, lower memory consumption and better encoding performance across the board.

#### Fixed
- Potential import errors caused by `signal.windows` missing in the bootstrapper of pyinstaller(?).
- fixed `--outpoint`.

#### Known issues
- `--inpoint` doesn't currently work.

### Adobe Edition
#### Added
- N/A

#### Changed
- Largely insignificant tweaks to some margins.
- Removed log overflow generated by CEP on runtime.

#### Fixed
- N/A

## [2.1.3] - 2025-04-08
#### Added
- Test(s)
- Slowmotion support through `--slowmo`
- `constants.py` to more easily maintain the code and to avoid excessive imports and declarations.
- Experimental fractional interpolation support.
- Added `memory usage` to the progress bar for more info.
- Added `"og_small_v2"` / `"og_base_v2"` / `"og_large_v2"` / `"og_distill_small_v2"` /  `"og_distill_base_v2"` which as the name implies are a near original paper implementation of Depth Anything V2 and Distill Depth Anything V2. These are roughly 10-20% faster than the original paper whilst retaining, visually, the same quality.
- Added `anime1080fixer-directml` restore method.

#### Changed
- Upgraded to `python 3.13` including most dependencies.
- Added an automated memdealloc in `buildbuffer`
- Migrated to `onnx-weekly` until `onnx v18.0` gets released.
- Linux support will be indefinitely postponed and will no longer be in any shape or form a priority of TAS.
- Disabled audio when running `--depth`
- Removed `h266`
- Cleaned up the code.
- Some small adjustments to `main.py`
- Increased the performance and accuracy of all dedup methods.
- Lowered CPU usage by `~5%` on my system, this should be more significant for lower end cpus.
- Significant improvements to `--preview` with better performance and lower overhead than before.
- Switched from `celux` to `pyav` due to lack of development.
- Bumped Pytorch to `2.6.0`
- Deprecated: `--resize_method` and `--decode_threads`, in the future these well be readed with a better design, for now use `--custom_encoder`.

#### Known Issues:
- `--inpoint` and `--outpoint` are not functional due to the migration from `celux` to `pyav`, I will need to see how to set interest points based on them.

#### Fixed
- Fixed more issues with non `utf-8` compliant paths. [Thanks Sergio]
- Fixed an issue with `slow_nvenc_av1` [Thanks Sergio]

### Adobe Edition
#### Added
- Added a `copy to clipboard` button to the `logs` tab.
- Added `slowmo` button to the interpolation tab.
- Added `Force destroy all TAS processes` button which kills any lingering tasks in the background, just in case.

#### Changed
- Significant UI refresh across the board.
- "Path Errors" will now result in toasts rather than annoying alerts.
- Removed a couple of unnecessary alerts, there should only now be one or two left for critical pieces. Down the line these will also be Toasts.
- Added a `latest version` indicator the update toast generator.
- Changed the `Interpolation Factor` dropdown to an editable area + dropdown with support for custom values and fractional factors.
- Updated some descriptions and icons.
- Added a `twitter` button to the socials.
- Removed some unnecessary logs

#### Fixed
- N/A 

## [2.1.2] - 2025-03-14
#### Added
- N/A

#### Changed
- More error handlers for easier debugging

#### Fixed
- An issue where utf-8 decoding would break Adobe Edition.



## [2.1.1] - 2025-03-13
#### Added
- 2 New Depth Models, `distill_small_v2` and `distill_base_v2` which provide better quality with better performance.
- `medium` depth quality for better output results for marginal performance loss.

#### Changed
- Small tweaks to depth maps which should generate better outputs overall.

### Fixed
- A bug within the download models importing
- 

## [2.1.0] - 2025-03-12
#### Added
- An automated error handler for when the user wants to use a model which does not work on their system ( WIP ).

### Changed
- Removed unnecessary clamping functionality on decode
- Updated the mpv configs.
- Lowered the base strenght of `fastlinedarken`.
- `psutil` and `gputil` as an temporary alternative to `wmi`.
- Upgraded TensorRT to `10.9.0` adding Nvidia Blackwell ( RTX 5000 ) support.
- Slightly simplified `main.py`
- Added more logging for when the user processes a directory rather than a simple file.
- Slight adjustments to the code to be more resource efficient.
- Minor improvements to initial launch time ( ~150ms )

### Fixed
- Performance regression when encoding
- Fixed some padding issues with `rife4.25-lite`

### Adobe Edition
#### Added
- Added a progress bar during inference.
- Added a `Logs` panel for easier debugging.
- Added a progress circle when TAS is processing.

#### Changed
- Huge chunks of UI generations are now dynamically imported and loaded to make maintaining easier and potentially improve performance.
- Fully rewrote the Toast generation from the ground up to be more maintainable.
- The command prompt will no longer show up when processing.
- Fully redesigned the `About` Tab.
- Many minor UI Tweaks.
- Added an example for segmentation.
- Changed the run button from only having a logo to an actual text `Run Chain` which is also slightly separated from the rest of the UI elements.
- Run Chain will now be disabled during processing.

#### Fixed
- N/a
---

## [2.0.5] - 2025-02-17

### The Anime Scripter
#### Added
- N/A

#### Changed
- Bumped numpy version

#### Fixed
- Fixed a bug in TensorRT shape parsing when resolution is above 1920x1080

### Adobe Edition
#### Added
- N/A

#### Changed
- Added a fallback to lossless when QuickTime isn’t available

#### Fixed
- N/A

---

## [2.0.4] - 2025-02-16

### The Anime Scripter
#### Added
- Updated `rtmosr` models to @umzi2’s "v2" models

#### Changed
- Bumped Python to `3.12.9`
- Improved compression logic for better results:
  - TAS FULL reduced to 1.7GB (from 1.82GB)
  - TAS LITE reduced to 155MB (from 186MB)
  - Overall extracted filesize decreased by ~100MB
- Improved subprocess performance in Python 3.12.9 (approx. 0.25s faster)

#### Removed
- Removed unneeded files and libraries from TAS

#### Fixed
- N/A

### Adobe Edition
#### Added
- Added `lossless` encode method
- Added tip(s)
- Enforced a Pre-Render step to avoid potential bugs
- Render Queue panel now auto-closes after rendering
- Added `Add Null Layer` button
- Added a visual indicator for inference speed
- Added dropdowns in the About Tab for System Requirements and GPU Compatibility

#### Changed
- Moved all Pre-Render options to the Encode Settings dialog
- Simplified backend logic around pre-render
- Renamed `rtmosr` to `rtmosr v2`
- Added an example to Depth Maps (more options coming)

#### Fixed
- Fixed visual anomalies with tooltips

#### Removed
- Removed Pre-Render button(s)

---

## [2.0.3] - 2025-02-03

### The Anime Scripter
#### Added
- N/A

#### Changed
- N/A

#### Fixed
- Fixed an issue where a static step was still parsed even if the current interpolation method didn’t support it

### Adobe Edition
#### Added
- N/A

#### Changed
- N/A

#### Fixed
- N/A

---

## [2.0.2] - 2025-02-02

### The Anime Scripter
#### Added
- Added `ffprobe` for more accurate metadata seeking
- Introduced experimental `--static_step` command to improve inference speed for Rife CUDA (available for models `rife4.6` and `rife4.15-lite`)

#### Changed
- Increased bitrate for `slow_nvenc_h265`
- Bumped celux from `0.5.4` to `0.6.0`
- Refactored parts of the ffmpeg code
- Improved error handling for yt-dlp
- Added a warning when using anipycli

#### Fixed
- Fixed a bug in `TRT Handler.py`
- Fixed issue with SVT AV1 generating unneeded login
- Fixed subtitle issues
- Fixed padding issues with `rife4.25`

#### Removed
- Removed `PyMediaInfo` for accuracy reasons
- Removed the mbps meter from the download progress bar
- Removed the `upscale_skip` and `interpolate_skip` commands

### Adobe Edition
#### Added
- Added `TensorRT Static Engine Generation` button in the upscaling settings

#### Changed
- Improved error handling
- Renamed some UI elements

#### Fixed
- N/A

#### Removed
- Removed `Check For updates` button

---

## [2.0.1] - 2025-01-12

### The Anime Scripter
#### Added
- Introduced `rtmosr-directml` and `rtmosr-tensorrt` upscaling methods by @umzi
- Added `slow_nvenc_av1` encoding preset

#### Changed
- Enhanced depth maps for TensorRT and DirectML
- Renamed log files (`TAS.log` → `TAS-Log.log`; `ffmpeg.log` → `FFmpeg-Log.log`)
- Generalized logging and reduced unnecessary terminal prints from ffmpeg (when using x265)
- Improved realtime playback with MPV preview and switched playback from FFmpeg to MPV for `--realtime`
- Skipped encoding during `--realtime` to save system resources
- Enhanced TRT shape printing and improved ffmpeg command handling
- Added forced exit (`sys.exit()`) if FFMPEG crashes
- Changed subtitle handling from `srt` to `copy` to retain metadata

#### Removed
- Deprecated `--depth_quality` option (models now run internally at 518x518)

#### Fixed
- Fixed shape issues with Segment Anime TensorRT
- Fixed issues with `rawvideo` and `celux`
- Fixed problems in the 16bit workflow with `nvenc_h264`

### Adobe Edition
#### Added
- Added a “Take a Screenshot” feature in the Toolbox
- Introduced a revamped About Panel with additional TAS info
- Added System Requirements in the About panel

#### Changed
- Improved the “Extracting TAS…” indicator
- Made minor backend adjustments for better reusability
- Defaulted depth quality to `low`
- Added error handling for image inputs (TAS works only with video)
- Updated UI elements and descriptions
- Refactored code for clarity

#### Fixed
- Fixed issues with the Remove Background segment not updating correctly
- Fixed repeated unnecessary tasks on UI mount

---

## [2.0.0] - 2024-12-29

### The Anime Scripter
#### Added
- Partially ported TAS to Adobe Premiere (core functionality now supported)
- Added Full CUDA Graph Acceleration for TensorRT and CUDA workflows (up to 25% performance boost)
- Introduced variable `--decode_threads` for more granularity
- Added `anime1080fixer-directml` support for restoration
- Added `slow_nvenc_h264` & `slow_nvenc_h265`
- Introduced new features: `--restore_method`, `fastlinedarken`, and `fastlinedarken-tensorrt`
- Added support for dynamic `upscale_factor` (for CUDA workflows)
- Introduced experimental `--realtime` mode for real-time output preview
- Reintroduced GMFSS support
- Added Dynamic Flow Scale via `--dynamic_scale` (experimental)
- Temporarily added video input metadata collection
- Added Flow-based deduplication with FlownetS (experimental)

#### Changed
- Replaced the existing progress bar with Rich Progress for faster startup and improved FPS accuracy
- Improved inference speed for 16bit workloads (up to 50% faster)
- Enhanced error handling around ffmpeg calls
- Upgraded TensorRT to version 10.7.0
- Refactored the TRT Handler backend for better logging and granularity
- Optimized normalization by removing redundant operations
- Simplified CUDA logs and calls
- Converted the `--half` argument from boolean to string (default: True; options: [True, False])
- Improved segmentation efficiency using CUDA
- Bumped Pytorch from `2.5.0` to `2.5.1`
- Added a torch.fx model optimization backend (up to 20% CUDA inference speed boost)
- Upgraded numpy to version 2.2.1

#### Notes
- Due to the TensorRT upgrade, all engines must be regenerated

#### Removed
- Removed Polygraphy in favor of the built-in TensorRT API
- Removed the `--sample_size` command and pre-compilation of TAS’ bytecode
- Removed image support and Alive-Progress (now using Rich-Progress)

#### Fixed
- Fixed issues with Rife NCNN
- Fixed CUDA race conditions at interpolation factors above 2x
- Fixed audio copying for .webm inputs
- Fixed depth TensorRT flickering in low quality mode
- Fixed black output issues with Upscale NCNN
- Fixed a bug causing the first frame to be black in DirectML workflows
- Fixed Rife Elexor and related import issues
- Fixed misuse of `--encode_method` with .webm containers
- Fixed a bug where TAS would output only a temp file if `--output` was declared without other processing

### Adobe Edition
#### Notes
- This version is built against AE 2025. Some functionalities may not work with older versions.
- Update checking remains manual for now.

#### Added
- Introduced an auto Update Checking system
- Added proper Pre-Comp support to avoid user errors
- Added new interpolation and encoding methods
- Added quick action buttons for “Add Adjustment Layer” and “Add Solid Layer”
- Introduced an “Arrange Layers” function for easier sorting
- Added tooltips for enhanced user guidance
- Added full-path indicators for backend download locations
- Introduced Toast notifications as an alternative to alerts
- Added an “Open Changelogs” button

#### Changed
- Renamed certain UI elements and improved error handling
- Enhanced preview window responsiveness and backend reusability

#### Fixed
- Fixed Remove Background update issues
- Fixed repeated tasks on UI mount

---

## [1.9.11] - 2024-10-26

### The Anime Scripter
#### Added
- Added H266 encoding support via `--encode_method h266`
- Added `slow_x264`, `slow_x265`, and `slow_av1` encoding methods
- Introduced `rife_elexor` and `rife_elexor-tensorrt`
- Renamed `--denoise` to `--restore`
- Added Anime1080Fixer by @Zarxrax to `--restore`
- Added early TRT backend for `anime1080fixer-tensorrt`
- Added support for `yuv420p10le` and `yuv422p8le` inputs
- Enabled anipy support using `--input "anime"`

#### Changed
- Restructured TAS’ backend for improved I/O handling
- Upgraded the GitHub Actions runner (Python 3.12.4 → 3.12.7)
- Optimized memory allocations and defaults for `--upscale_skip`
- Removed redundant calculations in deduplication
- Upgraded Pytorch from 2.4.0 to 2.5.0 and improved input/output handling

#### Fixed
- Fixed variable issues in the 16bit workflow
- Fixed RIFE TRT duplicate frame issues
- Fixed benchmark script issues caused by backend changes

#### Breaking Changes
- All engines must be rebuilt due to the TensorRT upgrade

### Adobe Edition
#### Fixed
- Fixed model selection issues between Full and Lite methods

---

## [1.9.10] - 2024-09-29

### The Anime Scripter
#### Fixed
- Fixed issues where missing `--inpoint` and `--outpoint` caused FFMPEG crashes
- Fixed sharpening values exceeding allowed bounds

---

## [1.9.9] - 2024-09-28

### The Anime Scripter
#### Added
- Enhanced encoding now supports real-time video processing with adequate hardware
- Added Rife 4.25 and Rife 4.25-TensorRT
- Reintroduced VP9 and QSV_VP9 encoding for .webm inputs
- Precompiled `.pyc` files now included for faster startup
- Restructured the interpolation pipeline for TensorRT (up to 25% performance gain)
- Implemented a CUDA pinned memory workflow to boost data transfer efficiency
- Upgraded Span models to Spanimation-V2

#### Changed
- Reduced the Rife TensorRT weights folder size (recommend deleting the folder to regenerate engines)
- Enhanced metadata retrieval using pymediainfo
- Improved benchmarking accuracy and merged audio/subtitle processing during encoding
- Enabled parallel processing between encoding and inference
- Improved FFMPEG logging and overall system checks
- Auto-create output folders if missing

#### Fixed
- Fixed exaggerated performance reports
- Fixed a regression from 1.9.8 impacting encoding performance
- Fixed the first frame being black in upscaling
- Fixed frame processing issues from 1.9.8

### Adobe Edition
#### Added
- Added a Manual Method button for step-by-step TAS backend installation

#### Changed
- Enhanced backend download functionality

---

## [1.9.8] - 2024-09-15

### The Anime Scripter
#### Added
- Extended `--input` to support:
  - `.txt` files with video lists
  - Multiple videos in a single string (separated by semicolons)
- Moved interpolation before upscaling for higher quality and VRAM efficiency
- Added support for multi-subtitle and multi-audio extraction
- Introduced `--preset` and `--preset_list` for saving and reusing configurations
- Improved CTRL+C support for graceful termination

#### Changed
- Enhanced upscale TRT performance (~10% faster)
- Implemented new CUDA detection coroutines for better performance
- Decoupled STDOUT from the main loop (up to 50% performance boost)
- Reduced total model size and improved TensorRT engine robustness
- Rewrote backend chained processes and optimized deduplication (both SSIM-CUDA and CPU)
- Upgraded Torch from 2.3.1 to 2.4.1
- Improved Linux build sizes and encoding efficiency

#### Removed
- Removed the Scikit-Image package

#### Regressions
- Noted a RAM usage increase (~500MB) due to new encoding changes

### The Anime Scripter - Adobe Edition
#### Added
- Enabled Preview Windows using `--preview`
- Added new options for Remove Background with TensorRT acceleration
- Made Pre-Render more flexible with a choice between Lossless (default) and QuickTime

#### Changed
- UI cleanup and minor adjustments

#### Fixed
- Fixed UI issues and minor scaling problems

---

## [1.9.7] - 2024-09-02

### The Anime Scripter
#### Added
- Added Rife4.20-NCNN, 4.21-NCNN, 4.22-NCNN, and 4.22-Lite-NCNN

#### Fixed
- Fixed CUDA errors in the Lite Edition for Windows

#### Regressions
- Some performance gains from 1.9.6 were lost due to dependency handling issues

### Adobe Edition
#### Added
- Added new NCNN Models

#### Changed
- Backend downloads now up to 2300% faster
- Fixed race condition issues and improved path/error handling

#### Removed
- Removed older 4.15-lite & 4.17 models in favor of newer, more performant options

---

## [1.9.6] - 2024-09-02

### The Anime Scripter
#### Added
- Introduced Rife4.22-Lite and Rife4.22-Lite-Tensorrt
- Added `--static` flag to force static engine generation for Upscale TensorRT
- Added temporary state for ensuring file integrity during model downloads
- Introduced `--depth_quality` flag (options: low, high; default is high)
- Provided testing builds for linux-full and linux-lite

#### Changed
- Upgraded base polygraphy from 0.49.12 to 0.49.13
- Removed unnecessary clamps and optimized metadata/progress reporting
- Made log.txt more compact by removing unused arguments
- Ensured Rife TRT builds static engines for improved stability
- Improved Rife CUDA and TRT performance (up to 15% faster)
- Enhanced chained processes for better memory efficiency and benchmarking
- Upgraded ONNXRuntime-DirectML to 1.19.1 and improved download progress reporting

#### Fixed
- Fixed CUDA race conditions in extremely high FPS workflows
- Fixed issues in Rife 4.22 and Rife 4.22-lite that reduced output quality

### Adobe Edition
#### Added
- Added Rife4.22-Lite and Rife4.22-Lite-Tensorrt options to the dropdown menus
- Introduced Depth Quality options (low and high)

#### Fixed
- Fixed path issues when spaces are present

---

## [1.9.5] - 2024-08-17

### The Anime Scripter
- No significant changes

### Adobe Edition
#### Fixed
- Fixed issues with Full vs. Lite method selection during model downloads

---

## [1.9.4] - 2024-08-17

### The Anime Scripter
#### Added
- Introduced a new 3D-like Banner
- Added OpenProteus and AniScale2 by Sirosky
- Launched a new lite version of TAS (WIP)
- Added support for multi-audio and multi-subtitle extraction
- Introduced ShuffleSpan by Sudo (WIP)
- Added Differential and Differential-TensorRT scene change methods (WIP)
- Added Shift_LPIPS-TensorRT scene change methods by Sudo
- Added `--hide_banner` flag to suppress the TAS banner
- Introduced preview functionality via `--preview` (access via 127.0.0.1:5050/frame)

#### Changed
- Improved parameter examples and upgraded to nightly ONNXRuntime DirectML
- Added recursive checks for model downloads
- Enhanced I/O folder handling for better edge-case management

#### Fixed
- Fixed a bug with 16bit workflows and x264_animation_10bit
- Resolved issues with input/output folder case sensitivity
- Fixed concurrency issues with Rife TensorRT and high CPU usage
- Fixed span-directml host misdirection

#### Regressions
- Temporarily reverted to pytorch 2.2.2 and cuda 12.1 due to .dll issues

#### Removed
- Removed Updater.exe

### Adobe Edition
#### Added
- TAS now dynamically switches between “Full” and “Lite” modes based on GPU availability

#### Changed
- Improved error handling and minimized terminal disruption (now starts minimized)
- Enhanced output formatting with keyword highlighting (e.g., _DE for deduplication)
- Improved download functionality with potential for higher speeds

---

## [1.9.3] - 2024-08-09

### The Anime Scripter
#### Added
- Added Rife4.22 TensorRT
- Introduced Video Stabilization using Vidgear

#### Changed
- N/A

#### Removed
- N/A

### Adobe Edition
#### Added
- TBA

#### Changed
- TBA

---

## [1.9.2] - 2024-08-09

### The Anime Scripter
#### Added
- Upgraded to Python 3.12
- Added Rife 4.22 CUDA, Rife 4.21 CUDA and TensorRT, and Rife 4.20 TensorRT
- Introduced `--bit_depth` option (choices: 8bit, 16bit)
- Added experimental .GIF input support via `--encode_method gif`
- Added experimental .png/.jpeg support via `--encode_method image`
- Overhauled model downloading (recommend deleting the weights folder)
- Upgraded SPAN models to 2x_Spanimation V1.5

#### Changed
- Achieved up to 25% faster Rife TensorRT inference
- Improved TensorRT engine robustness and reduced available model count
- Streamlined deduplication and encoding performance

#### Removed
- Removed Rife4.15, Rife4.15-ncnn, and Rife4.15-TensorRT

#### Fixed
- Fixed TensorRT artifacting in Depth Maps
- Resolved inconsistent Depth Map outputs and aliasing issues
- Fixed issues with `--custom_encoder`
- Partially addressed issue [#44](https://github.com/NevermindNilas/TheAnimeScripter/issues/44)
- Fixed pathing issues for Linux

### The Anime Scripter - Adobe Edition
#### Added
- Enabled auto clip functionality
- Introduced half-automated TAS updates
- Added `--bit_depth` to encoding settings

#### Changed
- Various UI improvements

#### Fixed
- Fixed scaling issues and random TAS update popups

---

## [1.9.1] - 2024-07-30

### The Anime Scripter
#### Added
- Added Rife 4.20 CUDA
- Introduced Depth Anything V2 – DirectML
- Added `--scenechange_method` (options: maxvit-directml & maxvit-tensorrt)
- Added ProRes 444 HQ Encoder via `--encode_method prores` and ProRes encoding support

#### Changed
- Improved upscaling performance with TensorRT (up to 15% faster)
- Improved upscaling performance with CUDA (up to 20% faster)
- Enhanced interpolation performance with TensorRT (up to 10% faster) and CUDA (up to 30% faster)
- Decoupled scene change from interpolation with a performance boost (up to 20% faster with DirectML/TensorRT)
- Optimized Rife UHD mode and overall code formatting
- Upgraded Torch from 2.2.2 to 2.4.0 and CUDA from 12.1 to 12.5
- Reformatted help output for clarity

#### Fixed
- Fixed issues [#43](https://github.com/NevermindNilas/TheAnimeScripter/issues/43) and [#41](https://github.com/NevermindNilas/TheAnimeScripter/issues/41)
- Fixed scene change caching and benchmark script issues
- Resolved missing DLL issues with onnxruntime-directml

### Adobe Edition
#### Added
- Introduced a new secure installation method with code signing
- Enabled dynamic downloading and updating of TAS
- Enhanced Pre-Render to support multiple active/selected layers

#### Changed
- Redesigned the UI for a more compact layout (WIP)
- Moved the About page to a Help icon
- Improved tooltips and progress indicators
- Reduced the initial executable size (230kb) with Full and Lite versions available
- Updated dropdown hints and model sectioning
- Added new models and encoders; changed preset template from lossless to high quality

#### Fixed
- Fixed inpoint/outpoint boundary issues
- Fixed YouTube import issues when no composition was selected
- Fixed Pre-Render work area adjustment issues
