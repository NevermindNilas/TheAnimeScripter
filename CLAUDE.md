# TheAnimeScripter

TAS is a Python 3.14 (cp314) video-enhancement CLI and After Effects server; capabilities are in the table below. Backends: CUDA, TensorRT (TRT), DirectML (DML), OpenVINO, NCNN, MPS (Apple Silicon). MPS covers only upscale, interpolate, restore, depth (`src/depth/backends/mps.py`), and motion blur (via RIFE); every other capability has no MPS path yet. The `.jsx` UI is in another repo. Version: `src/version.py`.

## Commands

```powershell
python main.py --input <video> --output <video> [--upscale --interpolate --restore --dedup ...]
python -m pytest tests/ -q          # or a single file: tests/test_encodingSettings.py
python build.py                     # portable build; see BUILD.md (--develop redirects output)
```

- Before committing: `ruff check --fix` + `ruff format` + `uvx ty@0.0.63 check` + `python -m pytest tests/ -q`; update `CHANGELOG.MD` for user-visible changes.
- CLI definitions/choices: `src/cli/parser.py`; validation: `src/cli/validator.py`. `PARAMETERS.MD` is the full flag reference — update it whenever you add or change a flag.
- `build.py` downloads portable Python 3.14.6, installs dependencies, and copies `src` to `dist-portable/main/`.

### Test gotchas

- New tests needing torch/cv2/nelux must guard with `importorskip`.
- CI runs a bare venv: some tests install a fake torch stub only when real torch is absent (e.g. `tests/test_ffmpeg_writer.py` imports `src.io.ffmpegSettings` under it), so those modules must stay importable without real torch. Green locally with torch installed proves nothing about CI — and `import X` succeeding does not prove `from X import name` does; check lazy from-imports after any import refactor.

## Runtime map

- Entry: `main.py` → `VideoProcessor`. `start()` initializes models via `src/initializeModels.py`, creates I/O buffers, then runs read/process/write with `ThreadPoolExecutor`.
- Standard frame path in `processFrame()`: decode → dedup → restore → interpolate↔upscale → encode. `--interpolate_first` selects the order.
- Specialized autoclip/depth/segment/obj_detect/stabilize/motion_blur operations bypass that loop through `_selectProcessingMethod()`; their drivers are in `src/factories/standalone.py`.
- `initializeModels()` only decides which capabilities are on; the `match` on backend-suffixed method strings, the lazy backend import, and the returned inference callable live in `src/factories/{upscale,interpolate,restore,dedup,sceneChange,standalone}.py`.
- I/O: `src/io/ffmpegSettings.py`. `BuildBuffer` uses `nelux.VideoReader`; `createWriteBuffer()` selects `NeluxWriteBuffer` or legacy FFmpeg-subprocess `WriteBuffer`. Nelux handles audio/subtitle passthrough in `_setupPassthrough` and needs FFmpeg DLLs registered by `src/infra/getFFMPEG.py`.
- `requirements.txt` pins `nelux==0.16.0`; on cu132, verify `import nelux` because a wheel can be torch-ABI-incompatible.

| Capability | Implementation | Backends/details |
|---|---|---|
| Upscale | `src/upscale/` | CUDA, TRT, DML, OpenVINO, NCNN, MPS |
| Interpolate | `src/interpolate/` | RIFE; CUDA, TRT, DML, OpenVINO, NCNN, MPS |
| Restore (denoise) | `src/unifiedRestore.py` | CUDA, TRT, DML, OpenVINO, Maxine, MPS |
| Dedup | `src/dedup/dedup.py` | CUDA/CPU SSIM, MSE, flownets — sensitivity notes below |
| Smooth dedup | `main.py:processFrame`, `src/interpolate/_timesteps.py:gapPlan` | notes below |
| Depth | `src/depth/backends/` | CUDA, TRT, DML, OpenVINO, MPS; temporal VideoDepthAnything in `video.py` |
| Segment | `src/segment/animeSegment.py` | CUDA, TRT, DML |
| Object detect | `src/objectDetection/objectDetection.py` | TRT, DML, OpenVINO (the plain `ObjectDetection` class has no CLI choice) |
| Autoclip | `src/autoclip/` | PySceneDetect CPU, TRT, DML, TransNetV2 |
| Scene-cut (interp) | `src/sceneChange/` | streaming per-pair cut detector for `--scenechange`; ssim/mse + maxxvit 6ch (shared `scorer6ch.py`); holds instead of morphing across cuts |
| Stabilize | `src/stabilize/` | `--stabilize_method`: classic (ORB/LK global similarity, CPU; SuperPoint disabled) or dut (`dutStabilizer.py`, CUDA mesh warp, vendored net in `src/stabilize/dut/`) |
| Motion blur | `src/motionBlur.py` | `MotionBlurPipeline` |

- **Dedup sensitivity:** `--dedup_sens` default 35 maps to SSIM `1-s/1000`, VMAF `100-s/10`, flownets `s/100`; MSE takes it raw. SSIM/MSE come from `frame_analytics`, shared with `src/sceneChange/detector.py` and `src/rifearches/dynamic_scale.py`.
- **Smooth dedup (`--smooth_dedup`):** same detectors as dedup off `--smooth_dedup_method`/`--smooth_dedup_sens`, but a dropped duplicate widens the next gap instead of shortening the video; duration and audio preserved. Forces `--dedup` off, implies `--interpolate`. Runs longer than `--smooth_dedup_max_span` (6) are filled with copies.
- URL input: `src/ytdlp.py`.

## Change map

### Recipes

- **Add a model = TWO edits, plus hosting:** add its weight mapping to `src/model/registry.py:modelsMap()` and its CLI choice to `src/cli/parser.py` (and document it in `PARAMETERS.MD`). `modelsList()` is the canonical method registry, used by `src/infra/backendFallback.py` and pinned by `tests/test_registryDrift.py`; weightless/aliasing methods belong in that test's frozen per-capability exception sets. New weights must first be uploaded (maintainer-gated) as flat assets to the TAS-Models-Host `main` release — the code edits alone 404 at download. Weight URLs live in `registry.py` (`TASURL`; a few models use `SUDOURL`/HuggingFace).
- **Add a backend:** add a sibling backend class, a `match` arm in the matching `src/factories/*.py`, CLI choices, and `modelsList()`/`modelsMap()` entries. **Never rewrite the CUDA path.** A factory arm without a `parser.py` choice is dead code (`anime-openvino` is). `-mps` is stripped before `modelsMap()` — MPS backends reuse the base weight entry and add no `modelsMap` arm.
- Backend classes follow convention, not an ABC: dashless class suffixes (`UniversalPytorch`/`UniversalTensorRT`/`UniversalDirectML`/`UniversalNCNN`/`UniversalPytorchMPS`; `RifeCuda`/`RifeTensorRT`/etc.) implementing `__call__()` and `handleModel()`. CLI method strings use a `-<backend>` suffix.
- OpenVINO normally rides inside the DirectML/ORT class (only Segment has `AnimeSegmentOpenVino` — no CLI choice, dead). These families use dedicated model-specific classes instead of the `Universal*` path: `AnimeSR*`, `ArtCNN*`, `DistilDRBA*`, `NvidiaVSR`.

### Invariants

- **Temporal drivers declare their own lookahead.**
  - Declaration: a driver that needs neighbouring frames handed to it sets `temporalWindow = (past, future)` on the class (`AnimeSR*`, `DistilDRBA*` = `(0, 1)`). Frames it caches itself (RIFE's `I0`, AnimeSR's padded `prevFrame`) are not part of the declaration.
  - Ring sizing: `main.py:process` sizes a `src/io/frameWindow.py:FrameWindow` ring to the stage chain's demand — `max` in interpolate-first (interp and upscale both read the source stream), `sum` in interpolate-last (interp's neighbour is itself upscaled) — and hands drivers `frameWindow.successorFrame()`.
  - **Never gate a lookahead on a method-name string in `main.py`** — that is how `animesr-tensorrt`/`-directml`/`-openvino` silently ran with `next == curr`. `tests/test_frameWindow.py` fails if a new `AnimeSR*`/`DistilDRBA*` class omits the declaration.
- **The window has a domain, a validity, and an order.**
  - Domain: dedup and restore run at window *entry* (`main.py:_enterFrame`, passed as `FrameWindow(enter=…)`), so every slot is a restore-domain frame and a driver's neighbour matches it; deduped frames never enter, so restore is not spent on them.
  - Order: downstream upscale in interpolate-last is memoized per slot (`FrameWindow.staged`) so a frame two stages read is upscaled once, in order (AnimeSR's `prevFrame`/`state` recurrence stays sequential).
  - Validity: `successor()`/`successorFrame()` return `None` across a hard cut (`FrameSlot.isCut`), so `distildrba` gets no motion cue from the next shot. Because the neighbour is always same-domain, `distildrba` *raises* on a size mismatch instead of bilinearly resampling `I2`.
  - Counters live on the window (`consumed`/`dropped`); `dedupCount = frameWindow.dropped`. `FrameSlot.dupsBefore` carries how many frames entry dropped ahead of that slot — how smooth dedup recovers a gap's true source width without any lookahead.
- **Interp model resolution follows the pipeline order.**
  - Every interp backend sizes fixed I/O buffers (and a CUDA graph) at construction, so `initializeModels` resolves the resolution the driver will actually see — source dims in interpolate-first, post-upscale dims in interpolate-last (`self.upscale and not self.interpolateFirst`) — and passes it to `buildInterpolateProcess` as `interpWidth/interpHeight`, used by ALL backends.
  - **Do not revert any backend to `self.width/self.height`** or hand GMFSS the post-upscale dims directly; that reintroduces the mirror bugs (interpolate-last RIFE/DistilDRBA fed upscaled frames into input-sized buffers → hang; interpolate-first GMFSS fed source frames into output-sized buffers → shape assert).
- **One run, one outcome.**
  - `src/io/runOutcome.py:outputWasWritten()` is the single "did the output get written" test, used by both `main.py:_videoFailed` (batch exit code) and `src/server/aeComms.py:reportTerminalStatus` (the AE panel) — they must agree, and it knows an `--encode_method png` output is a `%05d` pattern rather than a file.
  - Every capability that bypasses `start()` reports for itself: the standalone drivers record `self.processingError` and `src/factories/standalone.py` hands it back; depth does it through `src/depth/backends/_shared.py:DepthRunOutcome` (`recordFailure`/`reportOutcome`/`guardedProcess`) because every depth backend class needs it.
  - A driver that swallows a per-frame exception must record it, or a run that dropped every frame reports success.
- **Scene-cut skip (`--scenechange`):**
  - `main.py:processFrame` runs a `src/sceneChange/` detector; on a cut it emits held frames and calls `interpolate_process.cacheFrameReset(frame)`.
  - One contract across ALL interp backends (every existing one implements the hook, GMFSS included): re-anchor `I0=frame`, re-seed encoder feature, keep `firstRun` false.
  - Graph/binding safety on `RifeCuda`/`RifeTensorRT`: re-seed `I0`/`f0` **in-place, never reassign** (arch `cacheReset` reassigns `f0` → would break CUDA-graph replay).
- **One gap planner for every factor.**
  - `main.py:processFrame` asks `src/interpolate/_timesteps.py:gapPlan(prevPos, curPos, factorNum, factorDen)` for the frames owed to the source interval `(prev, current]` — the interval every driver actually synthesizes, since `__call__` interpolates its cached `I0` against the frame handed in.
  - Integer factors over a unit gap get `timesteps=None` so the driver's own ladder runs and the timestep buffer stays cached; smooth dedup just widens `curPos`.
  - **Do not reintroduce a separate fractional-factor branch** — the old one planned `[current, next)` and landed a gap late.

### Perf/refactor landmines

- Keep `src/upscale/_shared` imports function-level lazy: a module-level import runs the package `__init__` → eager `.pytorch` → `modelOptimizer`'s `torch.contiguous_format` default arg → breaks torch-less loading (shipped once as PR #403; invisible locally with torch installed).
- Decode/encode tensor math is deliberately out-of-place — in-place `mul_`/`clamp_` there corrupts frames.
- The writer copies frames on its own private CUDA stream and never waits on yours: any new op added before `sink.put()` must `current_stream().synchronize()` first, or output goes non-deterministic (`record_stream()` does not fix it).
- The output `.clone()` after CUDA-graph replay guards the 32-deep lazily-drained writer queue — do not remove it.
- The DML per-frame rebind is a required workaround, not waste: ORT snapshots bound inputs at `bind_input()` time, so any loop-varying input must be rebound each iteration.
- `RifeCuda`'s graph path is not run-reproducible; judge output parity against the run-to-run noise floor, not bit-hashes.

### Pointers

- Weights: `modelsMap()` resolves name/scale/dtype; `resolveWeightPath()` uses `weights/{model}/`. TRT/DML/OpenVINO use `weights/{model}-onnx/`, except RIFE keeps its base folder.
- Output naming: `src/io/inputOutputHandler.py:generateOutputName()`; encoders: `src/io/encodingSettings.py:matchEncoder()` plus mirrored CLI choices.
- ONNX export: `tools/onnxConverter.py`; TensorRT engine build/cache: `src/model/trtHandler.py`.
- Global runtime state: `src/constants.py` (`WHEREAMIRUNFROM`, `SYSTEM`, `FFMPEGPATH`, `FFPROBEPATH`, `METADATAPATH`, `ADOBE`, `AUDIO`), initialized by CLI startup/validation and imported as `cs`.
- Dependencies/FFmpeg/hardware: `src/infra/{dependencyHandler,getFFMPEG,checkSpecs}.py`.
- AE bridge/preview/presets: `src/server/{aeComms,previewSettings,presetLogic}.py`; metadata: `src/io/getVideoMetadata.py`.
- Logs: `main()` sets `cs.LOG_PATH` to `TAS-Log.log` with overwrite mode; `src/infra/logAndPrint.py` wraps stdlib logging but does not choose the path.
- `src/utils/` holds only the live `tensorrt_import.py` (`trt` import shim, widely imported); the legacy compatibility shims are gone — import the canonical modules above.

## Vendored/model code boundary

Do not read, search, summarize, lint, or modify these trees unless the task explicitly targets that model (adding or porting an arch there counts) or a traceback enters it:

- `src/spandrel/`, `src/gmfss/`, `src/rifearches/`
- `src/depth/{distillanydepth,depth_anything_3,video_depth_anything,dinov2_layers}/`
- `src/stabilize/dut/` (a heavily-modified port; provenance and deviations in its `NOTICE.md`)

Exception: `src/rifearches/dynamic_scale.py` is first-party TAS code despite its location (the shared SSIM cache used by dedup/scenechange) — treat it as first-party.

Lint scope differs from this boundary: ruff's `extend-exclude` covers spandrel, gmfss, and the four depth trees but NOT `src/rifearches` or `src/stabilize/dut` — both are ruff-linted and kept clean by CI; ty additionally excludes `src/rifearches`. On py314, ruff strips parens from `except (A, B):` (PEP 758) — correct output, not damage; don't "fix" it.

`src/spandrel/` is the in-repo fork of TNTwise/spandrel at `e747f27` (`adding_extra_archs`), not a submodule; provenance is in `src/spandrel/NOTICE.md`. It is exposed through `src/spandrelCompat.py`. Restrictive `spandrel_extra_arches` code was removed. Keep architecture changes ONNX-exportable and FP16-capable. `src/extraArches/` is first-party model code (not vendored); dtype/memory-format model prep lives in `src/model/modelOptimizer.py`.

## Quality and build

- Ruff: config in `pyproject.toml`, CI pins 0.15.16, rules `E,F,I,UP,B`; the formatter owns line length. Keep the excluded vendored trees diff-clean against upstream.
- `ty` (pinned 0.0.63; run `uvx ty@0.0.63 check`, config in `pyproject.toml`) catches first-party names that do not exist — a `match` arm lazily importing a class that was renamed or never written, which ruff cannot see into and `test_registryDrift.py` does not check. Only `unresolved-import`/`unresolved-reference` are errors; every other rule is ignored in `[tool.ty.rules]` with a note. Ratchet those up one at a time — do not bulk-enable.
- No mypy or pre-commit. CI: `tests.yaml`, blocking `lint.yaml`, platform build workflows, and `prune-releases.yml`.
- Dependencies: `requirements.txt` plus `extra-requirements-{windows,linux,macos}[-lite].txt`. Some cp314 wheels (onnxruntime-openvino, rife-ncnn) are direct `NevermindNilas/TAS-Models-Host` `main`-release URLs in those files; plain PyPI alone is insufficient for a full install.

## Non-negotiable conventions

Inviolable:

- No AI co-author trailer or “Generated with Claude” footer in commits or PR bodies.
- ONNX models ship real paired FP16/FP32 files selected by `half`; never relabel FP32 as FP16. CUDA `.pth` models may use runtime `.half()`.
- ONNX/DML returns CPU tensors; CUDA keeps tensors on GPU and may use CUDA graphs/streams. Preserve device placement in shared code.
- Spandrel normalization: compute statistics in FP32, then cast back; already applied to PLKSR/NAFNet/SPAN.

Workflow:

- For optimization, `channels_last`/`torch.compile` already exist; prioritize architectural/kernel gains while preserving weights and output parity (bit-identical where the baseline is deterministic; within the run-to-run noise floor where it is not, e.g. RifeCuda's graph path).
- Benchmark against a snapshot of the original unedited code on a warm, otherwise-idle GPU; report FPS, VRAM, and parity.
- For bugs/reviews, read every relevant caller, discard false positives explicitly, prove changes with before/after tests, and update `CHANGELOG.MD`.
- Never load the full large changelog for orientation; grep the relevant entry.
- `CHANGELOG.MD` entries are short and to the point: one or two sentences per entry — what changed, and the measured number or user-visible symptom. No investigation narrative, no bisect story, no "verified by two reviewers", no restating the fix in three ways. Keep the deep write-up in the PR body or commit message, not here.

## Agent skills

- **Issue tracker:** GitHub issues via the `gh` CLI — see `docs/agents/issue-tracker.md`.
- **Triage labels:** canonical set (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`) — see `docs/agents/triage-labels.md`.
- **Domain docs:** single-context `CONTEXT.md` + `docs/adr/` at repo root, created lazily — see `docs/agents/domain.md`.

### Codebase graph (graphify)

`graphify-out/` holds a graphify knowledge graph of the repo (`graph.json`, `GRAPH_REPORT.md`, interactive `graph.html`). For architecture/relationship questions ("what depends on X", "how do these modules connect"), query it via the graphify skill before grepping — the repo map above covers the what, the graph covers the edges. Check `built_at_commit` in `graph.json` first; if it lags the area in question, say so and offer an incremental rebuild rather than answering from a stale graph.
