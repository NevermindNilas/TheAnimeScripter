# CLAUDE.md

TheAnimeScripter (TAS) — AI video enhancement toolkit for anime/general video: upscale, frame interpolation, restore/denoise, dedup, depth, segment, object detection, stabilize, autoclip, motion blur. CLI + After Effects bridge (this repo is the Python server; the `.jsx`/ExtendScript UI lives in a separate repo). Multi-backend: CUDA, TensorRT, DirectML, OpenVINO, NCNN, MPS (partial). **Python 3.14** (cp314; migrated from 3.13). Version in `src/version.py`.

## Setup / Run

```powershell
python main.py --input <video> --output <video> [--upscale --interpolate --restore --dedup ...]
```

Args + flag `choices` in `src/cli/parser.py` (`createParser`). Full CLI flag reference: `PARAMETERS.MD`.

## Architecture

Entry: `main.py` → `VideoProcessor` (`main.py:50`). `start()` (`main.py:418`) inits models via `initializeModels(self)` + I/O buffers, then runs read/process/write on a ThreadPoolExecutor. `process()` (`main.py:354`) is the frame loop; per-frame work in `processFrame()` (`main.py:249`).

**Standard pipeline** (per frame, in `processFrame`): decode → dedup → restore → interpolate↔upscale → encode. interpolate-vs-upscale order set by `--interpolate_first` (`self.interpolateFirst`; branches `ifInterpolateFirst`/`ifInterpolateLast` ~`main.py:284`).

**I/O backend = nelux** (native FFmpeg+torch extension, `nelux==0.12.0`): decode via `nelux.VideoReader` inside `BuildBuffer` (`src/io/ffmpegSettings.py:25`); encode chosen by `createWriteBuffer()` factory (`src/io/ffmpegSettings.py:1282`) → `NeluxWriteBuffer` (`*_nelux` encode methods, `:1063`) or legacy FFmpeg-subprocess `WriteBuffer` (`:391`). Audio/subtitle passthrough: `NeluxWriteBuffer._setupPassthrough`. nelux needs FFmpeg DLLs on the search path — `src/cli/parser.py` (~`:1695`, Windows) does `os.add_dll_directory(dirname(cs.FFMPEGPATH))`. (Landmine: the `requirements.txt` nelux pin can pull a torch-ABI-mismatched wheel on a cu132 box — verify `import nelux` works before trusting a fresh install.)

**Specialized ops** (autoclip/depth/segment/obj_detect/stabilize/motion_blur) bypass the standard loop. The bypass decision is in `VideoProcessor._selectProcessingMethod()` (`main.py:198`); each op's standalone driver function lives in `src/initializeModels.py`.

Standard-loop model/backend selection: `initializeModels(self)` (`src/initializeModels.py`) uses `match`/`case` on the suffixed method strings (e.g. `--upscale_method x-tensorrt`) to pick + lazy-import the backend class, returning a tuple of inference callables wired in by flag.

### Capability → file map

| Capability | File | Backends |
|---|---|---|
| Upscale | `src/upscale/` | CUDA, TensorRT, DirectML, OpenVINO, NCNN, MPS |
| Interpolate (RIFE) | `src/interpolate/` | CUDA, TensorRT, DirectML, OpenVINO, NCNN, MPS |
| Restore/denoise | `src/unifiedRestore.py` | CUDA, TensorRT, DirectML, OpenVINO, Maxine, MPS |
| Dedup | `src/dedup/dedup.py` | CUDA, CPU (SSIM/MSE/flownets). Knob `--dedup_sens` raw default **35**, remapped: ssim→`1−s/1000`, flownets→`s/100` |
| Depth | `src/depth/depth.py` | CUDA, TensorRT, DirectML (incl. VideoDepthAnything temporal family) |
| Segment | `src/segment/animeSegment.py` | CUDA, TensorRT, DirectML, OpenVINO |
| Object detect | `src/objectDetection/objectDetection.py` | CUDA, TensorRT, DirectML |
| Autoclip | `src/autoclip/` | CPU (PySceneDetect), TRT, DML, TransNetV2 |
| Stabilize | `src/stabilize/` | SuperPoint (+ ORB/LK fallback) |
| Motion blur | `src/motionBlur.py` | `MotionBlurPipeline` |

Other helpers: `src/scenechange.py` (scene-change detection for interpolate/dedup), `src/ytdlp.py` (URL input download).

### Backend pattern (IMPORTANT)

Each capability = one base CUDA/PyTorch class + one **sibling class per backend**, dashless camelCase suffix: `UniversalPytorch`/`UniversalTensorRT`/`UniversalDirectML`/`UniversalNCNN`/`UniversalPytorchMPS` (upscale); `RifeCuda`/`RifeTensorRT`/`RifeDirectML`/`RifeNCNN`/`RifeMPS` (interpolate). The `-<backend>` dash appears only in the method *string* (stripped at load — backend class does e.g. `.replace("-tensorrt","-onnx")` to resolve weights).

New backend = new sibling class + new `match`/`case` arm in `initializeModels.py` or `src/factories/` + register names in argparse `choices` (`src/cli/parser.py`) and `modelsList()`/`modelsMap()` (`src/model/registry.py`). **NEVER rewrite the CUDA path.** No ABC — convention only; each class implements `__call__(frame, ...)` + `handleModel()`.

Deviations to expect: **OpenVINO is usually a branch inside the DirectML/ORT class, not its own sibling** (only Segment has a real `AnimeSegmentOpenVino`). Model-specific arches break the "one base + siblings" shape (`AnimeSR*`, `ArtCNN*`, `DistilDRBA*`, `NvidiaVSR`, `DepthGuidedRife*`). Base-class suffix is inconsistent (`UniversalPytorch` has no `Cuda`, but `RifeCuda` does).

### Where things live (common tasks)

- **Add a model** = TWO edits: weight mapping in `src/model/registry.py` `modelsMap()` AND the flag `choices` in `src/cli/parser.py`. (`modelsList()` in `src/model/registry.py` is the canonical method-name registry, reused by `--offline.`)
- **Weights**: `modelsMap()` resolves name+scale+dtype → filename; `resolveWeightPath()` → `weights/{model}/...`;
- **Output filename** (`Up`/`Int` suffixing, uniquification, image-seq/URL input): `src/io/inputOutputHandler.py` `generateOutputName`.
- **Encode methods** (x264/nvenc/…): `match` arms in `src/io/encodingSettings.py` `matchEncoder()`, mirrored in argparse `choices`.
- **Custom-model ONNX export**: `tools/onnxConverter.py` (`pthToOnnx`, fp16/slim). TRT ONNX→engine build/cache: `src/model/trtHandler.py`.
- **Global singletons** (`WHEREAMIRUNFROM`, `SYSTEM`, `FFMPEGPATH`, `FFPROBEPATH`, `METADATAPATH`, `ADOBE`, `AUDIO`): `src/constants.py`, set once in `src/cli/startup.py`, read everywhere as `import src.constants as cs`.
- **Runtime deps install**: `src/infra/dependencyHandler.py` picks the per-platform `extra-requirements-*.txt` and pip-installs. FFmpeg auto-download: `src/infra/getFFMPEG.py`. Hardware probe: `src/infra/checkSpecs.py`.
- **AE bridge**: `--ae` starts a Socket.IO server (`src/server/aeComms.py`); `cs.ADOBE` mode. Live frame preview server: `src/server/previewSettings.py`. Preset save/load (`--preset`): `src/server/presetLogic.py`. Video metadata probe: `src/io/getVideoMetadata.py`.
- Logs: `TAS-Log.log` in `cs.WHEREAMIRUNFROM` (configured ~`main.py:676`); colored-print helper `src/infra/logAndPrint.py` (wraps stdlib logging — it does NOT own the file path).

### Supporting code

- **`src/spandrel/` — VENDORED in-repo (no longer a submodule).** Fork of `github.com/TNTwise/spandrel` @ `e747f27` (branch `adding_extra_archs`); provenance in `src/spandrel/NOTICE.md`. Arch code lives under `src/spandrel/libs/spandrel/spandrel/architectures/...` (40+ permissive arches: ESRGAN, SCUNet, SPAN, PLKSR, NAFNet, …). The restrictive `spandrel_extra_arches` package was **removed** (non-permissive / research-only licenses). Wrapped via `src/spandrelCompat.py` (prepends `libs/spandrel` to `sys.path`). Arch edits are now normal in-repo edits (one repo, one commit) — must stay ONNX-exportable + FP16-capable (see perf note below).
- Model arches: `src/rifearches/`, `src/gmfss/`, `src/extraArches/`.
- `src/model/modelOptimizer.py` — torch.fx graph transforms (channels_last lives in the backend classes, not here).

## Test / Build

```powershell
python -m pytest tests/ -q                              # torch/cv2/nelux-dependent tests use importorskip (skip without those deps)
python -m pytest tests/test_encodingSettings.py -q      # single file
python build.py                                         # portable-Python bundle: downloads Python 3.14.5 standalone, pip-installs reqs, copies src → dist-portable/main/ (see BUILD.md; --develop redirects output)
```

No lint/format/type tooling (no ruff/black/mypy/pre-commit). CI in `.github/workflows/`: `tests.yaml` (PR+push, Win py3.14), `Build-All-Platforms.yaml` + `Build-macOS.yaml` (portable bundles, dispatch), `prune-releases.yml` (trims old release assets). Deps: `requirements.txt` + `extra-requirements-{windows,windows-lite,linux,linux-lite,macos}.txt`.

**Python 3.14 deps note**: several wheels have no cp314 build on PyPI — custom builds (tensorrt cp314 fork, onnxruntime-openvino cp314, nelux) are hosted as flat assets on the `NevermindNilas/TAS-Models-Host` GitHub release (tag `main`) and pointed at by `extra-requirements-*.txt`. Don't assume plain PyPI `pip install` resolves the full set.

## Conventions / gotchas

- Backend additions → new sibling class (see Backend pattern), never rewrite the CUDA path.
- Commit messages: **no AI co-author / "Generated with Claude" trailer.**
- Model dtype: ONNX variants ship **paired fp16 + fp32 files** in `modelsMap` (gated on `half`) — never fake fp16 ONNX from fp32. (`.pth`/CUDA path ships one file + runtime `.half()`.)
- Spandrel arch perf work must stay ONNX-exportable + FP16-capable. FP16 normalization kernels are slow/lossy → compute norm *stats* in FP32, cast back. (Already done for PLKSR/NAFNet/SPAN.)
- ONNX/DirectML backends return CPU tensors; CUDA backends keep tensors on GPU (CUDA graphs/streams). Mind device placement in shared pipeline code.
- `CHANGELOG.MD` ~82K chars — never read whole for orientation; grep the specific entry.

## Project conventions

- Performance: `channels_last`/`torch.compile` are already used and fine, but do NOT offer them as the *primary* answer to an optimization request — pursue real architectural/kernel gains with preserved weight compat + verified output parity.
- Benchmarking: baseline = the ORIGINAL unedited code (snapshot it; never compare an edited path vs an already-edited baseline). Warm GPU, no concurrent GPU jobs, report FPS + VRAM + parity.
- Bugs/review: verify each claim by reading the code before fixing, drop false positives explicitly, prove with before/after tests, update CHANGELOG.
