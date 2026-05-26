# CLAUDE.md

TheAnimeScripter (TAS) — AI video enhancement toolkit for anime/general video: upscale, frame interpolation, restore/denoise, dedup, depth, segment, object detection, stabilize, autoclip. CLI + After Effects bridge (this repo is the Python server; the `.jsx`/ExtendScript UI lives in a separate repo). Multi-backend: CUDA, TensorRT, DirectML, OpenVINO, NCNN. Python 3.13.

## Run

```powershell
python main.py --input <video> --output <video> [--upscale --interpolate --restore --dedup ...]
```

Args + flag `choices` in `src/utils/argumentsChecker.py` (`createParser`). Full CLI flag reference: `PARAMETERS.MD`.

## Architecture

Entry: `main.py` → `VideoProcessor` (`main.py:50`). `start()` inits models + I/O buffers; `process()` is the frame loop. Model selection lives in `src/initializeModels.py` (returns the inference callables wired into the pipeline based on flags).

Pipeline: decode (`src/utils/ffmpegSettings.py` `BuildBuffer`) → dedup → restore → interpolate↔upscale (order via `interpolateFirst`) → encode (`WriteBuffer`). Specialized ops (depth/segment/obj_detect/stabilize/motion_blur/autoclip) bypass the standard loop via `initializeModels`.

### Capability → file map

| Capability | File | Backends |
|---|---|---|
| Upscale | `src/unifiedUpscale.py` | CUDA, TensorRT, DirectML, OpenVINO, NCNN |
| Interpolate (RIFE) | `src/unifiedInterpolate.py` | CUDA, TensorRT, DirectML, OpenVINO, NCNN |
| Restore/denoise | `src/unifiedRestore.py` | CUDA, TensorRT, DirectML, OpenVINO, Maxine |
| Dedup | `src/dedup/dedup.py` | CUDA, CPU (metric ops; knob `dedupSens`, default 0.9) |
| Depth | `src/depth/depth.py` | CUDA, TensorRT, DirectML |
| Segment | `src/segment/animeSegment.py` | CUDA, TensorRT, DirectML, OpenVINO |
| Object detect | `src/objectDetection/objectDetection.py` | CUDA, TensorRT, DirectML |
| Autoclip | `src/autoclip/` | CPU (PySceneDetect), TRT, DML, TransNetV2 |
| Stabilize | `src/stabilize/` | SuperPoint |

### Backend pattern (IMPORTANT)

Each capability = one base CUDA/PyTorch class + one **sibling class per backend**. Class names are dashless camelCase suffixes: `UniversalPytorch`, `UniversalTensorRT`, `UniversalDirectML`, `UniversalNCNN` (the `-<backend>` dash only appears in the method *string* like `--upscale_method ...-directml`, stripped at load). New backend = new sibling class, never rewrite the CUDA path. Each implements `__call__(frame, ...)` + `handleModel()`. No ABC — convention only.

### Where things live (common tasks)

- **Add a model** = TWO edits: weight mapping in `src/utils/downloadModels.py` `modelsMap()` AND the flag `choices` in `argumentsChecker.py`.
- **Weights**: `modelsMap()` resolves name+scale+dtype → filename; `resolveWeightPath()` → `weights/{model}/...`; auto-downloads from TAS-Models-Host. `-tensorrt`/`-directml`/`-openvino` pull ONNX into `weights/{model}-onnx/`.
- **Output filename** (the `Up`/`Int` suffixing, uniquification, image-seq/URL input): `src/utils/inputOutputHandler.py` `generateOutputName`. (`ffmpegSettings.py` only consumes the path.)
- **Encode methods** (x264/nvenc/…): `match` arms in `src/utils/encodingSettings.py` `matchEncoder()` + mirrored in argparse `choices`.
- **Global singletons** (`FFMPEGPATH`, `WHEREAMIRUNFROM`, `ADOBE`, paths): `src/constants.py`, set once in argumentsChecker, read everywhere as `import src.constants as cs`.
- **Runtime deps install**: `src/utils/dependencyHandler.py` picks per-platform `extra-requirements-*.txt` and pip-installs. FFmpeg auto-download: `src/utils/getFFMPEG.py`.
- **AE bridge**: `--ae` starts a Socket.IO server (`src/utils/aeComms.py`); `cs.ADOBE` mode.
- **Logs**: `TAS-Log.log` in `cs.WHEREAMIRUNFROM`; helper `src/utils/logAndPrint.py`.

### Supporting code

- `src/spandrel/` — vendored Spandrel (PyTorch model auto-detect/loader, 40+ arches: ESRGAN, SCUNet, SPAN, PLKSR, NAFNet, …). Wrapped via `src/spandrelCompat.py`. Treat as third-party.
- `src/rifearches/`, `src/gmfss/`, `src/extraArches/`, `src/atr/` — model architectures.
- `src/utils/trtHandler.py` — TensorRT ONNX→engine build/cache. `src/utils/modelOptimizer.py` — torch.fx graph transforms (channels_last lives in the backend classes, not here).

## Test / Build

```powershell
python -m pytest tests/ -q                              # all; torch/cv2 tests importorskip (skip without GPU deps)
python -m pytest tests/test_encodingSettings.py -q      # single file
python build.py                                         # portable-Python bundle (downloads embeddable/standalone Python, pip-installs reqs, copies src) → dist-portable/main/ (see BUILD.md)
```

No lint/format/type tooling (no ruff/black/mypy/pre-commit). CI in `.github/workflows/`: `tests.yaml` (PR/push, Win py3.13), build + prune workflows. Version: `src/version.py`. Deps: `requirements.txt` + `extra-requirements-{windows,windows-lite,linux,linux-lite,macos}.txt`.

## Conventions / gotchas

- Backend additions → new sibling class (see Backend pattern), never rewrite the CUDA path.
- Commit messages: **no AI co-author / "Generated with Claude" trailer.**
- Model dtype: ONNX variants ship **paired fp16 + fp32 files** in `modelsMap` (gated on `half`) — never fake fp16 ONNX from fp32. (`.pth`/CUDA path ships one file + runtime `.half()`.)
- Spandrel arch perf work must stay ONNX-exportable + FP16-capable. FP16 normalization kernels are slow/lossy → compute norm *stats* in FP32, cast back.
- ONNX/DirectML backends return CPU tensors; CUDA backends keep tensors on GPU (CUDA graphs/streams). Mind device placement in shared pipeline code.
- `CHANGELOG.MD` ~82K chars — never read whole for orientation; grep the specific entry.

## Project conventions

- Performance: `channels_last`/`torch.compile` are already used and fine, but do NOT offer them as the *primary* answer to an optimization request — pursue real architectural/kernel gains with preserved weight compat + verified output parity.
- Benchmarking: baseline = the ORIGINAL unedited code (snapshot it; never compare an edited path vs an already-edited baseline). Warm GPU, no concurrent GPU jobs, report FPS + VRAM + parity.
- Bugs/review: verify each claim by reading the code before fixing, drop false positives explicitly, prove with before/after tests, update CHANGELOG.
