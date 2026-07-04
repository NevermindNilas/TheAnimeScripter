# TheAnimeScripter

TAS is a Python 3.14 (cp314) video-enhancement CLI and After Effects server for upscale, interpolation, restore/denoise, dedup, depth, segmentation, object detection, stabilization, autoclip, and motion blur. Backends: CUDA, TensorRT, DirectML, OpenVINO, NCNN, and partial MPS. The `.jsx` UI is in another repo. Version: `src/version.py`.

## Commands

```powershell
python main.py --input <video> --output <video> [--upscale --interpolate --restore --dedup ...]
python -m pytest tests/ -q
python -m pytest tests/test_encodingSettings.py -q
python build.py
```

- CLI definitions/choices: `src/cli/parser.py`; validation: `src/cli/validator.py`; full reference: `PARAMETERS.MD`.
- Tests needing torch/cv2/nelux use `importorskip`.
- `build.py` downloads portable Python 3.14.5, installs dependencies, and copies `src` to `dist-portable/main/`; see `BUILD.md` (`--develop` redirects output).

## Runtime map

- Entry: `main.py` → `VideoProcessor`. `start()` initializes models via `src/initializeModels.py`, creates I/O buffers, then runs read/process/write with `ThreadPoolExecutor`.
- Standard frame path in `processFrame()`: decode → dedup → restore → interpolate↔upscale → encode. `--interpolate_first` selects the order.
- Specialized autoclip/depth/segment/obj_detect/stabilize/motion_blur operations bypass that loop through `_selectProcessingMethod()`; their drivers are in `src/initializeModels.py`.
- `initializeModels()` matches backend-suffixed method strings, lazy-imports the backend, and returns inference callables.
- I/O: `src/io/ffmpegSettings.py`. `BuildBuffer` uses `nelux.VideoReader`; `createWriteBuffer()` selects `NeluxWriteBuffer` or legacy FFmpeg-subprocess `WriteBuffer`. Nelux handles audio/subtitle passthrough in `_setupPassthrough` and needs FFmpeg DLLs registered by `src/infra/getFFMPEG.py`.
- `requirements.txt` pins `nelux==0.12.7`; on cu132, verify `import nelux` because a wheel can be torch-ABI-incompatible.

| Capability | Implementation | Backends/details |
|---|---|---|
| Upscale | `src/upscale/` | CUDA, TRT, DML, OpenVINO, NCNN, MPS |
| Interpolate | `src/interpolate/` | RIFE; CUDA, TRT, DML, OpenVINO, NCNN, MPS |
| Restore | `src/unifiedRestore.py` | CUDA, TRT, DML, OpenVINO, Maxine, MPS |
| Dedup | `src/dedup/dedup.py` | CUDA/CPU SSIM, MSE, flownets; `--dedup_sens` default 35 maps to SSIM `1-s/1000`, flownets `s/100` |
| Depth | `src/depth/depth.py`, `src/depth/backends/` | CUDA, TRT, DML; includes temporal VideoDepthAnything |
| Segment | `src/segment/animeSegment.py` | CUDA, TRT, DML, OpenVINO |
| Object detect | `src/objectDetection/objectDetection.py` | CUDA, TRT, DML |
| Autoclip | `src/autoclip/` | PySceneDetect CPU, TRT, DML, TransNetV2 |
| Stabilize | `src/stabilize/` | SuperPoint with ORB/LK fallback |
| Motion blur | `src/motionBlur.py` | `MotionBlurPipeline` |

URL input is handled by `src/ytdlp.py`.

## Change map

- **Add a model = TWO edits:** add its weight mapping to `src/model/registry.py:modelsMap()` and its CLI choice to `src/cli/parser.py`. `modelsList()` is the canonical method registry used by `--offline`.
- **Add a backend:** add a sibling backend class, a `match` arm in `src/initializeModels.py`, CLI choices, and `modelsList()`/`modelsMap()` entries. Never rewrite the CUDA path.
- Backend classes follow convention, not an ABC: dashless class suffixes (`UniversalPytorch`/`UniversalTensorRT`/`UniversalDirectML`/`UniversalNCNN`/`UniversalPytorchMPS`; `RifeCuda`/`RifeTensorRT`/etc.) implementing `__call__()` and `handleModel()`. Method strings alone use `-<backend>`.
- OpenVINO is normally handled inside the DirectML/ORT class; only Segment has `AnimeSegmentOpenVino`. Model-specific exceptions include `AnimeSR*`, `ArtCNN*`, `DistilDRBA*`, `NvidiaVSR`, and `DepthGuidedRife*`.
- Weights: `modelsMap()` resolves name/scale/dtype; `resolveWeightPath()` uses `weights/{model}/`. TRT/DML/OpenVINO use `weights/{model}-onnx/`, except RIFE keeps its base folder. Downloads come from TAS-Models-Host.
- Output naming: `src/io/inputOutputHandler.py:generateOutputName()`; encoders: `src/io/encodingSettings.py:matchEncoder()` plus mirrored CLI choices.
- ONNX export: `tools/onnxConverter.py`; TensorRT engine build/cache: `src/model/trtHandler.py`.
- Global runtime state: `src/constants.py` (`WHEREAMIRUNFROM`, `SYSTEM`, `FFMPEGPATH`, `FFPROBEPATH`, `METADATAPATH`, `ADOBE`, `AUDIO`), initialized by CLI startup/validation and imported as `cs`.
- Dependencies/FFmpeg/hardware: `src/infra/{dependencyHandler,getFFMPEG,checkSpecs}.py`.
- AE bridge/preview/presets: `src/server/{aeComms,previewSettings,presetLogic}.py`; metadata: `src/io/getVideoMetadata.py`.
- Logs: `main()` sets `cs.LOG_PATH` to per-process `TAS-Log-<pid>.log`; `src/infra/logAndPrint.py` wraps stdlib logging but does not choose the path.

Legacy `src/utils/` compatibility shims are gone; import the canonical modules above.

## Vendored/model code boundary

Do not read, search, summarize, lint, or modify these trees unless the task explicitly targets that model or a traceback enters it:

- `src/spandrel/`, `src/gmfss/`, `src/rifearches/`
- `src/depth/{distillanydepth,depth_anything_3,video_depth_anything,dinov2_layers}/`

`src/spandrel/` is the in-repo fork of TNTwise/spandrel at `e747f27` (`adding_extra_archs`), not a submodule; provenance is in `src/spandrel/NOTICE.md`. It is exposed through `src/spandrelCompat.py`. Restrictive `spandrel_extra_arches` code was removed. Keep architecture changes ONNX-exportable and FP16-capable. Other model code lives in `src/rifearches/`, `src/gmfss/`, and `src/extraArches/`; torch.fx optimization lives in `src/model/modelOptimizer.py`.

## Quality and build

- Ruff config: `pyproject.toml`; CI pins 0.15.16. Before committing, run `ruff check --fix` and `ruff format`.
- Ruff rules: `E,F,I,UP,B`; formatter owns line length. Vendored exclusions are declared in `pyproject.toml`; keep them diff-clean against upstream.
- No mypy or pre-commit. CI: `tests.yaml`, blocking `lint.yaml`, platform build workflows, and `prune-releases.yml`.
- Dependencies: `requirements.txt` and `extra-requirements-{windows,windows-lite,linux,linux-lite,macos}.txt`.
- Several cp314 wheels (TensorRT fork, ONNX Runtime OpenVINO, nelux) come from the `NevermindNilas/TAS-Models-Host` `main` release; plain PyPI is insufficient for a full install.

## Non-negotiable conventions

- No AI co-author or “Generated with Claude” commit trailer.
- ONNX models ship real paired FP16/FP32 files selected by `half`; never relabel FP32 as FP16. CUDA `.pth` models may use runtime `.half()`.
- ONNX/DML returns CPU tensors; CUDA keeps tensors on GPU and may use CUDA graphs/streams. Preserve device placement in shared code.
- Spandrel normalization: compute statistics in FP32, then cast back; already applied to PLKSR/NAFNet/SPAN.
- For optimization, `channels_last`/`torch.compile` already exist; prioritize architectural/kernel gains while preserving weights and output parity.
- Benchmark against a snapshot of the original unedited code on a warm, otherwise-idle GPU; report FPS, VRAM, and parity.
- For bugs/reviews, read every relevant caller, discard false positives explicitly, prove changes with before/after tests, and update `CHANGELOG.MD`.
- Never load the full large changelog for orientation; grep the relevant entry.
