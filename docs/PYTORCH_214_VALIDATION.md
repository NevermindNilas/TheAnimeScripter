# PyTorch 2.14 upgrade validation

Validated on Windows CUDA, 2026-09-14. Runtime pins now use PyTorch 2.14.0, torchvision 0.29.0, Nelux 0.19.0, and Triton Windows 3.8.0.post28. Windows retains CUDA 13.2; Linux retains CUDA 13.0 with a matching torchvision wheel. Linux lite uses the unsuffixed PyPI pair.

Nelux 0.19.0 publishes `214torch` wheels for Python 3.14 on Windows, Linux, and Apple Silicon. Its import reports `__torch_abi__ == "2.14"`. The Nelux pin is present in both core requirements and every profile: the runtime profile installer installs only the profile file, so a core-only bump could leave the incompatible 0.18 wheel installed. A six-profile regression test enforces this agreement.

## Test environment and scope

Windows, NVIDIA RTX 3090, Python 3.14.5. Separate scratch environments compared torch 2.13.0+cu132 / torchvision 0.28.0+cu132 / Nelux 0.18.0 / Triton 3.7.1 against torch 2.14.0+cu132 / torchvision 0.29.0+cu132 / Nelux 0.19.0 / Triton 3.8.0. Remaining dependencies were reused from the existing development environment; these were not clean full-profile installations. The normal development environment was not upgraded.

- The pre-existing suite passed on the new stack: 1,092 non-preview tests and 20 preview tests. Preview tests ran in a separate process because their HTTP handler teardown severely slows subsequent tests when combined; this also affected the 2.13 baseline.
- Six additional regression cases check that each profile upgrades Nelux alongside torch.
- Real x265 encode/decode tests passed for 8-bit and 16-bit inputs.
- Four TAS pipeline smoke runs passed: CPU/NVDEC decode, each with 8-bit/16-bit output, through `BuildBuffer`, RIFE 4.25, ShuffleCUGAN 2x, and `NeluxWriteBuffer`. Each decoded exactly frames [24, 36), emitted 23 frames after 2x interpolation, and produced a decodable 640x384 file. Output formats were yuv420p and yuv444p10le. No OpenCV decode fallback was allowed.
- CUDA torchvision NMS and a full-graph `torch.compile` elementwise kernel passed. The compiled kernel matched its saved 2.13 outputs exactly.
- The full Windows requirements set resolved successfully. Matching torch/torchvision pairs also resolved for Windows PyPI, Linux CUDA/PyPI, and macOS ARM64 with a macOS 14 deployment target. Execution on Linux, MPS, DirectML, and OpenVINO was not validated here.
- Ruff lint/format checks passed. Type checking still reports eight pre-existing external dependency/import-resolution diagnostics; none match the CI gate for undefined first-party names.

## Cross-version numerical comparisons

Model parameters and RIFE's cached warp grids were identical across versions. An eager convolution trace found the first differences in convolution outputs on identical inputs, followed by amplification through later layers and warps. The CUDA wheels changed cuDNN from 9.20 to 9.24. This is consistent with numerical differences between convolution implementations; no model weights, precision defaults, or CUDA graph behavior were changed to force bitwise equality.

The original synthetic translated-noise stress test remains a limitation: RIFE 4.6 measured 30.88 dB PSNR against 2.13 and RIFE 4.25 measured 27.55 dB. Those results are not a representative video-quality measurement, and strict elementwise cross-version equality is not established.

Two local video samples (`input/720.mp4` and `input/real720.mp4`) were decoded once with the baseline and saved as identical input tensors for both versions: 12 frames per sample, resized to 320x192, producing 11 interpolated frames per model. Results on FP16 RIFE and odd-sized 2x upscaling inputs:

| Comparison | Sample 1 PSNR | Sample 2 PSNR |
|---|---:|---:|
| RIFE 4.6 | 86.53 dB | 87.13 dB |
| RIFE 4.25 | 92.83 dB | 93.51 dB |
| ShuffleCUGAN FP32 | 80.19 dB | 80.38 dB |
| ShuffleCUGAN FP16 | 72.60 dB | 72.25 dB |

RIFE mean absolute differences were below 0.0000025 on normalized [0, 1] pixels; isolated maximum differences reached 0.01221. This supports close output agreement on the tested clips, not bit-identical output or an exhaustive quality/performance guarantee for every model.

## Local evidence

Ignored `scratch_out/` contains the environments, input/output tensors, and scripts:

- `torch214-final-tests.log`, `torch214-preview.log`: regression results.
- `torch214-pipeline.log`, `torch214_pipeline.py`: four decode/process/encode checks.
- `torch214-720.log`, `torch214-real720.log`, `torch_parity_video.py`: video comparisons.
- `torch-trace-comparison.log`, `torch_trace.py`: layer-level comparison.
- `torch214-resolution.log`: full Windows dependency resolution.

## Upstream references

- [PyTorch 2.14 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.14.0)
- [PyTorch/torchvision release pairing](https://dev-discuss.pytorch.org/t/pytorch-2-14-0-general-availability/3431)
- [Triton Windows compatibility matrix](https://github.com/triton-lang/triton-windows#3-pytorch)
- [Nelux 0.19.0 release and ABI requirements](https://github.com/NevermindNilas/Nelux/releases/tag/v0.19.0)
