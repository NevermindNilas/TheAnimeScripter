# Limbo TensorRT streaming validation — 2026-09-12

Tested on Windows, RTX 3090, PyTorch 2.12.0+cu132 and TensorRT 10.16.1.11. Export format: 2. CUDA baseline: `LimboStreamingCuda` with default compilation and its native BF16 autocast. Candidate: FP16 TensorRT with attention products and sensitive operations retained in FP32.

## Warm processing performance

Five interleaved full-clip runs per backend/configuration, four CPU threads, synchronized CUDA timing. Includes RGB normalization, transfers, inference, 50% overlap alignment, and per-frame disparity percentile mapping. Excludes decoding, encoding, export and engine creation. Both arms receive identical predecoded, OpenCV-resized RGB frames. Raw uses all 78 frames; real720 uses its first 48. Both inputs are 504×280.

Values are median throughput. These short desktop measurements have visible run-to-run variation; raw durations below show the range. They are not kernel-only or end-to-end video FPS.

| Model / clip / window | CUDA FPS | TensorRT FPS | Ratio | Flicker change |
|---|---:|---:|---:|---:|
| v1_raw_w8 | 44.7 | 107.8 | 2.41× | +0.08% |
| v1_real720_w8 | 48.3 | 146.7 | 3.04× | +0.34% |
| v1_raw_w32 | 54.8 | 100.9 | 1.84× | -0.06% |
| v1_real720_w32 | 68.3 | 139.7 | 2.05× | -1.12% |
| v2_raw_w8 | 43.5 | 126.9 | 2.92× | -0.14% |
| v2_real720_w8 | 47.8 | 118.7 | 2.48× | -0.37% |
| v2_raw_w32 | 50.1 | 102.4 | 2.05× | +0.14% |
| v2_real720_w32 | 59.5 | 115.8 | 1.94× | -1.01% |

Flicker is mean absolute display-depth change after DIS optical-flow compensation, using identical source-based visibility/photometric masks and excluding detected cuts. It is a temporal-consistency proxy, not depth accuracy. All configurations meet the predeclared ≥10% throughput improvement, ≤5% flicker regression, and <0.01 display-MAE limits.

## Numerical and boundary checks

| Model / height | View counts passed | Worst relative raw-depth MAE | Worst display MAE |
|---|---:|---:|---:|
| v1_280 | 32/32 | 0.682% | 0.00381 |
| v1_378 | 32/32 | 0.501% | 0.00296 |
| v2_280 | 32/32 | 1.701% | 0.00366 |
| v2_378 | 32/32 | 0.318% | 0.00212 |

Every integer view count 1–32 was compared with unautocast FP32 eager inference at both 504×280 and 504×378. Required global relative MAE <2%, each frame <5%, and display MAE <0.01; every check passed. Relative MAE is mean absolute error divided by mean absolute reference depth. Changing neighboring views while holding the target fixed changed its prediction in both backends.

Repeated black/gray/white frames and abrupt black-to-white transitions pass at counts 1/4/16/32 for both models and shapes. An earlier FP16-attention candidate produced NaNs on black frames in V1 at 378×504, N=32; it was rejected. Format 2 retains attention matrix products in FP32 and all reported performance was rerun on that corrected graph.

Full clips exercise real partial tails (Raw: 6 views for window 8, 30 for window 32). No duplicate padding is used. The separately exported raw FP32 safetensors were checked in ONNX Runtime at counts 1/2/3/8; worst relative MAE was 2.37e-6.

Isolated TensorRT-only process: sampled whole-device peak usage 3036–3614 MiB across both models and shapes at window 32, including desktop usage. This clears the 8 GiB runtime guardrail; export/build memory is excluded. The numerical comparison process also loaded eager reference models, so its larger combined allocation is not used as backend memory.

Additional profile checks passed for both models at 504×280: FP16 windows 4 and 16, and FP32 window 4, each using 1, 3, and full-window view counts. Actual CLI encodes succeeded with exact frame counts: V1/window8 Raw 78/78, V2/window32 Raw 78/78, V2/4:3/window8 11/11, and V1/FP32/window4 short input 3/3.

## Raw timing samples (seconds)

| Configuration | CUDA: five runs | TensorRT: five runs |
|---|---|---|
| v1_raw_w8 | 1.6714, 1.7466, 1.6656, 2.2772, 2.0922 | 0.7238, 0.6988, 0.6900, 0.7327, 0.9237 |
| v1_real720_w8 | 0.9932, 0.8130, 0.8567, 1.1595, 1.0472 | 0.3163, 0.3199, 0.3271, 0.3572, 0.4328 |
| v1_raw_w32 | 1.5506, 1.4226, 1.5870, 1.4056, 1.2976 | 0.7981, 0.8031, 0.7729, 0.7436, 0.6220 |
| v1_real720_w32 | 0.7960, 0.6965, 0.6520, 0.7026, 0.8125 | 0.3384, 0.3505, 0.3435, 0.3251, 0.4798 |
| v2_raw_w8 | 1.7928, 1.4221, 2.0880, 2.1142, 1.7852 | 0.5869, 0.5730, 0.8650, 1.0056, 0.6148 |
| v2_real720_w8 | 0.8713, 0.9754, 1.0037, 1.0365, 1.1207 | 0.3844, 0.3756, 0.5052, 0.4563, 0.4042 |
| v2_raw_w32 | 1.8146, 2.3397, 1.4950, 1.5228, 1.5578 | 0.8786, 1.2096, 0.7301, 0.7617, 0.7342 |
| v2_real720_w32 | 0.8088, 0.7430, 0.7482, 1.5233, 0.8062 | 0.3955, 0.4014, 0.4145, 0.6780, 0.6800 |

## Reproduction and limits

First use exports ONNX from cached safetensors and builds an engine for the chosen window. Both tasks may take minutes; subsequent runs reuse caches. Build measurements used an existing shared TensorRT tactic cache and are not cold-build guarantees. Other GPUs, TensorRT versions, resolutions, larger windows, and video content can perform differently.

Local validation scripts and logs are retained in `output/limbo_trt/`: `acceptance.py`, `acceptance_v2.log`, `acceptance.json`, `flicker.py`, `flicker.json`, `memory.py`, `memory_v2.log`, and `raw_fp32_validation.json`. The input clips and binary engines are not committed.

Input SHA-256:

- `input\Raw.mp4`: `17bf2da8564d229e2c6147e07b8d0ccd2ffbe3604a062d4528dcfb25e28b2db7`
- `input\real720.mp4`: `d923fea9d8b044531c009cdc18f2dd33aeba6b543d3c50f0cb5d46c761daeb46`

Weight SHA-256:

- `weights/limbo/Limbo.safetensors`: `be4a8f22a134c683cef71913fdf981293f2a5e0b6510463c3442da1d1e825b46`
- `weights/limbo_v2/LimboV2.safetensors`: `29ba612baade44c064cc3db51925bfef73818693f762d4ea8cc414ab5da3717d`

Checks: broad pytest run passed 1,076 tests, excluding `test_rifeCudaGraph.py`, `test_rife_directml_uhd.py`, and `test_spandrel_extra_archs.py` due to previously identified slow real-model execution. The final focused suite, including the added precision regression test, passes 331 tests. Ruff passes; changed production files pass ty. Whole-repository ty retains eight unrelated optional-import diagnostics (matplotlib, rife_ncnn, spandrel, and ONNX conversion tools).
