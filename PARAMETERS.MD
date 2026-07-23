<div align="center">

# 📋 TheAnimeScripter Parameters Guide

#### _Complete reference for AI video enhancement parameters_

[![Parameters](https://img.shields.io/badge/Parameters-Complete%20Guide-blue?style=flat-square)](PARAMETERS.md)
[![Examples](https://img.shields.io/badge/Examples-Ready%20to%20Use-green?style=flat-square)](#usage-examples)
[![Troubleshooting](https://img.shields.io/badge/Help-FAQ%20Available-orange?style=flat-square)](#troubleshooting--faq)

</div>

---

## 📖 Table of Contents

- [🚀 Quick Start Guide](#-quick-start-guide)
- [⚙️ Prerequisites & System Requirements](#️-prerequisites--system-requirements)
- [🎯 Core Workflow Examples](#-core-workflow-examples)
- [📋 Parameter Reference](#-parameter-reference)
  - [📁 Input & Output Configuration](#-input--output-configuration)
  - [🎬 Video Processing Options](#-video-processing-options)
  - [🤖 AI Enhancement Features](#-ai-enhancement-features)
  - [⚡ Performance & Hardware Settings](#-performance--hardware-settings)
  - [🔧 Advanced Configuration](#-advanced-configuration)
- [🎯 Model Selection Guide](#-model-selection-guide)
- [🎞️ Encoding & Output Options](#️-encoding--output-options)
- [📝 Preset System](#-preset-system)
- [❓ Troubleshooting & FAQ](#-troubleshooting--faq)
- [💡 Complete Usage Examples](#-complete-usage-examples)

---

## 🚀 Quick Start Guide

> **New to TAS?** Start here for the fastest path to enhancing your videos!

### 🎬 Common Workflows

<table>
<tr>
<td width="50%">

**🎯 Basic Anime Upscaling**
```bash
./python.exe ./main.py --input video.mp4 --upscale
```
*Enhances resolution 2x with anime-optimized AI*

**⚡ Smooth Motion Enhancement**
```bash
./python.exe ./main.py --input video.mp4 --interpolate
```
*Creates buttery-smooth 60fps from 30fps*

</td>
<td width="50%">

**🎨 Complete Enhancement**
```bash
./python.exe ./main.py --input video.mp4 \
  --upscale --interpolate
```
*Full AI enhancement with 4K output*

**🔧 Custom Quality**
```bash
./python.exe ./main.py --input video.mp4 \
  --upscale_method span-tensorrt \
  --interpolate_method rife4.25
```
*High-end processing with latest models — auto-enables features when method is specified*

</td>
</tr>
</table>

### 📊 Feature Difficulty Levels

| Level | Features | Use Case |
|-------|----------|----------|
| 🟢 **Beginner** | `--upscale`, `--interpolate` | Basic enhancement, anime upscaling |
| 🟡 **Intermediate** | `--restore`, `--dedup` | Quality optimization, batch processing |
| 🔴 **Advanced** | TensorRT models, custom encoders, model chaining | Professional workflows, maximum performance |

### 🚀 Auto-Enable Feature

TheAnimeScripter now automatically enables processing features when you specify their method! Instead of:
```bash
--upscale --upscale_method span-tensorrt --interpolate --interpolate_method rife4.25
```

You can now simply use:
```bash
--upscale_method span-tensorrt --interpolate_method rife4.25
```

The `--upscale` and `--interpolate` flags are **automatically enabled** when their corresponding `*_method` arguments are provided. This applies to all 7 processing operations:
- `--upscale_method` → auto-enables `--upscale`
- `--interpolate_method` → auto-enables `--interpolate`
- `--dedup_method` → auto-enables `--dedup`
- `--restore_method` → auto-enables `--restore`
- `--segment_method` → auto-enables `--segment`
- `--depth_method` → auto-enables `--depth`
- `--obj_detect_method` → auto-enables `--obj_detect`

---

## ⚙️ Prerequisites & System Requirements

### 🖥️ Minimum Requirements

<table>
<tr>
<td width="50%">

**🔧 Hardware**
- **GPU:** GTX 1060 6GB / RX 580 8GB
- **RAM:** 8GB+ (16GB recommended)
- **Storage:** 2GB free space for models
- **OS:** Windows 10/11

</td>
<td width="50%">

**📦 Software**
- **Python:** 3.13 ( automatically bundled by TAS )
- **FFmpeg:** Latest version ( automatically installed by TAS )
- **GPU Drivers:** Latest NVIDIA/AMD drivers

</td>
</tr>
</table>

### 🚀 Recommended Hardware by Use Case

| Use Case | GPU | RAM | Notes |
|----------|-----|-----|-------|
| **1080p Anime** | RTX 3060 | 16GB | Good balance of speed/quality |
| **TensorRT Optimization** | RTX Series | 16GB+ | Best performance gains |
| **4K & Complex Workflows** | RTX 3090/4090/5090 | 32GB+ | Handles high-res and multi-model |

> **💡 Performance Tip:** TensorRT models offer 2-3x speed improvements but require RTX GPUs

---

## 🎯 Core Workflow Examples

### 🎬 Anime Enhancement Workflows

<details>
<summary><b>📺 TV Anime → Theatrical Quality</b></summary>

```bash
# Standard anime enhancement
./python.exe ./main.py --input anime_episode.mp4 \
  --upscale --upscale_method shufflecugan-tensorrt \
  --interpolate --interpolate_method rife4.22-tensorrt \
  --encode_method x264_animation_10bit
```
**What it does:** Upscales, smooths motion, optimizes for anime
</details>

<details>
<summary><b>🎥 Old Anime Restoration</b></summary>

```bash
# Comprehensive restoration
./python.exe ./main.py --input old_anime.mp4 \
  --upscale --upscale_method shufflecugan-tensorrt \
  --restore --restore_method anime1080fixer-tensorrt autocas \
  --interpolate --interpolate_method rife4.25 \
```
**What it does:** Restores quality, removes noise, enhances details, smooths motion
</details>

### 🎮 Gaming & Live Action

<details>
<summary><b>🎮 Gaming Footage Enhancement</b></summary>

```bash
# Gaming content optimization
./python.exe ./main.py --input gameplay.mp4 \
  --upscale --upscale_method openproteus-tensorrt \
  --interpolate --interpolate_method rife4.25-tensorrt \
  --restore --restore_method autocas \
  --encode_method x264_animation
```
**What it does:** Enhances motion and detail for gaming footage
</details>

---

## 📋 Parameter Reference

### 📁 Input & Output Configuration

#### Core I/O Parameters

| Parameter | Type | Default | Description | Example |
|-----------|------|---------|-------------|---------|
| `--input` | str | - | **Input video path(s)** - Supports multiple files separated by `;` | `video.mp4` or `vid1.mp4;vid2.mp4` |
| `--output` | str | auto | **Output path** - Auto-generates if not specified | `enhanced_video.mp4` |
| `--json` | str | - | **Path to JSON configuration file** (loads options from JSON; when used it must be used alone without additional CLI arguments) | `config.json` |

#### Supported Input Formats
```
Video: .mp4, .mkv, .mov, .avi, .webm, .m4v
Batch: .txt (one file path per line)
URLs: YouTube, direct video links
```

#### Input Examples

<details>
<summary><b>📁 Multiple Files Processing</b></summary>

```bash
# Method 1: Semicolon separated
--input "C:\Videos\anime1.mp4;C:\Videos\anime2.mp4"

# Method 2: Text file batch
--input "C:\batch_list.txt"

# Contents of batch_list.txt:
C:\Videos\episode1.mp4
C:\Videos\episode2.mp4
C:\Videos\episode3.mp4

# Method 3: Directory processing
--input "C:\Videos\"
```
</details>

<details>
<summary><b>🌐 YouTube Processing</b></summary>

```bash
# Direct YouTube URL
--input "https://www.youtube.com/watch?v=VIDEO_ID"

# With quality selection (handled automatically)
./python.exe ./main.py --input "https://youtu.be/VIDEO_ID" --upscale
```
</details>

---

### 🎬 Video Processing Options

#### Time Range Control

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--inpoint` | float | 0 | 0+ | **Start time** in seconds |
| `--outpoint` | float | end | 0+ | **End time** in seconds |

```bash
# Process 30 seconds starting from 1 minute
--inpoint 60 --outpoint 90
```

#### Basic Processing Flags

| Parameter | Type | Description | Performance Impact |
|-----------|------|-------------|-------------------|
| `--benchmark` | flag | **Performance testing** - No output, measures speed | None (testing only) |
| `--preview` | flag | **Preview mode** - Serves a live preview at `http://127.0.0.1:5000/`, at output resolution, for every encoder (including `*_nelux`) | None while the page is closed; ~2 frames/sec while it is open (~3% at 1080p) |
| `--ae` | str | **After Effects host string (e.g., "127.0.0.1:PORT")** - Enables AE comms and sets Adobe mode | None |

---

### 🤖 AI Enhancement Features

### 🆙 Upscaling Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--upscale` | flag | False | - | **Enable AI upscaling** — automatically enabled when `--upscale_method` is specified |
| `--upscale_factor` | int | 2 | 2, 3, 4 | **Upscale multiplier (minimum 2)** |
| `--upscale_method` | str | "shufflecugan" | [see table](#upscaling-models) | **AI model selection** — specifying a non-default method auto-enables upscaling |
| `--custom_model` | str | "" | path | **Custom model file**. Use `.pt`, `.pth`, `.ckpt`, or `.safetensors` with CUDA methods like `shufflecugan`; use `.onnx` with `-directml`, `-openvino`, or `-tensorrt` methods |

> **💡 Tip:** You can now simplify commands by omitting `--upscale` when specifying a method. For example: `--upscale_method span-tensorrt` automatically enables upscaling.

> **⚠️ Backend Match Required:** `--custom_model` must match the selected upscale backend. A custom ONNX model will not load with plain CUDA methods such as `shufflecugan`; use `shufflecugan-directml`, `shufflecugan-openvino`, or `shufflecugan-tensorrt` instead.

#### 🎯 Upscaling Models

| Model | Speed | Quality | VRAM | Best For | Acceleration |
|-------|-------|---------|------|----------|--------------|
| **shufflecugan** | ⚡⚡⚡ | ⭐⭐⭐ | 2GB | General anime | CUDA, TRT, DML, OpenVINO |
| **adore** | ⚡⚡⚡ | ⭐⭐⭐ | 2GB | General anime (shufflecugan-arch, 2x) | CUDA, TRT, DML, OpenVINO, NCNN |
| **fallin_soft** | ⚡⚡⚡ | ⭐⭐⭐⭐ | 2GB | Soft enhancement with denoising | CUDA, TRT, DML, OpenVINO |
| **fallin_strong** | ⚡⚡⚡ | ⭐⭐⭐⭐ | 2GB | Sharper enhancement with denoising | CUDA, TRT, DML, OpenVINO |
| **span** | ⚡⚡ | ⭐⭐⭐ | 2GB | High quality anime | CUDA, TRT, DML, OpenVINO |
| **open-proteus** | ⚡ | ⭐⭐⭐⭐ | 2GB | Maximum quality | CUDA, TRT, DML, OpenVINO |
| **aniscale2** | ⚡ | ⭐⭐⭐⭐ | 2GB | Modern anime | CUDA, TRT, DML, OpenVINO |
| **rtmosr** | ⚡⚡⚡⚡ | ⭐⭐ | 2GB | Real-time focus | TRT, DML, OpenVINO |
| **saryn** | ⚡⚡ | ⭐⭐⭐ | 3GB | Anime-focused, balanced quality | CUDA, TRT, DML, OpenVINO |
| **gauss** | ⚡⚡ | ⭐⭐⭐⭐ | 2GB | DIS-based, quality focus | CUDA, TRT, DML, OpenVINO |
| **animesr** | ⚡ | ⭐⭐⭐⭐⭐ | 4GB | High-quality restoration | CUDA, TRT, DML, OpenVINO |

> **💡 Recommendation:** Use `shufflecugan-tensorrt` for balanced speed/quality, `span-tensorrt` for best quality

### ⏱️ Interpolation Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--interpolate` | flag | False | - | **Enable frame interpolation** — automatically enabled when `--interpolate_method` is specified |
| `--interpolate_factor` | float | 2 | >=1 | **Frame multiplication** (2 = 30→60fps). Accepts fractional values for fractional interpolation; when non-integer the `--static_step` flag will be disabled. |
| `--interpolate_method` | str | "rife4.6" | [see table](#interpolation-models) | **Interpolation model** — specifying a non-default method auto-enables interpolation |
| `--ensemble` | flag | False | - | **Higher quality** (slower, RIFE only) |
| `--dynamic_scale` | flag | False | - | **Dynamic scaling** (CUDA only) |

> **💡 Tip:** You can now simplify commands by omitting `--interpolate` when specifying a method. For example: `--interpolate_method rife4.25` automatically enables interpolation.

#### 🎯 Interpolation Models

| Model | Speed | Quality | VRAM | Best For | Acceleration |
|-------|-------|---------|------|----------|--------------|
| **rife4.22** | ⚡⚡ | ⭐⭐⭐ | 4GB | General use | CUDA, TRT, NCNN |
| **rife4.25** | ⚡⚡ | ⭐⭐⭐⭐ | 6GB | High quality | CUDA, TRT |
| **rife4.25-lite** | ⚡⚡⚡ | ⭐⭐ | 3GB | Balanced performance | CUDA, TRT |
| **rife4.25-heavy** | ⚡ | ⭐⭐⭐⭐ | 8GB | Maximum quality | CUDA, TRT |
| **rife4.22-lite** | ⚡⚡⚡ | ⭐⭐ | 2GB | Low-end GPUs | CUDA, TRT, NCNN |
| **rife4.6** | ⚡⚡⚡⚡ | ⭐⭐ | 4GB | Stable choice | CUDA, TRT, NCNN |
| **rife4.15-lite** | ⚡⚡⚡ | ⭐⭐ | 2GB | Legacy lite | CUDA, TRT, NCNN |
| **rife4.15** | ⚡⚡⚡ | ⭐⭐⭐ | 4GB | Legacy version | CUDA, TRT |
| **rife4.16-lite** | ⚡⚡⚡ | ⭐⭐ | 2GB | Legacy lite | CUDA, NCNN |
| **rife4.17** | ⚡⚡ | ⭐⭐⭐ | 4GB | Legacy version | CUDA, TRT, NCNN |
| **rife4.18** | ⚡⚡ | ⭐⭐⭐ | 4GB | Legacy version | CUDA, TRT, NCNN |
| **rife4.20** | ⚡⚡ | ⭐⭐⭐ | 4GB | Legacy version | CUDA, TRT, NCNN |
| **rife4.21** | ⚡⚡⚡ | ⭐⭐⭐ | 4GB | Legacy version | CUDA, TRT, NCNN |
| **gmfss** | ⚡ | ⭐⭐⭐⭐⭐ | 6GB | Alternative method | CUDA, TRT |
| **rife_elexor** | ⚡ | ⭐⭐⭐ | 8GB | Maximum quality | CUDA, TRT, DML, OpenVINO |
| **distildrba** | ⚡ | ⭐⭐⭐⭐ | 6GB | 3-frame context | CUDA, TRT, DML, OpenVINO |
| **distildrba-lite** | ⚡⚡ | ⭐⭐⭐ | 4GB | 3-frame context, lighter | CUDA, TRT, DML, OpenVINO |

> **💡 Performance Tip:** Use `--ensemble` for final renders, disable for previews

#### Additional Interpolation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--slowmo` | flag | False | **Slow motion mode** - Slows video instead of increasing frame rate |
| `--static_step` | flag | False | **Static timestep** - Force static timestep for RIFE CUDA |
| `--interpolate_first` | bool | True | **Processing order** - Write interpolated frames to queue immediately |

#### ✂️ Scene-Cut-Aware Interpolation

Interpolating across a hard scene cut produces a melt/ghost frame. When enabled, TAS detects the cut and **holds** (duplicates) the frame instead of morphing across it. Opt-in; off by default; output frame count is unchanged.

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--scenechange` | flag | False | - | **Skip interpolation across hard cuts** (requires `--interpolate`) |
| `--scenechange_method` | str | "ssim-cuda" | ssim, ssim-cuda, mse, mse-cuda, maxxvit-tensorrt, maxxvit-directml | **Streaming cut detector** — cheap `ssim`/`mse` (no model) or the `maxxvit` classifier |
| `--scenechange_sens` | float | 50 | 0-100 | **Cut sensitivity** (higher = more cuts detected); mapped to a per-method threshold |

> **Note:** `transnetv2` and `pyscenedetect` are prepass-only (windowed / whole-file) and are not available here — use `--autoclip` for those.

### 🔄 Deduplication

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--dedup` | flag | False | - | **Remove duplicate frames** — automatically enabled when `--dedup_method` is specified |
| `--dedup_method` | str | "ssim" | [see table](#deduplication-methods) | **Detection algorithm** — specifying a non-default method auto-enables deduplication |
| `--dedup_sens` | float | 35 | 0-100 | **Similarity threshold** |


#### 🎯 Deduplication Methods

| Method | Speed | Accuracy | VRAM | Best For | Acceleration |
|--------|-------|----------|------|----------|--------------|
| **ssim** | ⚡⚡⚡ | ⭐⭐⭐ | Low | General use | CPU |
| **ssim-cuda** | ⚡⚡⚡⚡ | ⭐⭐⭐ | <1GB | GPU accelerated | CUDA |
| **mse** | ⚡⚡⚡⚡⚡ | ⭐⭐ | Low | Fast processing | CPU |
| **mse-cuda** | ⚡⚡⚡⚡⚡ | ⭐⭐ | <1GB | Fast GPU processing | CUDA |
| **flownets** | ⚡ | ⭐⭐⭐⭐ | <2GB | Maximum accuracy | CUDA |
| **vmaf** | ⚡ | ⭐⭐⭐⭐ | <2GB | Perceptual duplicate detection | CPU |
| **vmaf-cuda** | ⚡⚡ | ⭐⭐⭐⭐ | <2GB | GPU-accelerated VMAF-based dedup | CUDA |

### 🔧 Restoration & Enhancement

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `--restore` | flag | False | - | **Enable denoising/restoration** — automatically enabled when `--restore_method` is specified |
| `--restore_method` | str | list | ["anime1080fixer"] | **Restoration model(s) — accepts multiple methods for chained processing** — specifying a non-default method auto-enables restoration |
| `--stabilize` | flag | False | - | **Enable conservative video stabilization** (translation/rotation with black border fill) |

#### 🎯 Restoration Models

| Model | Speed | Noise Removal | Detail Recovery | Best For | Acceleration |
|-------|-------|---------------|-----------------|----------|--------------|
| **scunet** | ⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐ | General denoising | CUDA, TRT, DML, OpenVINO |
| **nafnet** | ⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | High-quality denoising | CUDA |
| **dpir** | ⚡ | ⭐⭐⭐ | ⭐⭐⭐ | JPEG denoising | CUDA |
| **real-plksr** | ⚡ | ⭐⭐ | ⭐⭐⭐⭐ | JPEG artifact removal | CUDA |
| **anime1080fixer** | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Anime line enhancement | CUDA, TRT, DML, OpenVINO |
| **fastlinedarken** | ⚡⚡⚡⚡⚡ | ⭐⭐| ⭐⭐ | Line darkening | CUDA, TRT |
| **autocas** | ⚡⚡⚡⚡⚡ | ⭐ | ⭐⭐⭐ | Adaptive sharpening (auto-tuned FidelityFX CAS, no weights) | CUDA, CPU |
| **gater3** | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Advanced denoising | CUDA, DML, OpenVINO |
| **deh264_real** | ⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐ | De-H.264 artifact removal | CUDA, TRT, DML, OpenVINO |
| **deh264_span** | ⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐ | SPAN-based de-H.264 | CUDA, TRT, DML, OpenVINO |
| **hurrdeblur** | ⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐ | Deblurring | CUDA, TRT, DML, OpenVINO |

### 🎭 Segmentation & Depth

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--segment` | flag | False | **Background/foreground separation** — automatically enabled when `--segment_method` is specified |
| `--segment_method` | str | "anime" | **Segmentation model** [see table](#segmentation-methods) — specifying a non-default method auto-enables segmentation |
| `--segment_batch` | int | 1 | **Frames per model forward** for segmentation. `>1` batches independent frames to amortize kernel-launch overhead, at a proportional VRAM cost. ISNet is fully convolutional and per-frame independent, so batching cannot change a frame's mask. `1` matches the previous one-frame path exactly. Gains depend on where the model sits between launch-bound and compute-bound. **CUDA** (`anime`) always runs the model in a fixed 1024-box, so the gain is resolution-independent: RTX 3090, 402 frames, ~**+10%** at `4` (24.2 → 26.7 FPS end-to-end); `8` is usually a regression at 720p+ (peak model VRAM 0.95 → 3.2 GB at `4`, 6.2 GB at `8`). **TensorRT** (`anime-tensorrt`) runs at the native `/64`-padded resolution, so batching pays only at low resolution: at 640×360 `4` gives **+57%** (93.8 → 147.1 FPS), while at 1280×720 it is already compute-bound and gains nothing (38.4 → 38.6). A batch-`N` engine is a separate build cached under a batch-aware filename. Forced to `1` for DirectML/OpenVINO, where batching measured *slower*. |
| `--depth` | flag | False | **Generate depth maps** — automatically enabled when `--depth_method` is specified |
| `--depth_method` | str | "small_v2" | **Depth model** [see table](#depth-models) — specifying a non-default method auto-enables depth estimation |
| `--depth_quality` | str | "low" | **Processing quality** (low/medium/high). Note: higher quality modes are incompatible with TensorRT/DirectML backends and will be automatically downgraded to `low` when those backends are used. |
| `--depth_norm` | flag | False | **Ghost-free depth stabilization.** Image-depth methods robustly normalize the current prediction and estimate only a small global scale/shift. Temporal `video_*` methods preserve the model's shared depth coordinate system with a slowly adapting scene range instead of independently min/maxing every frame. Historical depth pixels are never composited into the current frame, preventing stamped textures, trails, and motion blur. For maximum stability, use `--depth_method video_small_v2 --half --depth_norm`. |
| `--depth_window` | int | 32 | **Temporal attention window** (frames attended) for the video depth methods (`video_small_v2`/`video_large_v2`); one of `32/16/8/4`. `32` = full-quality streaming; `16` gives ~+24% forward throughput (~+14% end-to-end) with no added flicker and SSIM 0.997 / VMAF 98.1 vs the full window; `8`/`4` trade more fidelity for more speed. Frame storage is unchanged (32 retained); only the attended window shrinks. CUDA-only; ignored by other depth methods. |
| `--depth_batch` | int | 1 | **Frames per model forward** for the CUDA image depth methods (v2/distill/`og_*_v2`/v3, plus the `*_v2-tensorrt`/`og_*_v2-tensorrt` image engines). `>1` batches independent frames to amortize kernel-launch overhead — biggest gains at low resolution where the model is launch-bound (e.g. RTX 3090 `small_v2` fp16: CUDA `--depth_quality low` ~+47% at `4`, `medium` ~+21%, `high` negligible; TensorRT ~+11% since it already fuses/graphs away most launch overhead), at a small VRAM cost. `1` matches the previous one-frame path exactly. Output matches one-at-a-time within the fp16/bf16 noise floor. TensorRT batching needs dynamic-batch ONNX (hosted on TAS-Models-Host). Not supported for the temporal `video_*` methods, `distill`/`og_distill` TensorRT (multi-input), or the DirectML/OpenVINO backends (forced to `1`). |

#### 🎯 Segmentation Methods

| Method | Speed | Quality | VRAM | Best For | Acceleration |
|--------|-------|---------|------|----------|--------------|
| **anime** | ⚡⚡⚡ | ⭐⭐⭐⭐ | 3GB | Anime content | CUDA |
| **anime-tensorrt** | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 3GB | Fast anime processing | TensorRT |

#### 🎯 Depth Models

| Model | Speed | Quality | VRAM | Best For | Acceleration |
|-------|-------|---------|------|----------|--------------|
| **small_v2** | ⚡⚡⚡⚡ | ⭐ | 2GB | General use | CUDA, TRT, DML, OpenVINO |
| **base_v2** | ⚡⚡⚡ | ⭐⭐ | 4GB | Better quality | CUDA, TRT, DML, OpenVINO |
| **large_v2** | ⚡⚡ | ⭐⭐⭐ | 6GB | Maximum quality | CUDA, TRT, DML, OpenVINO |
| **giant_v2** | ⚡ | ⭐⭐⭐⭐ | 10GB | Highest quality | CUDA |
| **distill_small_v2** | ⚡⚡⚡⚡ | ⭐⭐⭐ | 2GB | Fast processing | CUDA, TRT, DML, OpenVINO |
| **distill_base_v2** | ⚡⚡⚡ | ⭐⭐⭐⭐ | 3GB | Balanced | CUDA, TRT, DML, OpenVINO |
| **distill_large_v2** | ⚡⚡ | ⭐⭐⭐⭐⭐ | 5GB | High quality | CUDA, TRT, DML, OpenVINO |
| **og_small_v2** | ⚡ | ⭐⭐⭐⭐ | 3GB | Original implementation | CUDA, TRT, OpenVINO |
| **og_base_v2** | ⚡ | ⭐⭐⭐⭐⭐ | 5GB | Original quality | CUDA, TRT, OpenVINO |
| **og_large_v2** | ⚡ | ⭐⭐⭐⭐⭐ | 8GB | Original maximum | CUDA, TRT, OpenVINO |
| **og_large_v3** | ⚡ | ⭐⭐⭐⭐⭐ | 8GB | Original maximum v3 | CUDA |

### 🎯 Object Detection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--obj_detect` | flag | False | **Detect objects in video** — automatically enabled when `--obj_detect_method` is specified |
| `--obj_detect_method` | str | "yolov9_small-directml" | **Object detection model** [see table](#object-detection-methods) — specifying a non-default method auto-enables object detection |

#### 🎯 Object Detection Methods

| Method | Speed | Accuracy | VRAM | Best For | Acceleration |
|--------|-------|----------|------|----------|--------------|
| **yolov9_small-directml** | ⚡⚡⚡ | ⭐⭐⭐ | 2GB | Fast detection | DirectML |
| **yolov9_medium-directml** | ⚡⚡ | ⭐⭐⭐⭐ | 4GB | Balanced performance | DirectML |
| **yolov9_large-directml** | ⚡ | ⭐⭐⭐⭐⭐ | 6GB | Maximum accuracy | DirectML |

### 🌀 Motion Blur

| Parameter | Type | Default | Range/Choices | Description |
|-----------|------|---------|---------------|-------------|
| `--moblur` | flag | False | - | **Add motion blur** via interpolation + frame blending — auto-enabled when `--moblur_method` is set |
| `--moblur_method` | str | "rife4.25" | rife4.6, rife4.25 (each also `-tensorrt` / `-directml`) | **Interpolation model + backend.** rife4.25 sharper, rife4.6 faster. No suffix = CUDA (fastest); `-tensorrt` = NVIDIA, `-directml` = AMD/Intel |
| `--moblur_factor` | int | 8 | 8-16 | **Sample count** — higher = smoother trails, slower |
| `--moblur_shutter_angle` | float | 180 | 0-360 | **Blur amount.** 180 = cinema standard, 360 = max smear, 90 = crisp, 0 = off |
| `--moblur_strength` | str | "gaussian_sym" | gaussian_sym, equal, pyramid, ascending, descending | **Frame-weighting scheme** (gaussian_sym mimics a natural shutter) |
| `--moblur_no_linear_blend` | flag | False | - | Disable gamma-correct blending (not recommended) |

### 🎭 Masking

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--mask` | str | "" | path | **Protect regions** — transparent PNG; opaque dark areas stay sharp and static, transparent areas process normally |

One mask covers both `--moblur` and `--interpolate`. Paint the regions you want left alone — a HUD, hardsubs, a watermark, a logo — as opaque dark pixels on a transparent background. RGB (no alpha) and grayscale images work too: dark = protected. The mask is resized to match the frame if the dimensions differ.

- With `--moblur`, protected pixels are taken from the unblurred source frame, so they stay sharp while everything else smears.
- With `--interpolate`, protected pixels are taken from the source frame preceding each generated frame, so they hold instead of warping. This kills the rippling that RIFE and GMFSS produce on text and UI overlaid on moving footage. Static protected content is pixel-identical to the source; content that does change (a subtitle line swapping) snaps on its own source frame rather than morphing across the gap.

```bash
# Interpolate gameplay to 120fps without smearing the HUD
python main.py --input game.mp4 --output out.mp4 --interpolate --interpolate_factor 2 --mask hud.png

# Motion blur that leaves the hardsubs readable
python main.py --input anime.mp4 --output out.mp4 --moblur --mask subs.png
```

Masking costs one blend per generated frame (~0.97 ms at 1080p, roughly 7% of a rife4.6 frame) and nothing at all when `--mask` is unset. Pixels outside the mask are bit-identical to an unmasked run.

### 🔍 Auto Clip Detection

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--autoclip` | flag | False | - | **Automatic clip detection** |
| `--autoclip_sens` | float | 50 | 0-100 | **Detection sensitivity** |

> **Note:** The autoclip sensitivity is inverted internally (the script uses `100 - value`) to match the underlying algorithm.

### 📏 Resizing & Output Scaling

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--resize` | flag | False | - | **Enable resizing** |
| `--resize_factor` | float | 2.0 | 0.1-10.0 | **Resize multiplier** (>1 = larger, <1 = smaller) |
| `--output_scale` | str | "" | WIDTHxHEIGHT | **Output resolution** (e.g., 2560x1440, 1920x1080) |

#### 🎯 Output Scaling Examples

**Common Resolutions:**
```bash
# 1440p output
--output_scale 2560x1440

# 4K output  
--output_scale 3840x2160

# 1080p output
--output_scale 1920x1080
```

**Combined with AI Enhancement:**
```bash
# Upscale anime to 4K
./python.exe ./main.py --input anime.mp4 \
  --upscale --upscale_method span-tensorrt \
  --output_scale 3840x2160

# Create smooth 1440p output
./python.exe ./main.py --input video.mp4 \
  --interpolate --interpolate_method rife4.22-tensorrt \
  --output_scale 2560x1440
```

> **💡 Note:** `--output_scale` uses bilinear scaling for minimal artifacts and is applied during encoding for optimal performance.

---

### ⚡ Performance & Hardware Settings

#### Performance Optimization

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `--half` | str | True | True/False | **Use half precision for inference (default: True)** |
| `--static` | flag | False | - | **Force Static Mode engine generation for TensorRT** |
| `--compile_mode` | str | "default" | default/max/max-graphs | **PyTorch compilation** (CUDA only) |

#### Compilation Modes Explained

| Mode | Performance | Startup Time | Memory | Compatibility |
|------|-------------|--------------|---------|---------------|
| **default** | Baseline | Fast | Low | High |
| **max** | +20-40% | Slow | Medium | Medium |
| **max-graphs** | +40-60% | Very Slow | High | Low |

> **⚠️ Warning:** Compilation modes are experimental and may cause issues with some models

#### Hardware Acceleration

**TensorRT (NVIDIA RTX):**
```bash
# Maximum performance
--upscale_method shufflecugan-tensorrt \
--interpolate_method rife4.22-tensorrt \
```

**DirectML (AMD/Intel):**
```bash
# Cross-platform acceleration
--upscale_method span-directml \
--depth_method small_v2-directml
```

**NCNN (Mobile/Low-power):**
```bash
# Lightweight processing
--interpolate_method rife4.22-ncnn
```

---

### 🔧 Advanced Configuration

#### Model Management

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--offline` | str | "none" | **Model download mode** |
| `--download_requirements` | optional str | None | **Download dependencies** for `windows-cuda`, `windows-lite`, `linux-cuda`, `linux-lite`, or `macos-mps`. With no value, prompts for CUDA/lite profiles on Windows/Linux and selects the Apple Silicon MPS profile on macOS. |
| `--cleanup` | flag | False | **Remove unused libraries/dependencies** (attempts to uninstall extras based on detected GPU capability) |

**Offline Mode Options:**
```bash
# Download specific models
--offline "rife4.22 shufflecugan-tensorrt"

# Download all models
--offline "all"

# No downloads (default)
--offline "none"
```

#### Bit Depth Configuration

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `--bit_depth` | str | "8bit" | 8bit/16bit | **Color bit depth** |

**Bit Depth Impact:**
- **8bit:** Standard, fastest, smallest files
- **16bit:** Maximum quality, largest files

---

## 🎯 Model Selection Guide

### 📊 Performance vs Quality Matrix

<table>
<tr>
<td width="25%">

**🚀 Speed Priority**
```bash
# Fastest processing
--upscale_method shufflecugan-tensorrt
--interpolate_method rife4.22-lite-tensorrt
--half True
```

</td>
<td width="25%">

**⚖️ Balanced**
```bash
# Good speed + quality
--upscale_method shufflecugan-tensorrt
--interpolate_method rife4.22-tensorrt
--half True
```

</td>
<td width="25%">

**� OpenVINO (AMD/Intel)**
```bash
# Optimized for Intel/AMD
--upscale_method span-openvino
--interpolate_method rife4.6-openvino
--restore_method scunet-openvino
--half True
```

</td>
</tr>
</table>

### 🎬 Content-Specific Recommendations

| Content Type | Upscaling | Interpolation | Additional |
|--------------|-----------|---------------|------------|
| **TV Anime** | shufflecugan-tensorrt | rife4.22-tensorrt | restore_method anime1080fixer |
| **Movie Anime** | span-tensorrt | rife4.25 | ensemble, restore |
| **Gaming** | openproteus-tensorrt | rife4.20-tensorrt | restore_method autocas |
| **Live Action** | open-proteus-tensorrt | rife4.22-tensorrt | restore_method nafnet |
| **Old Content** | span-tensorrt | rife4.25 | restore_method scunet autocas |

---

## 🎞️ Encoding & Output Options

### Core Encoding Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--encode_method` | str | "x264" | **Encoder selection** |
| `--custom_encoder` | str | "" | **Custom FFmpeg parameters** |

### 📊 Encoder Comparison

#### Standard Encoders

| Encoder | Speed | Quality | File Size | Best For |
|---------|-------|---------|-----------|----------|
| **x264** | ⚡⚡⚡ | ⭐⭐⭐ | Medium | General use, compatibility |
| **x264_animation** | ⚡⚡⚡ | ⭐⭐⭐⭐ | Small | Anime content |
| **x264_animation_10bit** | ⚡⚡ | ⭐⭐⭐⭐⭐ | Small | Best anime quality |
| **x264_10bit** | ⚡⚡ | ⭐⭐⭐⭐ | Medium | HDR, gradients |
| **x265** | ⚡⚡ | ⭐⭐⭐⭐ | Small | Modern devices |
| **x265_10bit** | ⚡ | ⭐⭐⭐⭐⭐ | Small | High quality HEVC |
| **av1** | ⚡ | ⭐⭐⭐⭐⭐ | Smallest | Future-proof |

#### Hardware Encoders

| Encoder | Speed | Quality | Requirements | Notes |
|---------|-------|---------|--------------|-------|
| **nvenc_h264** | ⚡⚡⚡⚡⚡ | ⭐⭐ | NVIDIA GPU | Fast H.264 |
| **nvenc_h265** | ⚡⚡⚡⚡ | ⭐⭐⭐ | NVIDIA GPU | Fast, good quality |
| **nvenc_h265_10bit** | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | NVIDIA RTX | 10-bit HEVC |
| **nvenc_av1** | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | NVIDIA RTX 40+ | Latest AV1 |
| **qsv_h264** | ⚡⚡⚡⚡ | ⭐⭐ | Intel CPU | Intel Quick Sync H.264 |
| **qsv_h265** | ⚡⚡⚡⚡ | ⭐⭐⭐ | Intel CPU | Intel Quick Sync HEVC |
| **qsv_h265_10bit** | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Intel CPU | 10-bit Quick Sync |
| **qsv_vp9** | ⚡⚡⚡ | ⭐⭐⭐ | Intel CPU | VP9 hardware |
| **h264_amf** | ⚡⚡⚡⚡ | ⭐⭐ | AMD GPU | AMD H.264 |
| **hevc_amf** | ⚡⚡⚡⚡ | ⭐⭐⭐ | AMD GPU | AMD HEVC |
| **hevc_amf_10bit** | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | AMD GPU | 10-bit AMD HEVC |

#### High-Quality Presets

| Preset | Encoding Time | Quality | File Size | Use Case |
|--------|---------------|---------|-----------|----------|
| **slow_x264** | ⚡ | ⭐⭐⭐⭐⭐ | Large | Archival quality |
| **slow_nvenc_h264** | ⚡⚡ | ⭐⭐⭐⭐ | Medium | Quality NVIDIA H.264 |
| **slow_x265** | ⚡ | ⭐⭐⭐⭐⭐ | Medium | Best compression |
| **slow_nvenc_h265** | ⚡⚡ | ⭐⭐⭐⭐ | Medium | Quality NVIDIA HEVC |
| **slow_av1** | ⚡ | ⭐⭐⭐⭐⭐ | Small | Future standard |
| **slow_nvenc_av1** | ⚡⚡ | ⭐⭐⭐⭐⭐ | Small | Quality NVIDIA AV1 |

#### Special Output Formats

| Format | Speed | Quality | Use Case | Notes |
|--------|-------|---------|----------|-------|
| **prores** | ⚡⚡ | ⭐⭐⭐⭐⭐ | Professional editing | Large files |
| **prores_segment** | ⚡⚡ | ⭐⭐⭐⭐⭐ | Segmentation output | With alpha channel |
| **gif** | ⚡⚡⚡ | ⭐⭐ | Web/social media | Animated GIF |
| **png** | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Image sequence | Lossless frames |
| **vp9** | ⚡⚡ | ⭐⭐⭐⭐ | Web delivery | Open codec |
| **lossless** | ⚡ | ⭐⭐⭐⭐⭐ | Archival | No compression |
| **lossless_nvenc** | ⚡⚡ | ⭐⭐⭐⭐⭐ | Fast lossless | Hardware lossless |

### 🎯 Encoder Recommendations

**For Anime Content:**
```bash
--encode_method x264_animation_10bit  # Best anime encoding
```

**For Speed:**
```bash
--encode_method nvenc_h265  # Hardware acceleration
```

**For File Size:**
```bash
--encode_method x265  # Best compression
```

**For Maximum Quality:**
```bash
--encode_method slow_x265  # Slow but excellent
```

---

## 📝 Preset System

### Understanding Presets

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--preset` | str | - | **Load saved preset** |
| `--list_presets` | flag | False | **Show available presets** |

> Tip: You can also load a configuration from a JSON file using `--json <path>`. When `--json` is used it must be supplied alone (no other CLI flags) — the script prevents mixing JSON with other CLI arguments.

### Creating Presets

```bash
# Run command with desired settings
./python.exe ./main.py --input video.mp4 \
  --upscale --upscale_method span-tensorrt \
  --interpolate --interpolate_method rife4.25 \
  --preset "high_quality_anime"

# Preset is automatically saved and can be reused
./python.exe ./main.py --input video2.mp4 --preset "high_quality_anime"
```

### 📋 Recommended Preset Configurations

<details>
<summary><b>🎬 Anime Enhancement Presets</b></summary>

**Fast Anime Processing:**
```bash
./python.exe ./main.py --input anime.mp4 \
  --upscale --upscale_method shufflecugan-tensorrt \
  --interpolate --interpolate_method rife4.22-lite-tensorrt \
  --encode_method x264_animation \
  --preset "fast_anime"
```

**Quality Anime Processing:**
```bash
./python.exe ./main.py --input anime.mp4 \
  --upscale --upscale_method span-tensorrt \
  --interpolate --interpolate_method rife4.25 \
  --ensemble \
  --restore --restore_method scunet \
  --encode_method x264_animation_10bit \
  --preset "quality_anime"
```
</details>

---

## ❓ Troubleshooting & FAQ

### 🔧 Common Issues

<details>
<summary><b>💾 Out of Memory Errors</b></summary>

**Symptoms:** CUDA/GPU memory errors during processing

**Solutions:**
1. **Reduce resolution:** Use `--resize_factor 0.5` before upscaling
2. **Enable FP16:** Use `--half True` (default)
3. **Switch models:** Try "lite" variants (e.g., `rife4.22-lite`)
4. **Process shorter clips:** Use `--inpoint` and `--outpoint`

**Model Memory Requirements:**
- Lite models: 2-4GB VRAM
- Standard models: 4-8GB VRAM  
- Large models: 8-12GB VRAM
</details>

<details>
<summary><b>🐌 Slow Processing Speed</b></summary>

**Symptoms:** Very slow processing, low GPU utilization

**Solutions:**
1. **Use TensorRT:** Add `-tensorrt` to model names
2. **Enable compilation:** Use `--compile_mode max`
3. **Check GPU drivers:** Update to latest
4. **Verify CUDA:** Ensure CUDA 11.8+ installed
5. **Close other apps:** Free up GPU memory

**Performance Optimization:**
```bash
# Maximum speed configuration
--half True --static
--upscale_method shufflecugan-tensorrt
--interpolate_method rife4.22-lite-tensorrt
```
</details>

<details>
<summary><b>❌ Model Loading Failures</b></summary>

**Symptoms:** "Model not found" or loading errors

**Solutions:**
1. **Download models:** Use `--offline "model_name"`
2. **Check internet:** Ensure stable connection
3. **Clear cache:** Delete model cache and re-download
4. **Check disk space:** Ensure adequate free space (5GB+)

**Model Download:**
```bash
# Download specific models
./python.exe ./main.py --offline "rife4.22 shufflecugan-tensorrt"

# Download all models (large download)
./python.exe ./main.py --offline "all"
```
</details>

<details>
<summary><b>🎨 Quality Issues</b></summary>

**Symptoms:** Artifacts, blurriness, or poor quality

**Solutions:**
1. **Adjust encoding:** Try `--encode_method x264_animation_10bit`
2. **Disable compression:** Use higher CRF values in custom encoder
3. **Try different models:** Test `span-tensorrt` for upscaling
4. **Add restoration:** Use `--restore --restore_method nafnet`
5. **Check input quality:** Ensure source isn't heavily compressed

**Quality Settings:**
```bash
# Maximum quality processing
--upscale_method span-tensorrt
--interpolate_method rife4.25 --ensemble
--restore --restore_method nafnet
--encode_method slow_x265
--half False
```
</details>

### 💡 Performance Tips

**🚀 Speed Optimization:**
- Use TensorRT models on RTX GPUs
- Enable `--compile_mode max` for CUDA
- Process shorter segments for testing
- Use "lite" model variants for low-end hardware

**🎨 Quality Optimization:**
- Disable `--half` for maximum precision
- Use `--ensemble` for interpolation
- Add `--restore` for old or compressed content
- Use 10-bit encoding for better gradients

**💾 Memory Management:**
- Monitor GPU memory usage
- Use `--resize_factor` to reduce memory load
- Process in segments for long videos
- Close unnecessary applications

### 🔍 Debug Information

**Check System Compatibility:**
```bash
./python.exe ./main.py --benchmark  # Test performance without output
```

**Verify Model Installation:**
```bash
./python.exe ./main.py --preset_list  # Shows available models
```

**Monitor Resource Usage:**
- Use Task Manager (Windows) or htop (Linux)
- Check GPU memory with nvidia-smi (NVIDIA)
- Monitor disk space during processing

---

## 💡 Complete Usage Examples

### 🎬 Workflow Examples

<details>
<summary><b>📺 Basic Anime Enhancement</b></summary>

```bash
# Simple 2x upscaling with motion smoothing
./python.exe ./main.py --input "anime_episode.mp4" \
  --upscale \
  --interpolate \
  --output "enhanced_anime.mp4"
```
**Processing:** 1080p → 2160p, 24fps → 48fps  
**Time:** ~2x video length  
**Quality:** Good for most anime content
</details>

<details>
<summary><b>🎨 High-Quality Anime Processing</b></summary>

```bash
# Professional-grade enhancement
./python.exe ./main.py --input "anime_movie.mkv" \
  --upscale --upscale_method span-tensorrt \
  --interpolate --interpolate_method rife4.25 --ensemble \
  --restore --restore_method nafnet \
  --encode_method x264_animation_10bit \
  --bit_depth 10bit
```
**Processing:** Maximum quality with all enhancements  
**Time:** ~5x video length  
**Quality:** Theatrical quality output
</details>

<details>
<summary><b>⚡ Fast Batch Processing</b></summary>

```bash
# Speed-optimized for multiple files
./python.exe ./main.py --input "batch_list.txt" \
  --upscale --upscale_method shufflecugan-tensorrt \
  --interpolate --interpolate_method rife4.22-lite-tensorrt \
  --encode_method nvenc_h265 \
```
**Processing:** Fast hardware encoding  
**Time:** ~1x video length  
**Quality:** Good with maximum speed
</details>

<details>
<summary><b>🎮 Gaming Content Enhancement</b></summary>

```bash
# Optimized for gaming footage
./python.exe ./main.py --input "gameplay.mp4" \
  --upscale --upscale_method openproteus-tensorrt \
  --interpolate --interpolate_method rife4.20-tensorrt \
  --dedup --dedup_method ssim-cuda \
  --encode_method nvenc_h264
```
**Processing:** Optimized for gaming content  
**Time:** ~1.5x video length  
**Quality:** Optimized for gaming content
</details>

<details>
<summary><b>🎞️ Old Content Restoration</b></summary>

```bash
# Comprehensive restoration workflow
./python.exe ./main.py --input "old_video.avi" \
  --upscale --upscale_method span-tensorrt \
  --interpolate --interpolate_method rife4.25 \
  --restore --restore_method nafnet autocas \
  --dedup --dedup_method ssim \
  --encode_method slow_x265 \
  --bit_depth 10bit
```
**Processing:** Maximum restoration and enhancement  
**Time:** ~6x video length  
**Quality:** Restored vintage content
</details>

<details>
<summary><b>🌐 YouTube Processing</b></summary>

```bash
# Direct YouTube enhancement
./python.exe ./main.py --input "https://youtube.com/watch?v=VIDEO_ID" \
  --upscale --upscale_method shufflecugan-tensorrt \
  --interpolate --interpolate_method rife4.22-tensorrt \
  --inpoint 30 --outpoint 180 \
  --encode_method x264_animation
```
**Processing:** Downloads and enhances YouTube content  
**Time:** Download time + 2x clip length  
**Quality:** Good for YouTube source material
</details>

<details>
<summary><b>🎭 Segmentation & Depth</b></summary>

```bash
# Generate segmentation and depth maps
./python.exe ./main.py --input "anime_scene.mp4" \
  --upscale --upscale_method span-tensorrt \
  --segment --segment_method anime-tensorrt \
  --depth --depth_method large_v2-tensorrt --depth_quality low \
  --encode_method prores
```
**Processing:** Creates segmentation masks and depth maps  
**Time:** ~4x video length  
**Output:** Video + segmentation + depth data
</details>

### 🎯 Parameter Combinations

**Memory-Constrained Systems:**
```bash
--upscale_method rtmosr-tensorrt \  # Lightweight model
--interpolate_method rife4.22-lite-tensorrt \  # Lite variant
--half True  # FP16 precision
```

**Maximum Quality (Slow):**
```bash
--upscale_method open-proteus-tensorrt \
--interpolate_method rife4.25 --ensemble \
--restore --restore_method anime1080fixer-tensorrt \
--encode_method slow_x265 \
```

**Balanced Performance:**
```bash
--upscale_method shufflecugan-tensorrt \
--interpolate_method rife4.22-tensorrt \
--encode_method x264_animation_10bit \
```

---

<div align="center">

### 🎯 Ready to enhance your videos?

[![Discord](https://img.shields.io/discord/1041502781808328704?style=for-the-badge&logo=discord&logoColor=white&label=Join%20Discord&color=5865F2)](https://discord.gg/hwGHXga8ck)
[![GitHub](https://img.shields.io/github/stars/NevermindNilas/TheAnimeScripter?style=for-the-badge&logo=github&label=Star%20on%20GitHub&color=yellow)](https://github.com/NevermindNilas/TheAnimeScripter)

</div>

---

> **📚 Need more help?** Join our [Discord community](https://discord.gg/hwGHXga8ck) for real-time support and examples!

*Last updated: January 2025 | TAS Version: Latest*
