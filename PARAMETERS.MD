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

The `--upscale` and `--interpolate` flags are **automatically enabled** when their corresponding `*_method` arguments are provided. This applies to all 8 processing operations:
- `--upscale_method` → auto-enables `--upscale`
- `--interpolate_method` → auto-enables `--interpolate`
- `--dedup_method` → auto-enables `--dedup`
- `--smooth_dedup_method` → auto-enables `--smooth_dedup` (which in turn enables `--interpolate` and disables `--dedup`)
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
- **GPU:** GTX 1060 6GB / RX 580 8GB / Apple Silicon (M1+)
- **RAM:** 8GB+ (16GB recommended)
- **Storage:** 2GB free space for models
- **OS:** Windows 10/11, Linux, macOS (Apple Silicon)

</td>
<td width="50%">

**📦 Software**
- **Python:** 3.14 ( automatically bundled by TAS )
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
| **macOS** | Apple Silicon (M1+) | 16GB+ | `-mps` methods; unified memory is shared with the system |

> **💡 Performance Tip:** TensorRT models offer 2-3x speed improvements but require RTX GPUs

### 🧩 Backend Availability

| Backend | Suffix | Hardware | Covers |
|---------|--------|----------|--------|
| **CUDA** | *(none)* | NVIDIA | Everything except object detection (ONNX-only) |
| **TensorRT** | `-tensorrt` | NVIDIA RTX | Upscale, interpolate, restore, depth, segment, object detection, autoclip, scene-cut, motion blur |
| **DirectML** | `-directml` | AMD / Intel / NVIDIA (Windows) | Upscale, interpolate, restore, depth, segment, object detection, autoclip, scene-cut, motion blur |
| **OpenVINO** | `-openvino` | Intel CPU/GPU/NPU | Upscale, interpolate, restore, depth, object detection |
| **NCNN** | `-ncnn` | Vulkan / low-power | Upscale (`shufflecugan`, `span`), interpolate (RIFE) |
| **MPS** | `-mps` | Apple Silicon | Upscale, interpolate, restore, depth, motion blur |

> **🍎 macOS note:** `-mps` has no path yet for segmentation, object detection, autoclip, deduplication, stabilization, or scene-cut detection. Those run on their CPU variants (`ssim`, `mse`, `pyscenedetect`) or are unavailable.

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
  --upscale --upscale_method open-proteus-tensorrt \
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
| `--decode_method` | str | **Decoder backend** - `cpu` (default) or `nvdec`. `nvdec` needs an NVIDIA GPU with NVDEC and falls back to `cpu` automatically if decoding fails | `nvdec` frees CPU for encoding; the GPU decoder is capped around 690fps at 1080p |
| `--profile` | flag | **Developer profiling** - Runs under `torch.profiler` and writes a trace for finding GPU/CPU bottlenecks | Large; profiling only |

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
| **shufflecugan** | ⚡⚡⚡ | ⭐⭐⭐ | 2GB | General anime | CUDA, TRT, DML, OpenVINO, NCNN, MPS |
| **adore** | ⚡⚡⚡ | ⭐⭐⭐ | 2GB | General anime (shufflecugan-arch, 2x) | CUDA, TRT, DML, OpenVINO, MPS |
| **fallin_soft** | ⚡⚡⚡ | ⭐⭐⭐⭐ | 2GB | Soft enhancement with denoising | CUDA, TRT, DML, OpenVINO, MPS |
| **fallin_strong** | ⚡⚡⚡ | ⭐⭐⭐⭐ | 2GB | Sharper enhancement with denoising | CUDA, TRT, DML, OpenVINO, MPS |
| **span** | ⚡⚡ | ⭐⭐⭐ | 2GB | High quality anime | CUDA, TRT, DML, OpenVINO, NCNN, MPS |
| **shufflespan** | ⚡⚡ | ⭐⭐⭐ | 2GB | SPAN/shuffle hybrid (10.5M, 2x) | TRT, DML, OpenVINO, MPS |
| **open-proteus** | ⚡ | ⭐⭐⭐⭐ | 2GB | Maximum quality | CUDA, TRT, DML, OpenVINO, MPS |
| **aniscale2** | ⚡ | ⭐⭐⭐⭐ | 2GB | Modern anime | CUDA, TRT, DML, OpenVINO, MPS |
| **rtmosr** | ⚡⚡⚡⚡ | ⭐⭐ | 2GB | Real-time focus | CUDA, TRT, DML, OpenVINO, MPS |
| **saryn** | ⚡⚡ | ⭐⭐⭐ | 3GB | Anime-focused, balanced quality | CUDA, TRT, DML, OpenVINO, MPS |
| **gauss** | ⚡⚡ | ⭐⭐⭐⭐ | 2GB | DIS-based, quality focus | CUDA, TRT, DML, OpenVINO, MPS |
| **smosr** | ⚡⚡ | ⭐⭐⭐ | 2GB | enhancr DA SmoSR v1 (2x) | CUDA, TRT, DML, OpenVINO, MPS |
| **figsr** | ⚡⚡ | ⭐⭐⭐ | 2GB | enhancr DA FigSR (2x) | CUDA, MPS |
| **animesr** | ⚡ | ⭐⭐⭐⭐⭐ | 4GB | High-quality restoration (temporal, uses the next frame) | CUDA, TRT, DML, OpenVINO |
| **artcnn_c4f16** / `_dn` / `_ds` | ⚡⚡⚡⚡ | ⭐⭐ | <2GB | Luma-only 2x, very light | TRT, DML, OpenVINO |
| **artcnn_c4f32** / `_dn` / `_ds` | ⚡⚡⚡ | ⭐⭐⭐ | 2GB | Luma-only 2x, `_dn` denoises, `_ds` de-sharpens | TRT, DML, OpenVINO |
| **artcnn_r8f64** | ⚡⚡ | ⭐⭐⭐⭐ | 2GB | Luma-only 2x, heavier | TRT, DML, OpenVINO |
| **artcnn_r16f96** | ⚡ | ⭐⭐⭐⭐ | 3GB | Luma-only 2x, heaviest | TRT, DML, OpenVINO |
| **maxine-bicubic / -low / -medium / -high / -ultra** | ⚡⚡⚡ | ⭐⭐⭐ | varies | NVIDIA Maxine VSR (RTX only, SDK-provided) | CUDA |
| **maxine-highbitrate_low / _medium / _high / _ultra** | ⚡⚡⚡ | ⭐⭐⭐ | varies | Maxine VSR tuned for high-bitrate sources | CUDA |

> **💡 Recommendation:** Use `shufflecugan-tensorrt` for balanced speed/quality, `span-tensorrt` for best quality. On Apple Silicon use `shufflecugan-mps` or `span-mps`.

> **⚠️ Suffix required:** `shufflespan` and the `artcnn_*` family are ONNX-only — always pass a `-tensorrt`, `-directml`, or `-openvino` suffix (plus `-mps` for `shufflespan`). There is no bare-CUDA choice for them.

### ⏱️ Interpolation Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--interpolate` | flag | False | - | **Enable frame interpolation** — automatically enabled when `--interpolate_method` is specified |
| `--interpolate_factor` | float | 2 | >=1 | **Frame multiplication** (2 = 30→60fps). Accepts fractional values for fractional interpolation; when non-integer the `--static_step` flag will be disabled. |
| `--interpolate_method` | str | "rife4.6" | [see table](#interpolation-models) | **Interpolation model** — specifying a non-default method auto-enables interpolation |
| `--ensemble` | flag | False | - | **Higher quality** (slower, RIFE only) |
| `--dynamic_scale` | flag | False | - | **Dynamic scaling** — re-picks the RIFE pyramid scale per frame pair (0.5 / 1.0 / 2.0) from how much the pair changed. RIFE torch backends only (CUDA, MPS); ignored by `rife_elexor`, `gmfss` and `distildrba` |

> **💡 Tip:** You can now simplify commands by omitting `--interpolate` when specifying a method. For example: `--interpolate_method rife4.25` automatically enables interpolation.

#### 🎯 Interpolation Models

| Model | Speed | Quality | VRAM | Best For | Acceleration |
|-------|-------|---------|------|----------|--------------|
| **rife4.22** | ⚡⚡ | ⭐⭐⭐ | 4GB | General use | CUDA, TRT, DML, OpenVINO, NCNN, MPS |
| **rife4.25** | ⚡⚡ | ⭐⭐⭐⭐ | 6GB | High quality | CUDA, TRT, DML, OpenVINO, MPS |
| **rife4.25-lite** | ⚡⚡⚡ | ⭐⭐ | 3GB | Balanced performance | CUDA, TRT, DML, OpenVINO, MPS |
| **rife4.25-heavy** | ⚡ | ⭐⭐⭐⭐ | 8GB | Maximum quality | CUDA, TRT, DML, OpenVINO, MPS |
| **rife4.22-lite** | ⚡⚡⚡ | ⭐⭐ | 2GB | Low-end GPUs | CUDA, TRT, DML, OpenVINO, NCNN, MPS |
| **rife4.6** | ⚡⚡⚡⚡ | ⭐⭐ | 4GB | Stable choice (default) | CUDA, TRT, DML, OpenVINO, NCNN, MPS |
| **rife** | ⚡⚡ | ⭐⭐⭐ | 4GB | Alias for the current recommended RIFE build | TRT, NCNN, MPS |
| **rife4.15** | ⚡⚡⚡ | ⭐⭐⭐ | 4GB | Legacy version | TRT, DML, OpenVINO |
| **rife4.15-lite** | ⚡⚡⚡ | ⭐⭐ | 2GB | Legacy lite | CUDA, NCNN, MPS |
| **rife4.16-lite** | ⚡⚡⚡ | ⭐⭐ | 2GB | Legacy lite | CUDA, NCNN, MPS |
| **rife4.17** | ⚡⚡ | ⭐⭐⭐ | 4GB | Legacy version | CUDA, TRT, DML, OpenVINO, NCNN, MPS |
| **rife4.18** | ⚡⚡ | ⭐⭐⭐ | 4GB | Legacy version | CUDA, TRT, DML, OpenVINO, NCNN, MPS |
| **rife4.20** | ⚡⚡ | ⭐⭐⭐ | 4GB | Legacy version | CUDA, TRT, DML, OpenVINO, NCNN, MPS |
| **rife4.21** | ⚡⚡⚡ | ⭐⭐⭐ | 4GB | Legacy version | CUDA, TRT, DML, OpenVINO, NCNN, MPS |
| **rife_elexor** | ⚡ | ⭐⭐⭐ | 8GB | Maximum quality | CUDA, TRT, MPS |
| **distildrba** | ⚡ | ⭐⭐⭐⭐ | 6GB | Distilled DRBA; uses the next frame as a motion cue | CUDA, TRT |
| **distildrba-lite** | ⚡⚡ | ⭐⭐⭐ | 4GB | Lighter DRBA variant | CUDA, TRT |
| **gmfss** | ⚡ | ⭐⭐⭐⭐⭐ | 6GB | Alternative method | CUDA |

> **💡 Performance Tip:** Use `--ensemble` for final renders, disable for previews. On Apple Silicon use `rife4.6-mps` or `rife4.25-mps`.

> **⚠️ Suffix required:** `rife` and `rife4.15` have no bare-CUDA choice — pass one of the listed suffixes (e.g. `rife-tensorrt`, `rife4.15-directml`).

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

> **Note:** `--dedup` *removes* frames, so the output is shorter than the input and audio is dropped. To keep the duration and recover the motion instead, use `--smooth_dedup`.

### 🎞️ Smooth Deduplication

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--smooth_dedup` | flag | False | - | **Interpolate across duplicate frames** — implies `--interpolate`, disables `--dedup`, keeps audio |
| `--smooth_dedup_method` | str | "ssim" | [same as dedup](#-deduplication-methods) | **Duplicate detection algorithm** — auto-enables `--smooth_dedup` |
| `--smooth_dedup_sens` | float | 35 | 0-100 | **Similarity threshold** (same scale as `--dedup_sens`) |
| `--smooth_dedup_max_span` | int | 6 | 1+ | **Longest duplicate run to interpolate across**; longer holds are filled with copies |

Anime is animated on twos or threes: a 24 fps master carries 8–12 unique drawings per second, each
held for 2–3 identical frames. Plain interpolation synthesizes frames *between two identical
drawings*, so the output is 48 fps but the motion is still 12 fps. `--smooth_dedup` detects the
duplicates and widens each gap instead, so every held drawing gets real in-between motion while the
duration and audio stay untouched.

```
source, on 2s:    A   A   B   B   C     24 fps, 5 frames
--interpolate:    A A A ᵃᵇ B B B ᵇᶜ C   48 fps, 9 frames — the 'A A' and 'B B'
                                        in-betweens interpolate two identical
                                        drawings, so half the new frames are
                                        copies and the motion is unchanged
--smooth_dedup:   A ¹ ² ³ B ⁴ ⁵ ⁶ C     48 fps, 9 frames — every synthesized
                                        frame is a real in-between
```

```bash
./python.exe ./main.py --input "anime.mkv" \
  --smooth_dedup --smooth_dedup_method ssim-cuda \
  --interpolate_method rife4.25 --interpolate_factor 2
```

Runs longer than `--smooth_dedup_max_span` are filled with copies rather than interpolated: the
frames in such a gap were duplicates anyway, so inference there costs full throughput for no visible
gain. Raising it costs inference time, not memory — a long gap is written out as it is produced
rather than held whole. It cannot be set below 1.

Works alongside `--upscale` (either pipeline order), `--restore`, `--resize`, `--scenechange`,
`--mask` and `--slowmo`. It is ignored by the standalone operations that bypass the frame loop
(`--depth`, `--segment`, `--obj_detect`, `--autoclip`, `--stabilize`, `--moblur`).

### 🔧 Restoration & Enhancement

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `--restore` | flag | False | - | **Enable denoising/restoration** — automatically enabled when `--restore_method` is specified |
| `--restore_method` | str | list | ["anime1080fixer"] | **Restoration model(s) — accepts multiple methods for chained processing** — specifying a non-default method auto-enables restoration |
| `--stabilize` | flag | False | - | **Enable conservative video stabilization** (translation/rotation with black border fill) |

#### 🎯 Restoration Models

| Model | Speed | Noise Removal | Detail Recovery | Best For | Acceleration |
|-------|-------|---------------|-----------------|----------|--------------|
| **scunet** | ⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐ | General denoising | CUDA, TRT, DML, OpenVINO, MPS |
| **nafnet** | ⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | High-quality denoising | CUDA, MPS |
| **dpir** | ⚡ | ⭐⭐⭐ | ⭐⭐⭐ | JPEG denoising | CUDA, MPS |
| **real-plksr** | ⚡ | ⭐⭐ | ⭐⭐⭐⭐ | JPEG artifact removal | CUDA, MPS |
| **anime1080fixer** | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Anime line enhancement (default) | CUDA, TRT, DML, OpenVINO, MPS |
| **fastlinedarken** | ⚡⚡⚡⚡⚡ | ⭐⭐| ⭐⭐ | Line darkening | CUDA, TRT |
| **autocas** | ⚡⚡⚡⚡⚡ | ⭐ | ⭐⭐⭐ | Adaptive sharpening (auto-tuned FidelityFX CAS, no weights) | CUDA, CPU |
| **gater3** | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Advanced denoising | CUDA, DML, OpenVINO, MPS |
| **deh264_real** | ⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐ | De-H.264 artifact removal | CUDA, TRT, DML, OpenVINO, MPS |
| **deh264_span** | ⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐ | SPAN-based de-H.264 | CUDA, TRT, DML, OpenVINO, MPS |
| **hurrdeblur** | ⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐ | Deblurring | CUDA, TRT, DML, OpenVINO, MPS |
| **dehalo** | ⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐ | Halo/ringing removal around lines | CUDA, TRT, DML, OpenVINO, MPS |
| **deepdeband-f** | ⚡ | ⭐⭐⭐ | ⭐⭐⭐ | Debanding on flat gradients | CUDA |
| **linethinner-lite / -medium / -heavy** | ⚡⚡⚡⚡⚡ | ⭐ | ⭐⭐ | Line thinning, three strengths (no weights) | CPU, CUDA (`-cuda`) |
| **maxine-denoise_low / _medium / _high / _ultra** | ⚡⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐ | NVIDIA Maxine denoise (RTX only, SDK-provided) | CUDA |
| **maxine-deblur_low / _medium / _high / _ultra** | ⚡⚡⚡ | ⭐⭐ | ⭐⭐⭐⭐ | NVIDIA Maxine deblur (RTX only, SDK-provided) | CUDA |

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
| **anime-directml** | ⚡⚡ | ⭐⭐⭐⭐ | 3GB | AMD/Intel GPUs | DirectML |

> **Note:** `cartoon` is listed as a choice but is not implemented yet — it raises `NotImplementedError`. There is no MPS segmentation path.

#### 🎯 Depth Models

| Model | Speed | Quality | VRAM | Best For | Acceleration |
|-------|-------|---------|------|----------|--------------|
| **small_v2** | ⚡⚡⚡⚡ | ⭐ | 2GB | General use (default) | CUDA, TRT, DML, OpenVINO, MPS |
| **base_v2** | ⚡⚡⚡ | ⭐⭐ | 4GB | Better quality | CUDA, TRT, DML, OpenVINO, MPS |
| **large_v2** | ⚡⚡ | ⭐⭐⭐ | 6GB | Maximum quality | CUDA, TRT, DML, OpenVINO, MPS |
| **giant_v2** | ⚡ | ⭐⭐⭐⭐ | 10GB | Highest quality | CUDA, MPS |
| **distill_small_v2** | ⚡⚡⚡⚡ | ⭐⭐⭐ | 2GB | Fast processing | CUDA, TRT, DML, OpenVINO, MPS |
| **distill_base_v2** | ⚡⚡⚡ | ⭐⭐⭐⭐ | 3GB | Balanced | CUDA, TRT, DML, OpenVINO, MPS |
| **distill_large_v2** | ⚡⚡ | ⭐⭐⭐⭐⭐ | 5GB | High quality | CUDA, TRT, DML, OpenVINO, MPS |
| **og_small_v2** | ⚡ | ⭐⭐⭐⭐ | 3GB | Original implementation | CUDA, TRT, OpenVINO, MPS |
| **og_base_v2** | ⚡ | ⭐⭐⭐⭐⭐ | 5GB | Original quality | CUDA, TRT, OpenVINO, MPS |
| **og_large_v2** | ⚡ | ⭐⭐⭐⭐⭐ | 8GB | Original maximum | CUDA, TRT, OpenVINO, MPS |
| **og_giant_v2** | ⚡ | ⭐⭐⭐⭐⭐ | 10GB | Original, largest v2 | CUDA, MPS |
| **og_distill_small_v2** | ⚡⚡ | ⭐⭐⭐ | 3GB | Distill-Any-Depth weights, reference implementation | CUDA, TRT, MPS |
| **og_distill_base_v2** | ⚡⚡ | ⭐⭐⭐⭐ | 4GB | Distill-Any-Depth weights, reference implementation | CUDA, TRT, MPS |
| **og_distill_large_v2-tensorrt** | ⚡ | ⭐⭐⭐⭐⭐ | 6GB | Alias of `distill_large_v2-tensorrt`; there is no CUDA/MPS `og_distill_large_v2` — use `distill_large_v2` | TRT only |
| **small_v3** | ⚡⚡⚡⚡ | ⭐⭐ | 2GB | Depth Anything 3, fast | CUDA, MPS |
| **base_v3** | ⚡⚡⚡ | ⭐⭐⭐ | 4GB | Depth Anything 3, balanced | CUDA, MPS |
| **large_v3** | ⚡⚡ | ⭐⭐⭐⭐ | 6GB | DA3MONO-LARGE, relative depth, best v3 quality | CUDA, MPS |
| **og_large_v3** | ⚡⚡ | ⭐⭐⭐⭐ | 6GB | DA3METRIC-LARGE, metric-depth sibling of `large_v3` (same ViT-L size, not a larger model) | CUDA, MPS |
| **video_small_v2** | ⚡⚡ | ⭐⭐⭐⭐ | 4GB | Temporal (VideoDepthAnything), flicker-free | CUDA, TRT |
| **video_large_v2** | ⚡ | ⭐⭐⭐⭐⭐ | 8GB | Temporal, maximum quality | CUDA |
| **og_video_small_v2 / og_video_base_v2 / og_video_large_v2** | ⚡ | ⭐⭐⭐⭐ | 4-8GB | Original temporal implementations | CUDA |

> **🎞️ Temporal vs image:** `video_*` methods attend across frames (see `--depth_window`) and are far more stable over time; the rest predict each frame independently. Use `--depth_norm` to cut flicker on image methods.

### 🎯 Object Detection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--obj_detect` | flag | False | **Detect objects in video** — automatically enabled when `--obj_detect_method` is specified |
| `--obj_detect_method` | str | "yolov9_small-directml" | **Object detection model** [see table](#object-detection-methods) — specifying a non-default method auto-enables object detection |
| `--obj_detect_disable_annotations` | str | False | **Boxes only** — `True` drops the class label and confidence percentage drawn on each box. Takes a value: `--obj_detect_disable_annotations True` |

#### 🎯 Object Detection Methods

| Method | Speed | Accuracy | VRAM | Best For | Acceleration |
|--------|-------|----------|------|----------|--------------|
| **yolov9_small** | ⚡⚡⚡ | ⭐⭐⭐ | 2GB | Fast detection (default: `-directml`) | DML, OpenVINO, TRT |
| **yolov9_medium** | ⚡⚡ | ⭐⭐⭐⭐ | 4GB | Balanced performance | DML, OpenVINO, TRT |
| **yolov9_large** | ⚡ | ⭐⭐⭐⭐⭐ | 6GB | Maximum accuracy | DML, OpenVINO, TRT |

> **Note:** Object detection is ONNX-only — every choice carries a `-directml`, `-openvino`, or `-tensorrt` suffix. There is no plain CUDA or MPS path.

### 🌀 Motion Blur

| Parameter | Type | Default | Range/Choices | Description |
|-----------|------|---------|---------------|-------------|
| `--moblur` | flag | False | - | **Add motion blur** via interpolation + frame blending — auto-enabled when `--moblur_method` is set |
| `--moblur_method` | str | "rife4.25" | rife4.6, rife4.25 (each also `-tensorrt` / `-directml` / `-mps`) | **Interpolation model + backend.** rife4.25 sharper, rife4.6 faster. No suffix = CUDA (fastest); `-tensorrt` = NVIDIA, `-directml` = AMD/Intel, `-mps` = Apple Silicon |
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
| `--autoclip` | flag | False | - | **Automatic clip detection** — auto-enabled when `--autoclip_method` is specified |
| `--autoclip_method` | str | "pyscenedetect" | [see table](#autoclip-methods) | **Detection backend** |
| `--autoclip_sens` | float | 50 | 0-100 | **Detection sensitivity** |

#### 🎯 Autoclip Methods

| Method | Speed | Accuracy | Best For | Acceleration |
|--------|-------|----------|----------|--------------|
| **pyscenedetect** | ⚡⚡⚡ | ⭐⭐⭐ | No model, works everywhere (default) | CPU |
| **transnetv2** | ⚡⚡ | ⭐⭐⭐⭐⭐ | Learned shot-boundary detection | CUDA |
| **maxxvit-tensorrt** | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Fast classifier on NVIDIA | TensorRT |
| **maxxvit-directml** | ⚡⚡ | ⭐⭐⭐⭐ | Same classifier on AMD/Intel | DirectML |

> **Note:** The autoclip sensitivity is inverted internally (the script uses `100 - value`) to match the underlying algorithm. No MPS path — use `pyscenedetect` on macOS.

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

**OpenVINO (Intel CPU/GPU/NPU):**
```bash
--upscale_method span-openvino \
--interpolate_method rife4.6-openvino \
--restore_method scunet-openvino
```

**NCNN (Mobile/Low-power):**
```bash
# Lightweight processing
--interpolate_method rife4.22-ncnn
```

**MPS (Apple Silicon):**
```bash
# macOS, M1 and newer
--upscale_method shufflecugan-mps \
--interpolate_method rife4.25-mps \
--restore_method anime1080fixer-mps \
--depth_method small_v2-mps
```

> Install the macOS profile first: `python main.py --download_requirements macos-mps`. Run `python main.py --list_methods upscale` (or any capability, or no value for all) to print exactly which methods this install supports.

---

### 🔧 Advanced Configuration

#### Model Management

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--download_requirements` | optional str | None | **Download dependencies** for `windows-cuda`, `windows-lite`, `linux-cuda`, `linux-lite`, or `macos-mps`. With no value, prompts for CUDA/lite profiles on Windows/Linux and selects the Apple Silicon MPS profile on macOS. |
| `--list_methods` | optional str | None | **List available methods** for one capability (e.g. `--list_methods upscale`) or, with no value, for every capability |
| `--cleanup` | flag | False | **Remove unused libraries/dependencies** (attempts to uninstall extras based on detected GPU capability) |

Weights are downloaded on demand the first time a method runs, so no separate download step is needed.

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

**🧠 OpenVINO (Intel/AMD)**
```bash
# Optimized for Intel/AMD
--upscale_method span-openvino
--interpolate_method rife4.6-openvino
--restore_method scunet-openvino
--half True
```

</td>
<td width="25%">

**🍎 Apple Silicon**
```bash
# macOS, M1 and newer
--upscale_method shufflecugan-mps
--interpolate_method rife4.25-mps
--restore_method anime1080fixer-mps
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
| **Gaming** | open-proteus-tensorrt | rife4.20-tensorrt | restore_method autocas |
| **Live Action** | open-proteus-tensorrt | rife4.22-tensorrt | restore_method nafnet |
| **Old Content** | span-tensorrt | rife4.25 | restore_method scunet autocas |
| **macOS (Apple Silicon)** | shufflecugan-mps | rife4.25-mps | restore_method anime1080fixer-mps |

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

#### In-Process Encoders (`*_nelux`)

These encode inside TAS instead of piping raw frames to an FFmpeg subprocess, which removes the pipe copy on GPU-resident frames. The encoder settings match the plain encoder of the same name; the colour pipeline does not — the FFmpeg writer converts RGB to YUV through swscale at 16-bit with dithering and tags matrix/primaries/transfer/range, while the in-process encoder hands the encoder RGB directly.

| Encoder | Equivalent | Requirements |
|---------|-----------|--------------|
| **x264_nelux** | x264 | - |
| **x265_nelux** | x265 | - |
| **av1_nelux** | av1 | - |
| **nvenc_h264_nelux** | nvenc_h264 | NVIDIA GPU |
| **nvenc_h265_nelux** | nvenc_h265 | NVIDIA GPU |
| **nvenc_av1_nelux** | nvenc_av1 | NVIDIA RTX 40+ |

> **Note:** the in-process encoder cannot apply `--custom_encoder`, `--bit_depth 16bit` or `--output_scale`, and `--depth` does not use it at all. Ask for any of those and TAS warns once at startup, then encodes with the equivalent plain encoder so the output still matches the request — at that encoder's speed, not the in-process one's.

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
1. **Check internet:** Weights download on demand on first use; ensure a stable connection
2. **Clear cache:** Delete model cache and re-download
3. **Check disk space:** Ensure adequate free space (5GB+)
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
./python.exe ./main.py --list_methods  # Shows every model, grouped by capability
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
  --bit_depth 16bit
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
  --upscale --upscale_method open-proteus-tensorrt \
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
  --bit_depth 16bit
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
