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
  --upscale --interpolate --scenechange
```
*Full AI enhancement with smart scene detection*

**🔧 Custom Quality**
```bash
./python.exe ./main.py --input video.mp4 \
  --upscale --upscale_method span-tensorrt \
  --interpolate --interpolate_method rife4.25
```
*High-end processing with latest models*

</td>
</tr>
</table>

### 📊 Feature Difficulty Levels

| Level | Features | Use Case |
|-------|----------|----------|
| 🟢 **Beginner** | `--upscale`, `--interpolate` | Basic enhancement, anime upscaling |
| 🟡 **Intermediate** | `--restore`, `--dedup`, `--scenechange` | Quality optimization, batch processing |
| 🔴 **Advanced** | TensorRT models, custom encoders, model chaining | Professional workflows, maximum performance |

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
- **Python:** 3.13
- **FFmpeg:** Latest version ( automatically installed by TAS )
- **GPU Drivers:** Latest NVIDIA/AMD drivers

</td>
</tr>
</table>

### 🚀 Recommended Hardware by Use Case

| Use Case | GPU | RAM | Notes |
|----------|-----|-----|-------|
| **1080p Anime** | RTX 3060 | 16GB | Good balance of speed/quality |
| **4K Processing** | RTX 4070+ | 32GB | High-end workflows |
| **Batch Processing** | RTX 4080+ | 32GB+ | Maximum throughput |
| **TensorRT Optimization** | RTX Series | 16GB+ | Best performance gains |

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
  --scenechange --scenechange_method maxxvit-tensorrt \
  --encode_method x264_animation_10bit
```
**What it does:** Upscales, smooths motion, detects scene changes, optimizes for anime
</details>

<details>
<summary><b>🎥 Old Anime Restoration</b></summary>

```bash
# Comprehensive restoration
./python.exe ./main.py --input old_anime.mp4 \
  --upscale --upscale_method span-tensorrt \
  --restore --restore_method nafnet \
  --sharpen --sharpen_sens 30 \
  --interpolate --interpolate_method rife4.25 \
  --dedup --dedup_method ssim-cuda
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
  --interpolate --interpolate_method rife4.20-tensorrt \
  --scenechange --scenechange_sens 70
```
**What it does:** Handles rapid scene changes common in gaming
</details>

---

## 📋 Parameter Reference

### 📁 Input & Output Configuration

#### Core I/O Parameters

| Parameter | Type | Default | Description | Example |
|-----------|------|---------|-------------|---------|
| `--input` | str | - | **Input video path(s)** - Supports multiple files separated by `;` | `video.mp4` or `vid1.mp4;vid2.mp4` |
| `--output` | str | auto | **Output path** - Auto-generates if not specified | `enhanced_video.mp4` |

#### Supported Input Formats
```
Video: .mp4, .mkv, .mov, .avi, .webm, .m4v, .flv
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
| `--preview` | flag | **Preview mode** - Opens localhost preview | Low |
| `--ae` | flag | **After Effects mode** - Integration flag | None |

---

### 🤖 AI Enhancement Features

### 🆙 Upscaling Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--upscale` | flag | False | - | **Enable AI upscaling** |
| `--upscale_factor` | int | 2 | 2, 4 | **Upscale multiplier** |
| `--upscale_method` | str | "shufflecugan" | [see table](#upscaling-models) | **AI model selection** |
| `--custom_model` | str | "" | path | **Custom model file** (.pth/.onnx) |

#### 🎯 Upscaling Models

| Model | Speed | Quality | VRAM | Best For | Acceleration |
|-------|-------|---------|------|----------|--------------|
| **shufflecugan** | ⚡⚡⚡ | ⭐⭐⭐ | 4GB | General anime | CUDA, TRT, NCNN |
| **span** | ⚡⚡ | ⭐⭐⭐⭐ | 6GB | High quality anime | CUDA, TRT, DML |
| **compact** | ⚡⚡⚡ | ⭐⭐ | 2GB | Fast processing | CUDA, TRT, DML |
| **open-proteus** | ⚡⚡ | ⭐⭐⭐⭐⭐ | 8GB | Maximum quality | CUDA, TRT, DML |
| **aniscale2** | ⚡⚡ | ⭐⭐⭐⭐ | 6GB | Modern anime | CUDA, TRT, DML |

> **💡 Recommendation:** Use `shufflecugan-tensorrt` for balanced speed/quality, `span-tensorrt` for best quality

### ⏱️ Interpolation Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--interpolate` | flag | False | - | **Enable frame interpolation** |
| `--interpolate_factor` | int | 2 | 2-8 | **Frame multiplication** (2 = 30→60fps) |
| `--interpolate_method` | str | "rife" | [see table](#interpolation-models) | **Interpolation model** |
| `--ensemble` | flag | False | - | **Higher quality** (slower, RIFE only) |
| `--dynamic_scale` | flag | False | - | **Dynamic scaling** (CUDA only) |

#### 🎯 Interpolation Models

| Model | Speed | Quality | VRAM | Best For | Latest Version |
|-------|-------|---------|------|----------|----------------|
| **rife4.22** | ⚡⚡⚡ | ⭐⭐⭐ | 4GB | General use | ✅ Recommended |
| **rife4.25** | ⚡⚡ | ⭐⭐⭐⭐ | 6GB | High quality | ✅ Latest |
| **rife4.22-lite** | ⚡⚡⚡⚡ | ⭐⭐ | 2GB | Low-end GPUs | ✅ |
| **rife_elexor** | ⚡ | ⭐⭐⭐⭐⭐ | 8GB | Maximum quality | Custom mod |

> **💡 Performance Tip:** Use `--ensemble` for final renders, disable for previews

### 🔍 Scene Change Detection

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--scenechange` | flag | False | - | **Enable scene detection** |
| `--scenechange_sens` | float | 50 | 0-100 | **Detection sensitivity** |
| `--scenechange_method` | str | "maxxvit-directml" | maxxvit variants | **Detection model** |

**Sensitivity Guide:**
- **20-40:** Conservative (fewer cuts, may miss quick cuts)
- **50-70:** Balanced (recommended for most content)
- **80-100:** Aggressive (more cuts, may over-segment)

### 🔄 Deduplication

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--dedup` | flag | False | - | **Remove duplicate frames** |
| `--dedup_method` | str | "ssim" | ssim, mse, ssim-cuda, flownets | **Detection algorithm** |
| `--dedup_sens` | float | 50 | 0-100 | **Similarity threshold** |

**Method Comparison:**
- **ssim:** Standard, good balance
- **ssim-cuda:** GPU accelerated SSIM
- **mse:** Fast, less accurate
- **flownets:** Slow, most accurate

### 🔧 Restoration & Enhancement

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `--restore` | flag | False | - | **Enable denoising/restoration** |
| `--restore_method` | str | "scunet" | [see table](#restoration-models) | **Restoration model** |
| `--sharpen` | flag | False | - | **Enable sharpening** |
| `--sharpen_sens` | float | 50 | 0-100 | **Sharpening intensity** |

#### 🎯 Restoration Models

| Model | Speed | Noise Removal | Detail Recovery | Best For |
|-------|-------|---------------|-----------------|----------|
| **scunet** | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐ | General denoising |
| **nafnet** | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | High-quality denoising |
| **real-plksr** | ⚡⚡ | ⭐⭐ | ⭐⭐⭐⭐ | JPEG artifact removal |
| **anime1080fixer** | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Anime line enhancement |

### 🎭 Segmentation & Depth

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--segment` | flag | False | **Background/foreground separation** |
| `--segment_method` | str | "anime" | **Segmentation model** (anime, anime-tensorrt) |
| `--depth` | flag | False | **Generate depth maps** |
| `--depth_method` | str | "small_v2" | **Depth model** [see table](#depth-models) |
| `--depth_quality` | str | "high" | **Processing quality** (low/medium/high) |

### 🔍 Auto Clip Detection

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--autoclip` | flag | False | - | **Automatic clip detection** |
| `--autoclip_sens` | float | 40 | 0-100 | **Detection sensitivity** |

### 📏 Resizing

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--resize` | flag | False | - | **Enable resizing** |
| `--resize_factor` | float | 2.0 | 0.1-10.0 | **Resize multiplier** (>1 = larger, <1 = smaller) |

---

### ⚡ Performance & Hardware Settings

#### Performance Optimization

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `--half` | str | "True" | True/False | **FP16 precision** (faster, less VRAM) |
| `--static` | flag | True | - | **Static engine** (bypass TAS limitations) |
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
--compile_mode max
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
| `--download_requirements` | flag | False | **Download dependencies** (Adobe compatibility) |

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
| `--bit_depth` | str | "8bit" | 8bit/10bit/16bit | **Color bit depth** |

**Bit Depth Impact:**
- **8bit:** Standard, fastest, smallest files
- **10bit:** Better gradients, HDR support
- **16bit:** Maximum quality, largest files

---

## 🎯 Model Selection Guide

### 📊 Performance vs Quality Matrix

<table>
<tr>
<td width="33%">

**🚀 Speed Priority**
```bash
# Fastest processing
--upscale_method compact-tensorrt
--interpolate_method rife4.22-lite-tensorrt
--half True
--compile_mode max
```

</td>
<td width="33%">

**⚖️ Balanced**
```bash
# Good speed + quality
--upscale_method shufflecugan-tensorrt
--interpolate_method rife4.22-tensorrt
--scenechange
--half True
```

</td>
<td width="33%">

**🎨 Quality Priority**
```bash
# Maximum quality
--upscale_method span-tensorrt
--interpolate_method rife4.25
--ensemble
--restore_method nafnet
--half False
```

</td>
</tr>
</table>

### 🎬 Content-Specific Recommendations

| Content Type | Upscaling | Interpolation | Additional |
|--------------|-----------|---------------|------------|
| **TV Anime** | shufflecugan-tensorrt | rife4.22-tensorrt | scenechange |
| **Movie Anime** | span-tensorrt | rife4.25 | ensemble, restore |
| **Gaming** | openproteus-tensorrt | rife4.20-tensorrt | scenechange_sens 70 |
| **Live Action** | open-proteus-tensorrt | rife4.22-tensorrt | restore_method nafnet |
| **Old Content** | span-tensorrt | rife4.25 | restore, sharpen |

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
| **x264_10bit** | ⚡⚡ | ⭐⭐⭐⭐ | Medium | HDR, gradients |
| **x265** | ⚡⚡ | ⭐⭐⭐⭐ | Small | Modern devices |
| **av1** | ⚡ | ⭐⭐⭐⭐⭐ | Smallest | Future-proof |

#### Hardware Encoders

| Encoder | Speed | Quality | Requirements | Notes |
|---------|-------|---------|--------------|-------|
| **nvenc_h265** | ⚡⚡⚡⚡ | ⭐⭐⭐ | NVIDIA GPU | Fast, good quality |
| **qsv_h265** | ⚡⚡⚡⚡ | ⭐⭐⭐ | Intel CPU | Intel Quick Sync |
| **hevc_amf** | ⚡⚡⚡⚡ | ⭐⭐⭐ | AMD GPU | AMD hardware |

#### High-Quality Presets

| Preset | Encoding Time | Quality | File Size | Use Case |
|--------|---------------|---------|-----------|----------|
| **slow_x264** | ⚡ | ⭐⭐⭐⭐⭐ | Large | Archival quality |
| **slow_x265** | ⚡ | ⭐⭐⭐⭐⭐ | Medium | Best compression |
| **slow_av1** | ⚡ | ⭐⭐⭐⭐⭐ | Small | Future standard |

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
| `--preset_list` | flag | False | **Show available presets** |

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
  --scenechange \
  --encode_method x264_animation \
  --preset "fast_anime"
```

**Quality Anime Processing:**
```bash
./python.exe ./main.py --input anime.mp4 \
  --upscale --upscale_method span-tensorrt \
  --interpolate --interpolate_method rife4.25 \
  --ensemble \
  --scenechange \
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
--half True --static --compile_mode max
--upscale_method compact-tensorrt
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
  --scenechange --scenechange_method maxxvit-tensorrt \
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
  --upscale --upscale_method compact-tensorrt \
  --interpolate --interpolate_method rife4.22-lite-tensorrt \
  --scenechange \
  --encode_method nvenc_h265 \
  --half True --compile_mode max
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
  --scenechange --scenechange_sens 70 \
  --dedup --dedup_method ssim-cuda \
  --encode_method nvenc_h264
```
**Processing:** Handles rapid scene changes  
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
  --restore --restore_method nafnet \
  --sharpen --sharpen_sens 40 \
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
  --scenechange \
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
  --depth --depth_method large_v2-tensorrt --depth_quality high \
  --encode_method prores
```
**Processing:** Creates segmentation masks and depth maps  
**Time:** ~4x video length  
**Output:** Video + segmentation + depth data
</details>

### 🎯 Parameter Combinations

**Memory-Constrained Systems:**
```bash
--upscale_method superultracompact-tensorrt \  # Lightweight model  
--interpolate_method rife4.22-lite-tensorrt \  # Lite variant
--half True  # FP16 precision
```

**Maximum Quality (Slow):**
```bash
--upscale_method compact-tensorrt \
--interpolate_method rife4.25 --ensemble \
--restore --restore_method anime1080fixer-tensorrt \
--encode_method slow_x265 \
```

**Balanced Performance:**
```bash
--upscale_method shufflecugan-tensorrt \
--interpolate_method rife4.22-tensorrt \
--scenechange \
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
