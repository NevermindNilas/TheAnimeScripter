<div align="center">

# 🎬 The Anime Scripter (TAS)

#### _High-performance AI video enhancement toolkit for creators_

[![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FNevermindNilas%2FTheAnimeScripter%2F&labelColor=%23697689&countColor=%23ff8a65&style=flat-square&labelStyle=none)](https://visitorbadge.io/status?path=https%3A%2F%2Fgithub.com%2FNevermindNilas%2FTheAnimeScripter%2F)
[![Release](https://img.shields.io/github/release/NevermindNilas/TheAnimeScripter.svg?style=flat-square&color=blue)](https://github.com/NevermindNilas/TheAnimeScripter/releases)
[![Downloads](https://img.shields.io/github/downloads/NevermindNilas/TheAnimeScripter/total.svg?style=flat-square&color=%2364ff82)](https://github.com/NevermindNilas/TheAnimeScripter/releases)
[![Last Commit](https://img.shields.io/github/last-commit/NevermindNilas/TheAnimeScripter.svg?style=flat-square)](https://github.com/NevermindNilas/TheAnimeScripter/commits)
[![Discord](https://img.shields.io/discord/1041502781808328704?style=flat-square&logo=discord&logoColor=white&label=Discord&color=5865F2)](https://discord.gg/hwGHXga8ck)
[![License](https://img.shields.io/github/license/NevermindNilas/TheAnimeScripter?style=flat-square&color=orange)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/NevermindNilas/TheAnimeScripter?style=flat-square&color=yellow)](https://github.com/NevermindNilas/TheAnimeScripter/stargazers)

</div>

## 📋 Overview

TheAnimeScripter (TAS) is a cutting-edge AI-powered video enhancement toolkit specialized for anime and general video content. It seamlessly integrates with Adobe After Effects while also offering standalone functionality, bringing professional-grade AI upscaling, interpolation, and restoration to creators.

##  Table of Contents

- [ Key Features](#-key-features)
- [🖥️ User Interfaces](#️-user-interfaces)
- [🛠️ Installation Guide](#️-installation-guide)
- [📚 Command Reference](#-command-reference)
- [📁 Supported Models](#-supported-models)
- [📈 Performance Benchmarks](#-performance-benchmarks)
- [👨‍ Contributors](#-contributors)
- [⭐ Project Growth](#-project-growth)
- [🎥 Demo & Examples](#-demo--examples)

## 🚀 Key Features

<table>
<tr>
<td width="50%">

### 🎞️ Video Enhancement
- **Motion Interpolation:** Create buttery-smooth animation with advanced frame interpolation
- **AI Upscaling:** Enhance resolution with AI-powered upscaling (2x)
- **Smart Deduplication:** Optimize file size and interpolation by removing redundant frames

</td>
<td width="50%">

### 🎭 Advanced Editing
- **Background-Foreground Segmentation:** Precise automatic rotobrushing
- **Depth Map Generation:** 3D-ready depth maps for creative effects
- **Automatic Scene Detection:** Intelligent clip splitting at scene changes

</td>
</tr>
<tr>
<td>

### 🔧 Workflow Optimization
- **After Effects Integration:** Seamless plugin for AE workflow
- **Model Chaining:** Combine multiple effects in a single processing pass
- **In-Memory Processing:** Efficient frame handling without redundant disk operations

</td>
<td>

### 🧠 AI Flexibility
- **Multi-Backend Support:** CUDA, TensorRT, DirectML, and NCNN acceleration
- **Custom Model Support:** Import your own trained models
- **Restoration Options:** Denoise, dejpeg, sharpen, and line enhancement

</td>
</tr>
</table>

## 🖥️ User Interfaces

<div class="interface-container">

### <img src="https://img.icons8.com/color/24/000000/adobe-after-effects--v1.png" width="20"/> Adobe After Effects Integration
> *Currently being enhanced with new features and optimizations*

Adobe plugin provides seamless integration directly within your AE workflow, enabling AI-powered video enhancement without leaving your editing environment.

![Adobe GUI](https://github.com/user-attachments/assets/eb906836-cfd5-4f10-a004-cd4d0551b75e)

### <img src="https://img.icons8.com/color/24/000000/windows-10.png" width="20"/> Windows Standalone
> *Under active development - coming soon*

The native Windows application will provide a dedicated environment optimized for batch processing and advanced customization options.

</div>

## 🛠️ Getting Started

<div class="setup-container">

### <img src="https://img.icons8.com/color/24/000000/adobe-after-effects--v1.png" width="20"/> Adobe After Effects Plugin

<div class="requirements-box">

**System Requirements:**
- After Effects 2022 or higher
- Compatible GPU recommended:
  - **Modern NVIDIA** (RTX 20/30/40, GTX 16 series): Full CUDA/TensorRT support
  - **Older NVIDIA** (GTX 1000 series/Pascal): DirectML / NCNN backend (automatic)
  - **AMD/Intel GPUs**: DirectML / NCNN backend supported

</div>

#### Installation Steps:
1. Download the [**TAS-AdobeEdition**](https://github.com/NevermindNilas/TheAnimeScripter/releases/) from the releases page
2. Extract the `.zip` file to a location of your choice
3. Follow the [**installation tutorial**](https://nevermindnilas.github.io/zxp-installation/) to add TAS to After Effects

<div class="help-box">

> **Need help?** Watch the [video tutorial](https://youtu.be/JAdZ3z-os_A?si=fZQPmhMLtHfAktwn)

</div>

### <img src="https://img.icons8.com/fluency/24/000000/windows-client.png" width="20"/> Standalone Application

<div class="dev-status">

> **Development Status:** Currently in active development. Join the [Discord](https://discord.gg/hwGHXga8ck) for development updates.

</div>

### <img src="https://img.icons8.com/color/24/000000/console.png" width="20"/> Command Line Interface

Get the most powerful and flexible version of TAS with the command-line interface:

<div class="download-options">

- **[⬇️ Stable Release](https://github.com/NevermindNilas/TheAnimeScripter/releases)** — Recommended for production work
- **[⬇️ Nightly Builds](https://github.com/NevermindNilas/TAS-Nightly/releases)** — Latest features (may contain bugs)

</div>

</div>

## 📚 Available Inputs

All available parameters for interacting with the CLI or directly with `main.py` can be found in the [Parameters](PARAMETERS.MD) guide.

## 📁 Available Models

### 🆙 Upscaling Models

| Model                 | CUDA  | TensorRT | DirectML | NCNN  |
| --------------------- | :---: | :------: | :------: | :---: |
| ShuffleCugan          |   ✅   |    ✅     |    ❌     |   ✅   |
| Fallin Soft           |   ✅   |    ✅     |    ✅     |   ❌   |
| Fallin Strong         |   ✅   |    ✅     |    ✅     |   ❌   |
| Span                  |   ✅   |    ✅     |    ✅     |   ✅   |
| SRVGGNet (Compact)    |   ✅   |    ✅     |    ✅     |   ❌   |
| SRVGGNet (UltraCompact)      |   ✅   |    ✅     |    ✅     |   ❌   |
| SRVGGNet (SuperUltraCompact) |   ✅   |    ✅     |    ✅     |   ❌   |
| OpenProteus           |   ✅   |    ✅     |    ✅     |   ❌   |
| AniScale 2            |   ✅   |    ✅     |    ✅     |   ❌   |
| RTMOSR                |   ❌   |    ✅     |    ✅     |   ❌   |
| Custom (Spandrel)     |   ✅   |    ❌     |    ❌     |   ❌   |

### ⏱️ Interpolation Models (RIFE)

| Version               | CUDA  | TensorRT | DirectML | NCNN  |
| --------------------- | :---: | :------: | :------: | :---: |
| 4.6                   |   ✅   |    ✅     |    ❌     |   ✅   |
| 4.15                  |   ✅   |    ✅     |    ❌     |   ✅   |
| 4.15-lite             |   ✅   |    ✅     |    ❌     |   ✅   |
| 4.16-lite             |   ✅   |    ❌     |    ❌     |   ✅   |
| 4.17                  |   ✅   |    ✅     |    ❌     |   ✅   |
| 4.18                  |   ✅   |    ✅     |    ❌     |   ✅   |
| 4.20                  |   ✅   |    ✅     |    ❌     |   ✅   |
| 4.21                  |   ✅   |    ✅     |    ❌     |   ✅   |
| 4.22                  |   ✅   |    ✅     |    ❌     |   ✅   |
| 4.22-lite             |   ✅   |    ✅     |    ❌     |   ✅   |
| 4.25                  |   ✅   |    ✅     |    ❌     |   ❌   |
| 4.25-heavy            |   ✅   |    ✅     |    ❌     |   ❌   |
| Rife_Elexor (mod 4.7) |   ✅   |    ✅     |    ❌     |   ❌   |

### 🔧 Restoration Models

| Model               | CUDA  | TensorRT | DirectML | NCNN  |
| ------------------- | :---: | :------: | :------: | :---: |
| SCUNet (Denoise)    |   ✅   |    ✅     |    ❌     |   ❌   |
| NAFNet (Denoise)    |   ✅   |    ❌     |    ❌     |   ❌   |
| DPIR (Denoise)      |   ✅   |    ❌     |    ❌     |   ❌   |
| DeJpeg ( Real-Plksr ) |   ✅   |    ❌     |    ❌     |   ❌   |
| Anime1080fixer      |   ✅   |    ✅     |    ✅     |   ❌   |
| FastLineDarken      |   ✅   |    ✅     |    ❌     |   ❌   |
| GaterV3             |   ✅   |    ❌     |    ✅     |   ❌   |
| DeH264 ( Real-Plksr ) |   ✅   |    ✅     |    ✅     |   ❌   |

### 🌊 Depth Map Models

| Model                                 | CUDA  | TensorRT | DirectML | NCNN  |
| ------------------------------------- | :---: | :------: | :------: | :---: |
| **"Faster" Depth-Anything v2 Models** |       |          |          |       |
| Small v2                              |   ✅   |    ✅     |    ✅     |   ❌   |
| Base v2                               |   ✅   |    ✅     |    ✅     |   ❌   |
| Large v2                              |   ✅   |    ✅     |    ✅     |   ❌   |
| **"Faster" Distilled Models**         |       |          |          |       |
| Distill Small v2                      |   ✅   |    ✅     |    ✅     |   ❌   |
| Distill Base v2                       |   ✅   |    ✅     |    ✅     |   ❌   |
| Distill Large v2                      |   ✅   |    ✅     |    ✅     |   ❌   |
| **Original Implementation Models**    |       |          |          |       |
| OG Small v2                           |   ✅   |    ✅     |    ✅     |   ❌   |
| OG Base v2                            |   ✅   |    ✅     |    ✅     |   ❌   |
| OG Large v2                           |   ✅   |    ✅     |    ✅     |   ❌   |
| OG Distill Small v2                   |   ✅   |    ✅     |    ✅     |   ❌   |
| OG Distill Base v2                    |   ✅   |    ✅     |    ✅     |   ❌   |
| OG Distill Large v2                   |   ✅   |    ✅     |    ✅     |   ❌   |

## 🙏 Project Contributors

<div class="contributors-container">

### 🧠 Model & Algorithm Contributors
| Contributor                                       | Contribution                 | Repository                                                                             |
| ------------------------------------------------- | ---------------------------- | -------------------------------------------------------------------------------------- |
| [SUDO](https://github.com/styler00dollar)         | ShuffleCugan & ONNX models   | [VSGAN-tensorrt-docker](https://github.com/styler00dollar/VSGAN-tensorrt-docker)       |
| [renarchi](https://github.com/renarchi)           | Fallin Soft & Strong models  | [Fallin-Upscale](https://github.com/renarchi/models/releases/tag/Fallin)               |
| [HZWER](https://github.com/hzwer)                 | RIFE interpolation framework | [Practical-RIFE](https://github.com/hzwer/Practical-RIFE)                              |
| [the-database](https://github.com/the-database)   | SRVGGNet model variants      | [mpv-upscale-2x_animejanai](https://github.com/the-database/mpv-upscale-2x_animejanai) |
| [SkyTNT](https://github.com/SkyTNT)               | Anime segmentation models    | [anime-segmentation](https://github.com/SkyTNT/anime-segmentation)                     |
| [DepthAnything](https://github.com/DepthAnything) | Depth map generation         | [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)                |
| [Sirosky](https://github.com/Sirosky)             | Open-Proteus & AniScale 2    | [Upscale-Hub](https://github.com/Sirosky/Upscale-Hub)                                  |
| [Elexor](https://github.com/elexor)               | Custom RIFE modifications    | [Modded Rife Experiment(s)](https://github.com/elexor)                                 |
| [Zarxrax](https://github.com/Zarxrax)             | Anime1080Fixer restoration   | [GitHub](https://github.com/Zarxrax)                                                   |
| [umzi](https://github.com/umzi2)                  | RTMOSR & GaterV3 Models      | [GitHub](https://github.com/umzi2)                                                     |
| [Phhofm](https://github.com/Phhofm/models) | DeJpeg & DeH264 restoration |
****

### 🛠️ Framework & Tool Contributors
| Contributor                                     | Contribution                           | Repository                                                     |
| ----------------------------------------------- | -------------------------------------- | -------------------------------------------------------------- |
| [FFmpeg Group](https://github.com/FFmpeg)       | Video processing framework             | [FFmpeg](https://github.com/FFmpeg/FFmpeg)                     |
| [YT-DLP Team](https://github.com/yt-dlp)        | Media download capabilities            | [yt-dlp](https://github.com/yt-dlp/yt-dlp)                     |
| [Breakthrough](https://github.com/Breakthrough) | Scene detection algorithms             | [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) |
| [Chainner-org](https://github.com/chaiNNer-org) | Neural network architecture            | [spandrel](https://github.com/chaiNNer-org/spandrel)           |
| [TNTWise](https://github.com/TNTwise)           | RIFE ONNX/NCNN optimizations           | [GitHub](https://github.com/TNTwise)                           |
| [Hyperbrew](https://github.com/hyperbrew)       | Adobe integration framework            | [bolt-cep](https://github.com/hyperbrew/bolt-cep)              |


### 🌟 TAS Collaborators
| Contributor                                     | Contribution                                    |
| ----------------------------------------------- | ----------------------------------------------- |
| [Trentonom0r3](https://github.com/Trentonom0r3) | TAS Adobe Edition                               |
| [Adegerard](https://github.com/adegerard)       | Project architecture & optimization suggestions |

<div class="contact-info">

> 📧 **Missing contributor?** Please contact me at [nilascontact@gmail.com](mailto:nilascontact@gmail.com) or open a Github Issue!

</div>
</div>

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=NevermindNilas/TheAnimeScripter&type=Date)](https://star-history.com/#NevermindNilas/TheAnimeScripter&Date)

## 🎥 Promo Video

[![Promo Video](https://img.youtube.com/vi/V7ryKMezqeQ/0.jpg)](https://youtu.be/V7ryKMezqeQ)
**
