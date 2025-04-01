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
- **AI Upscaling:** Enhance resolution with AI-powered upscaling (2x, 4x)
- **Smart Deduplication:** Optimize file size by removing redundant frames

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

## 🖥️ Graphical User Interfaces

- **Windows and After Effects:** Actively being reworked and improved!
  
  *Adobe GUI*
  ![Adobe GUI](https://github.com/user-attachments/assets/b2ebe67a-b9a2-4361-8bf5-be5d9ee37a12)


## 🛠️ Getting Started

### Adobe After Effects Integration

1. You must have After Effects 2020 v17.5 or higher!
2. Download the latest `-AdobeEdition` release from [here](https://github.com/NevermindNilas/TheAnimeScripter/releases/).
3. Extract the `.zip` file.
4. Follow this step-by-step [tutorial](https://www.goodboy.ninja/help/install/extensions)
> If installation fails, refer to the [manual installation guide](https://www.goodboy.ninja/help/install/extensions-manually).

> Video tutorial: https://youtu.be/JAdZ3z-os_A?si=fZQPmhMLtHfAktwn

### Standalone / GUI:

#### Windows
- N/A - Under active development.

#### Linux
- N/A - Under active development.

### CLI Interface:

#### Windows / Linux
- **Stable:** Download the latest release from [here](https://github.com/NevermindNilas/TheAnimeScripter/releases)
- **Nightly:** Download nightly versions from [here](https://github.com/NevermindNilas/TAS-Nightly/releases)


## 📚 Available Inputs

All available parameters for interacting with the CLI or directly with `main.py` can be found in the [Parameters](PARAMETERS.MD) guide.

## 📁 Available Models

### � Upscaling Models

| Model                 | CUDA | TensorRT | DirectML | NCNN |
|-----------------------|:----:|:--------:|:--------:|:----:|
| ShuffleCugan          |  ✅   |    ✅     |    ❌     |  ✅   |
| Span                  |  ✅   |    ✅     |    ✅     |  ✅   |
| SRVGGNet (Compact)    |  ✅   |    ✅     |    ✅     |  ❌   |
| SRVGGNet (Ultra)      |  ✅   |    ✅     |    ✅     |  ❌   |
| SRVGGNet (SuperUltra) |  ✅   |    ✅     |    ✅     |  ❌   |
| OpenProteus           |  ✅   |    ✅     |    ✅     |  ❌   |
| AniScale 2            |  ✅   |    ✅     |    ✅     |  ❌   |
| RTMOSR                |  ❌   |    ✅     |    ✅     |  ❌   |
| Custom (Spandrel)     |  ✅   |    ❌     |    ❌     |  ❌   |

### ⏱️ Interpolation Models (RIFE)

| Version               | CUDA | TensorRT | DirectML | NCNN |
|-----------------------|:----:|:--------:|:--------:|:----:|
| 4.6                   |  ✅   |    ✅     |    ❌     |  ✅   |
| 4.15                  |  ✅   |    ✅     |    ❌     |  ✅   |
| 4.15-lite             |  ✅   |    ✅     |    ❌     |  ✅   |
| 4.16-lite             |  ✅   |    ❌     |    ❌     |  ✅   |
| 4.17                  |  ✅   |    ✅     |    ❌     |  ✅   |
| 4.18                  |  ✅   |    ✅     |    ❌     |  ✅   |
| 4.20                  |  ✅   |    ✅     |    ❌     |  ✅   |
| 4.21                  |  ✅   |    ✅     |    ❌     |  ✅   |
| 4.22                  |  ✅   |    ✅     |    ❌     |  ✅   |
| 4.22-lite             |  ✅   |    ✅     |    ❌     |  ✅   |
| 4.25                  |  ✅   |    ✅     |    ❌     |  ❌   |
| 4.25-heavy            |  ✅   |    ✅     |    ❌     |  ❌   |
| Rife_Elexor (mod 4.7) |  ✅   |    ✅     |    ❌     |  ❌   |

### 🔧 Restoration Models

| Model               | CUDA | TensorRT | DirectML | NCNN |
|---------------------|:----:|:--------:|:--------:|:----:|
| SCUNet (Denoise)    |  ✅   |    ❌     |    ❌     |  ❌   |
| NAFNet (Denoise)    |  ✅   |    ❌     |    ❌     |  ❌   |
| DPIR (Denoise)      |  ✅   |    ❌     |    ❌     |  ❌   |
| Real-Plksr (DeJpeg) |  ✅   |    ❌     |    ❌     |  ❌   |
| Anime1080fixer      |  ✅   |    ✅     |    ❌     |  ❌   |
| FastLineDarken      |  ✅   |    ✅     |    ❌     |  ❌   |

### Depth Models


## 📈 Benchmarks
Both internal and user-generated benchmarks can be found [here](BENCHMARKS.MD).

## 🙏 Acknowledgements

| Name                                                                      | For                                             |
|---------------------------------------------------------------------------|-------------------------------------------------|
| [SUDO](https://github.com/styler00dollar/VSGAN-tensorrt-docker)           | Shufflecugan and many more ONNX models          |
| [HZWER](https://github.com/hzwer/Practical-RIFE)                          | Rife                                            |
| [the-database](https://github.com/the-database/mpv-upscale-2x_animejanai) | Compact, UltraCompact, SuperUltraCompact models |
| [SkyTNT](https://github.com/SkyTNT/anime-segmentation)                    | Anime Segmentation                              |
| [DepthAnything](https://github.com/DepthAnything/Depth-Anything-V2)       | Depth Anything V2                               |
| [FFmpeg Group](https://github.com/FFmpeg/FFmpeg)                          | FFmpeg                                          |
| [YT-DLP](https://github.com/yt-dlp/yt-dlp)                                | YT-DLP                                          |
| [Hongyuanyu](https://github.com/hongyuanyu/span)                          | SPAN                                            |
| [Breakthrough](https://github.com/Breakthrough/PySceneDetect)             | Automated Scene Detection                       |
| [Chainner-org](https://github.com/chaiNNer-org/spandrel)                  | Spandrel, easy to use arch implementations      |
| [TNTWise](https://github.com/TNTwise)                                     | Rife ONNX / NCNN and Spanimation                |
| [Hyperbrew](https://github.com/hyperbrew/bolt-cep)                        | Bolt CEP                                        |
| [Sirosky](https://github.com/Sirosky/Upscale-Hub)                         | Open-Proteus and AniScale 2                     |
| [Trentonom0r3](https://github.com/Trentonom0r3)                           | Helping with TAS Adobe Edition and Celux        |
| [Adegerard](https://github.com/adegerard)                                 | Several ideas on how to further improve TAS     |
| [Elexor](https://github.com/elexor)                                       | Modded Rife Experiment(s)                       |
| [Zarxrax](https://github.com/Zarxrax)                                     | Anime1080Fixer model                            |
| [sdaqo](https://github.com/sdaqo)                                         | Anipy-CLI                                       |
| [umzi](https://github.com/umzi2)                                          | RTMOSR                                          |

If I forgot to mention anyone, please email: nilascontact@gmail.com

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=NevermindNilas/TheAnimeScripter&type=Date)](https://star-history.com/#NevermindNilas/TheAnimeScripter&Date)

## 🎥 Promo Video

[![Promo Video](https://img.youtube.com/vi/V7ryKMezqeQ/0.jpg)](https://youtu.be/V7ryKMezqeQ)
