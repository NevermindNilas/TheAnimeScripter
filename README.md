<p align="center">
    <a href="https://visitorbadge.io/status?path=https%3A%2F%2Fgithub.com%2FNevermindNilas%2FTheAnimeScripter%2F"><img src="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FNevermindNilas%2FTheAnimeScripter%2F&labelColor=%23697689&countColor=%23ff8a65&style=plastic&labelStyle=none" /></a>
    <a href="https://github.com/NevermindNilas/TheAnimeScripter/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/NevermindNilas/TheAnimeScripter.svg?style=flat-square" /></a>
    <a href="https://github.com/NevermindNilas/TheAnimeScripter/releases"><img alt="GitHub All Releases" src="https://img.shields.io/github/downloads/NevermindNilas/TheAnimeScripter/total.svg?style=flat-square&color=%2364ff82" /></a>
    <a href="https://github.com/NevermindNilas/TheAnimeScripter/commits"><img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/NevermindNilas/TheAnimeScripter.svg?style=flat-square" /></a>
    <a href="https://discord.gg/hwGHXga8ck">
      <img src="https://img.shields.io/discord/1041502781808328704?label=Discord" alt="Discord Shield"/></a>
</p>

# The Anime Scripter  ( TAS )

## Table of Contents
- [The Anime Scripter  ( TAS )](#the-anime-scripter---tas-)
  - [Table of Contents](#table-of-contents)
  - [🚀 Key Features](#-key-features)
  - [🖥️ Graphical User Interfaces](#️-graphical-user-interfaces)
  - [🛠️ Getting Started](#️-getting-started)
    - [Adobe After Effects Integration](#adobe-after-effects-integration)
    - [Standalone / GUI:](#standalone--gui)
      - [Windows](#windows)
      - [Linux](#linux)
    - [CLI Interface:](#cli-interface)
      - [Windows / Linux](#windows--linux)
  - [📚 Available Inputs](#-available-inputs)
  - [📁 Available Models](#-available-models)
    - [Upscaling Models](#upscaling-models)
    - [Interpolation Models](#interpolation-models)
    - [Restore Models](#restore-models)
  - [📈 Benchmarks](#-benchmarks)
  - [🙏 Acknowledgements](#-acknowledgements)
  - [🌟 Star History](#-star-history)
  - [🎥 Promo Video](#-promo-video)

## 🚀 Key Features

1. **Smooth Motion Interpolation:** Elevate video quality with seamless frame interpolation for fluid motion.

2. **Crystal-Clear Upscaling:** Immerse audiences in enhanced resolution, delivering sharper, more detailed visuals.

3. **Optimal Video Size Deduplication:** Streamline videos by intelligently removing redundant frames, optimizing file size.

4. **Effortless Background-Foreground Segmentation:** Simplify rotobrushing tasks with precision and ease.

5. **3D Wizardry - Depth Map Generation:** Unlock advanced editing possibilities with detailed depth maps for immersive 3D effects.

6. **Auto Clip Cutting with Scene Change Filter:** Boost productivity by automatically cutting clips with a Scene Change Filter.

7. **Seamless After Effects Integration:** Enhance After Effects projects effortlessly with our seamless integration.

8. **Multi-Effect Magic - Model Chaining:** Combine features seamlessly within After Effects, running Deduplication, Upscaling, and Interpolation in one go.

9. **Efficient In-Memory Processing:** Experience swift transformations without additional frame extraction cycles.

10. **Custom Model Support for Creativity:** Unleash your creativity by incorporating your own trained models effortlessly.

11. **Graphical User Interface:** Navigate through our user-friendly interface designed for both beginners and professionals, ensuring a smooth editing workflow from start to finish.

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

### Upscaling Models

**Officially Supported:**
- ShuffleCugan (CUDA, TensorRT, and NCNN)
- Span (CUDA, TensorRT, DirectML and NCNN versions)
- SRVGGNet (Available in Compact, UltraCompact, SuperUltraCompact, and their respective TensorRT and DirectML versions)
- OpenProteus ( Cuda, TensorRT and DirectML )
- AniScale 2 ( Cuda, TensorRT and DirectML )
- RTMOSR ( TensorRT and DirectML )

**Unofficially Supported:**
- Custom models compatible with [Spandrel](https://github.com/chaiNNer-org/spandrel) can be used via the `--custom_model` parameter.

### Interpolation Models
- Rife CUDA (Versions: 4.6, 4.15, 4.15-lite, 4.16-lite, 4.17, 4.18, 4.20, 4.21, 4.22, 4.22-lite, 4.25, 4.25-heavy, Rife_Elexor ( modded 4.7 ) )
- Rife TensorRT (Versions: 4.6, 4.15, 4.15-lite, 4.17, 4.18, 4.20, 4.21, 4.22, 4.22-lite, 4.25, 4.25-heavy, Rife_Elexor ( modded 4.7 ) )
- Rife NCNN (Versions: 4.6, 4.15, 4.15-lite, 4.16-lite, 4.17, 4.18, 4.20, 4.21, 4.22, 4.22-lite )

### Restore Models
- SCUNet ( Denoise ) ( Cuda )
- NAFNet ( Denoise ) ( Cuda )
- DPIR ( Denoise ) ( Cuda )
- Real-Plksr (DeJpeg) ( Cuda )
- Anime1080fixer ( Sharpen + Decompress ) ( Cuda & TensorRT )
- FastLineDarken ( Line darkening ) ( Cuda & TensorRT )

## 📈 Benchmarks
Both internal and user-generated benchmarks can be found [here](BENCHMARKS.MD).

## 🙏 Acknowledgements

| Name                                                                      | For                                             |
| ------------------------------------------------------------------------- | ----------------------------------------------- |
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
