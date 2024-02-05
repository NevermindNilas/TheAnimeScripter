<p align="center">
    <a href="https://visitorbadge.io/status?path=https%3A%2F%2Fgithub.com%2FNevermindNilas%2FTheAnimeScripter%2F"><img src="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FNevermindNilas%2FTheAnimeScripter%2F&labelColor=%23697689&countColor=%23ff8a65&style=plastic&labelStyle=none" /></a> 
    <a href="https://github.com/NevermindNilas/TheAnimeScripter/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/NevermindNilas/TheAnimeScripter.svg?style=flat-square" /></a>
    <a href="https://github.com/NevermindNilas/TheAnimeScripter/releases"><img alt="GitHub All Releases" src="https://img.shields.io/github/downloads/NevermindNilas/TheAnimeScripter/total.svg?style=flat-square&color=%2364ff82" /></a>
    <a href="https://github.com/NevermindNilas/TheAnimeScripter/commits"><img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/NevermindNilas/TheAnimeScripter.svg?style=flat-square" /></a>
    <a href="https://discord.gg/hwGHXga8ck">
      <img src="https://img.shields.io/discord/1041502781808328704?label=Discord" alt="Discord Shield"/></a>
</p>
<p align="center">
    <a href="https://www.buymeacoffee.com/nilas" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>
</p>


# TheAnimeScripter

Welcome to TheAnimeScripter, a comprehensive tool designed for both video processing enthusiasts and professionals all within After Effects. Our tool offers a wide array of functionalities, including interpolation, upscaling, deduplication, segmentation, and more.

## Promo Video
[![Promo Video](https://img.youtube.com/vi/V7ryKMezqeQ/0.jpg)](https://youtu.be/V7ryKMezqeQ)

## 🚀 Key Features

1. **Smooth Motion Interpolation:** Elevate video quality with seamless frame interpolation for fluid motion.

2. **Crystal-Clear Upscaling:** Immerse audiences in enhanced resolution, delivering sharper, more detailed visuals.

3. **Optimal Video Size Deduplication:** Streamline videos by intelligently removing redundant frames, optimizing file size.

4. **Effortless Background-Foreground Segmentation:** Simplify rotobrushing tasks with precision and ease.

5. **3D Wizardry - Depth Map Generation:** Unlock advanced editing possibilities with detailed depth maps for immersive 3D effects.

6. **Auto Clip Cutting with Scene Change Filter:** Boost productivity by automatically cutting clips with a Scene Change Filter.

7. **Realistic Dynamics - Motion Blur:** Infuse realism into videos with customizable motion blur for a cinematic touch.

8. **Seamless After Effects Integration:** Enhance After Effects projects effortlessly with our seamless integration.

9. **Multi-Effect Magic - Model Chaining:** Combine features seamlessly within After Effects, running Deduplication, Upscaling, and Interpolation in one go.

10. **Efficient In-Memory Processing:** Experience swift transformations without additional frame extraction cycles.

11. **Custom Model Support for Creativity:** Unleash your creativity by incorporating your own trained models effortlessly.

Empower your video editing journey with these robust, efficient features designed to elevate your content to new heights.


## 🛠️ Getting Started

### How to Download

- Download one of the latest releases from [here](https://github.com/NevermindNilas/TheAnimeScripter/releases)

### Or Manually Build

- **Cutting Edge Builds:**
  - Download and install Python 3.12 from [here](https://www.python.org/downloads/release/python-3121/) (add to System Path).
  - Open a terminal inside the folder.
  - Run: `python build.py`

### How to Use Inside of After Effects

1. Open `File -> Scripts -> Install ScriptUI Panel`.
2. Choose `TheAnimeScripter.jsx` file from the folder.
3. Restart After Effects when prompted.
4. After reopening, go to `Window -> TheAnimeScripter.jsx`.
5. Dock the panel wherever you prefer.

In the settings panel:
- Set the folder to the same directory as The Anime Scripter.
- Specify the output location.

## 📚 Available Inputs

The available inputs have been moved to [Parameters Guide](PARAMETERS.MD).


## 🙏 Acknowledgements

- [SUDO](https://github.com/styler00dollar/VSGAN-tensorrt-docker) - For helping me debug my code and shufflecugan model
- [HZWER](https://github.com/hzwer/Practical-RIFE) - For Rife.
- [AILAB](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) - For Cugan.
- [JingyunLiang](https://github.com/JingyunLiang/SwinIR) - For SwinIR.
- [Xintao](https://github.com/xinntao/Real-ESRGAN) - for Realesrgan, specifically compact arch.
- [the-database](https://github.com/the-database/mpv-upscale-2x_animejanai) - For Compact, UltraCompact, SuperUltraCompact models
- [Tohrusky](https://github.com/Tohrusky/realcugan-ncnn-py) - For RealCugan-NCNN-Vulkan wrapper.
- [SkyTNT](https://github.com/SkyTNT/anime-segmentation) - For Anime Segmentation.
- [LiheYoung](https://github.com/LiheYoung/Depth-Anything) - For Depth Anything.
- [98mxr](https://github.com/98mxr/GMFSS_Fortuna) - For GMFSS Fortuna Union.
- [HolyWU](https://github.com/HolyWu/vs-gmfss_fortuna/tree/master) - For VS GMFSS code.
- [FFmpeg Group](https://github.com/FFmpeg/FFmpeg) - For FFmpeg
- [Nihui](https://github.com/nihui/rife-ncnn-vulkan) - For Rife NCNN
- [Media2x](https://github.com/media2x/rife-ncnn-vulkan-python) - For Rife NCNN Wrapper
- [TNTWise](https://github.com/TNTwise/rife-ncnn-vulkan) - For newest implementations of Rife
- [YT-DLP](https://github.com/yt-dlp/yt-dlp) - For YT-DLP
- [Hongyuanyu](https://github.com/hongyuanyu/span) - For SPAN
- [Phhofm](https://github.com/phhofm) - For SwinIR, Span and OmniSR Models
- [Francis0625](https://github.com/Francis0625/Omni-SR) - For OmniSR
- [Breakthrough](https://github.com/Breakthrough/PySceneDetect) - For Automated Scene Detection

## 📈 Benchmarks

The following benchmarks were conducted on a system with a 13700k and 3090 GPU for 1920x1080p inputs and take x264 encoding into account, FP16 on where possible.

| Test Category | Method | FPS | Notes |
| --- | --- | --- | --- |
| **Interpolation 2x** |
| | Rife (v4.13-lite) | ~120 | Ensemble False |
| | Rife (v4.14-lite) | ~100 | Ensemble False |
| | Rife (v4.14) | ~100 | Ensemble False |
| | Rife (v4.13-lite) | ~100 | Ensemble True |
| | Rife (v4.14-lite) | ~80 | Ensemble True |
| | Rife (v4.14) | ~80 | Ensemble True |
| | Rife (v4.13-lite-ncnn) | ~60 |  |
| | Rife (v4.14-lite-ncnn) | ~50 |  |
| | Rife (v4.14-ncnn) | ~50 |  |
| | GMFSS Fortuna Union | ~7 | Ensemble False |
| **Upscaling 2x** | 
| | Shufflecugan | ~21 | |
| | Compact | ~15 | |
| | UltraCompact | ~25 | |
| | SuperUltraCompact | ~30 | |
| | SwinIR | ~1 | |
| | Cugan | ~9 | |
| | Cugan-NCNN | ~7 | |
| | SPAN | ~9 | |
| | OmniSR | ~1 | |
| **Depth Map Generation** | 
| | Depth Anything VITS | ~16 | |
| | Depth Anything VITB | ~11 | |
| | Depth Anything VITL | ~7 | |
| **Segmentation** | 
| | Isnet-Anime | ~10 | |
| **Motion Blur** | 
| | 2xRife + Gaussian Averaging | ~23 | Still in work |

Please note that these benchmarks are approximate and actual performance may vary based on specific video content, settings, and other factors.


## ✅ Stats
![Alt](https://repobeats.axiom.co/api/embed/4754b52201c8220b8611a8c6e43c53ed3dc82a9f.svg "Repobeats analytics image")
