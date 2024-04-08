![TheAnimeScripter](https://socialify.git.ci/NevermindNilas/TheAnimeScripter/image?description=1&descriptionEditable=Welcome%20to%20TheAnimeScripter%20%E2%80%93%20the%20ultimate%20tool%20for%20video%20processing.%20Enjoy%20%20%20seamless%20video%20processing%20with%20our%20intuitive%20GUIs%20for%20Windows%20and%20After%20Effects.&font=KoHo&forks=1&issues=1&language=1&name=1&owner=1&pattern=Solid&pulls=1&stargazers=1&theme=Dark)

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

Empower your video editing journey with these robust, efficient features designed to elevate your content to new heights.


## 🖥️ Graphical User Interfaces for Windows and After Effects

<p float="left">
  <img src="https://github.com/NevermindNilas/TheAnimeScripter/assets/128264457/e1b456a3-ed14-4fd7-a2ac-60688a0c8475" width="700" />
  <img src="https://github.com/NevermindNilas/TheAnimeScripter/assets/128264457/36671784-2512-4e9a-b7f2-a7035b741365" width="100" /> 
</p>

## 🛠️ Getting Started

### How to Download

- Download one of the latest releases from [here](https://github.com/NevermindNilas/TheAnimeScripter/releases)

### Or Manually Build

- **Cutting Edge Builds:**
  - Download and install Python 3.11 from [here](https://www.python.org/downloads/release/python-3110/) (add to System Path).
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

## 📚 [Available Inputs](PARAMETERS.MD)

## 📁 Available Models

Upscaling: 
  - Official Support - Cugan / Cugan-NCNN, ShuffleCugan / ShuffleCugan-NCNN, Span / Span-NCNN, RealESRGan / RealESRGan-NCNN, OmniSR, SRVGGNet ( Compact, UltraCompact, SuperUltraCompact )
  - Unofficial Support through `--custom_model` param, virtually anything that [Spandrel](https://github.com/chaiNNer-org/spandrel) supports.

Interpolation: 
  - Rife 4.6, Rife4.14, Rife 4.15, Rife 4.16-lite and NCNN versions for each of them and GMFSS.
  
Denoise:
  - SCUNet, NAFNet, Span, DPIR ( 1x pretrain model )

## 🙏 Acknowledgements

- [SUDO](https://github.com/styler00dollar/VSGAN-tensorrt-docker) - For ShuffleCugan and Rife Models.
- [HZWER](https://github.com/hzwer/Practical-RIFE) - For Rife.
- [AILAB](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) - For Cugan.
- [JingyunLiang](https://github.com/JingyunLiang/SwinIR) - For SwinIR.
- [Xintao](https://github.com/xinntao/Real-ESRGAN) - for Realesrgan and Compact architectures.
- [the-database](https://github.com/the-database/mpv-upscale-2x_animejanai) - For Compact, UltraCompact, SuperUltraCompact models
- [SkyTNT](https://github.com/SkyTNT/anime-segmentation) - For Anime Segmentation.
- [LiheYoung](https://github.com/LiheYoung/Depth-Anything) - For Depth Anything.
- [98mxr](https://github.com/98mxr/GMFSS_Fortuna) - For GMFSS Fortuna Union.
- [FFmpeg Group](https://github.com/FFmpeg/FFmpeg) - For FFmpeg
- [Nihui](https://github.com/nihui/rife-ncnn-vulkan) - For Rife NCNN
- [Media2x](https://github.com/media2x/rife-ncnn-vulkan-python) - For Rife NCNN Wrapper
- [TNTWise](https://github.com/TNTwise/rife-ncnn-vulkan) - For newest implementations of Rife NCNN, Universal NCNN and Span model.
- [YT-DLP](https://github.com/yt-dlp/yt-dlp) - For YT-DLP
- [Hongyuanyu](https://github.com/hongyuanyu/span) - For SPAN
- [Phhofm](https://github.com/phhofm) - For RealESRGAN and OmniSR Models
- [Francis0625](https://github.com/Francis0625/Omni-SR) - For OmniSR
- [Breakthrough](https://github.com/Breakthrough/PySceneDetect) - For Automated Scene Detection
- [Final2X](https://github.com/Final2x/realesrgan-ncnn-py) - For RealESRGAN-NCNN-Py and RealCugan-NCNN-Py
- [Chainner-org](https://github.com/chaiNNer-org/spandrel) - For Spandrel and easy to use arch implementations.
- [Kim2091](https://openmodeldb.info/models/1x-span-anime-pretrain) - For Span 1x Model
- [Kiteretsu77](https://github.com/Kiteretsu77/APISR) - For APISR
- [cszn](https://github.com/cszn/DPIR) - For DPIR

## 📈 [Benchmarks](BENCHMARKS.MD)

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=NevermindNilas/TheAnimeScripter&type=Date)](https://star-history.com/#NevermindNilas/TheAnimeScripter&Date)

## 🎥 Promo Video

[![Promo Video](https://img.youtube.com/vi/V7ryKMezqeQ/0.jpg)](https://youtu.be/V7ryKMezqeQ)
