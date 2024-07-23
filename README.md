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
  <img src="https://raw.githubusercontent.com/NevermindNilas/TheAnimeScripter/main/src/assets/image.png" width="700" />
  <img src="https://github.com/NevermindNilas/TheAnimeScripter/assets/128264457/36671784-2512-4e9a-b7f2-a7035b741365" width="100" /> 
</p>

## 🛠️ Getting Started

### How to Download

Windows:
  - Download one of the latest releases from [here](https://github.com/NevermindNilas/TheAnimeScripter/releases)

Linux:
  - GUI is N/A yet. Refer to manual usage with the CLI and the [Parameters](PARAMETERS.MD) guide


### How to Use Inside of Adobe After Effects.

1. First, make sure to download the latest available `-AdobeEdition` from [HERE](https://github.com/NevermindNilas/TheAnimeScripter/releases/)
2. After downloading, run the `DO THIS FIRST.reg` file. This will allow TheAnimeScripter to run in Adobe After Effects.
3. Navigate to the CEP extensions folder located at `C:\Program Files (x86)\Common Files\Adobe\CEP\extensions\`.
4. Copy the entire folder to the path mentioned in step 3.
5. Restart Adobe After Effects to apply the changes.
6. After reopening, go to `Window -> Extensions -> TheAnimeScripter` to access the script.

## 📚 Available Inputs

All of the available parameters for interacting with the CLI or directly with main.py can be found in the [Parameters](PARAMETERS.MD) guide.


## 📁 Available Models

### Upscaling Models

**Officially Supported:**
- ShuffleCugan ( CUDA, TensorRT, and NCNN )
- Span ( CUDA, TensorRT, DirectML and NCNN versions )
- SRVGGNet (Available in Compact, UltraCompact, SuperUltraCompact, and their respective TensorRT and DirectML versions)

**Unofficially Supported:**
- Custom models compatible with [Spandrel](https://github.com/chaiNNer-org/spandrel) can be used via the `--custom_model` parameter.

### Interpolation Models
- Rife CUDA (Versions: 4.6, 4.15, 4.15-lite, 4.16-lite, 4.17, 4.18 )
- Rife TensorRT (Versions: 4.6, 4.15, 4.15-lite, 4.17, 4.18 )
- Rife NCNN ( Versions: 4.6, 4.15, 4.15-lite, 4.16-lite, 4.17, 4.18 ) 
- GMFSS ( Available for CUDA )

### Denoise Models
- SCUNet
- NAFNet
- DPIR
- Real-Plksr ( deJpeg )

## 📈 Benchmarks
Both internal and user generated benchmarks can be found [here](BENCHMARKS.MD).

## 🙏 Acknowledgements

| **Name and Link**                                                                                     | **For**              |
|-------------------------------------------------------------------------------------------------------|----------------------|
| [SUDO](https://github.com/styler00dollar/VSGAN-tensorrt-docker)                                       | Shufflecugan and many more ONNX models |
| [HZWER](https://github.com/hzwer/Practical-RIFE)                                                      | Rife                 |
| [the-database](https://github.com/the-database/mpv-upscale-2x_animejanai)                              | Compact, UltraCompact, SuperUltraCompact models |
| [SkyTNT](https://github.com/SkyTNT/anime-segmentation)                                                | Anime Segmentation   |
| [DepthAnything](https://github.com/DepthAnything/Depth-Anything-V2)                                              | Depth Anything  V2      |
| [98mxr](https://github.com/98mxr/GMFSS_Fortuna)                                                       | GMFSS Fortuna Union  |
| [FFmpeg Group](https://github.com/FFmpeg/FFmpeg)                                                      | FFmpeg               |
| [YT-DLP](https://github.com/yt-dlp/yt-dlp)                                                            | YT-DLP               |
| [Hongyuanyu](https://github.com/hongyuanyu/span)                                                      | SPAN                 |
| [Breakthrough](https://github.com/Breakthrough/PySceneDetect)                                         | Automated Scene Detection |
| [Chainner-org](https://github.com/chaiNNer-org/spandrel)                                              | Spandrel, easy to use arch implementations |
| [cszn](https://github.com/cszn/DPIR)                                                                  | DPIR                 |
| [TNTWise](https://github.com/TNTwise) | For Rife ONNX and NCNN models | 
| [WolframRhodium](https://github.com/WolframRhodium) | For Rife V2 models |

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=NevermindNilas/TheAnimeScripter&type=Date)](https://star-history.com/#NevermindNilas/TheAnimeScripter&Date)

## 🎥 Promo Video

[![Promo Video](https://img.youtube.com/vi/V7ryKMezqeQ/0.jpg)](https://youtu.be/V7ryKMezqeQ)
