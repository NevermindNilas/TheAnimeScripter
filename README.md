<p align="center">
    <a href="https://visitorbadge.io/status?path=https%3A%2F%2Fgithub.com%2FNevermindNilas%2FTheAnimeScripter%2F"><img src="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FNevermindNilas%2FTheAnimeScripter%2F&labelColor=%23697689&countColor=%23ff8a65&style=plastic&labelStyle=none" /></a> 
    <a href="https://github.com/NevermindNilas/TheAnimeScripter/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/NevermindNilas/TheAnimeScripter.svg?style=flat-square" /></a>
    <a href="https://github.com/NevermindNilas/TheAnimeScripter/releases"><img alt="GitHub All Releases" src="https://img.shields.io/github/downloads/NevermindNilas/TheAnimeScripter/total.svg?style=flat-square&color=%2364ff82" /></a>
    <a href="https://github.com/NevermindNilas/TheAnimeScripter/commits"><img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/NevermindNilas/TheAnimeScripter.svg?style=flat-square" /></a>
</p>
<p align="center">
    <a href="https://www.buymeacoffee.com/nilas" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>
</p>

# TheAnimeScripter

Welcome to TheAnimeScripter, a comprehensive tool designed for both video processing enthusiasts and professionals all within After Effects. Our tool offers a wide array of functionalities, including interpolation, upscaling, deduplication, segmentation, and more.

[Join The Discord Server](https://discord.gg/bFA6xZxM5V)

## Promo Video

<iframe width="560" height="315" src="https://www.youtube.com/embed/V7ryKMezqeQ" frameborder="0" allowfullscreen></iframe>


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

- `--version`: (bool, action=store_true) Outputs the script version.
- `--input`: (str) Absolute path of the input video.
- `--output`: (str) Output string of the video, can be an absolute path or just a name.
- `--interpolate`: (int, default=0) Enable interpolation (1) or disable (0).
- `--interpolate_factor`: (int, default=2) Factor for interpolation.
- `--interpolate_method`: (str, default="rife") Interpolation method:
  - Options: "rife", "rife4.6", "rife4.14", "rife4.14-lite", "rife4.13-lite", "rife-ncnn", "rife4.6-ncnn", "rife4.14-ncnn", "rife4.14-lite-ncnn", "rife4.13-lite-ncnn".
  
- `--upscale`: (int, default=0) Enable upscaling (1) or disable (0).
- `--upscale_factor`: (int, default=2) Factor for upscaling.
- `--upscale_method`: (str, default="ShuffleCugan") Upscaling method:
  - Options: "cugan / cugan-ncnn / shufflecugan", "swinir", "compact / ultracompact / superultracompact", "span", "omnisr".
  
- `--cugan_kind`: (str, default="no-denoise") Cugan denoising kind:
  - Options: "no-denoise", "conservative", "denoise1x", "denoise2x".

- `--dedup`: (int, default=0) Enable deduplication (1) or disable (0).
- `--dedup_method`: (str, default="ffmpeg") Deduplication method.

- `--dedup_sens`: (float, default=50) Sensitivity of deduplication.
- `--half`: (int, default=1) Use half precision (1) or full precision (0).
- `--inpoint`: (float, default=0) Inpoint for the video.
- `--outpoint`: (float, default=0) Outpoint for the video.
- `--sharpen`: (int, default=0) Enable sharpening (1) or disable (0).
- `--sharpen_sens`: (float, default=50) Sensitivity of sharpening.
- `--segment`: (int, default=0) Enable segmentation (1) or disable (0).
- `--scenechange`: (int, default=0) Enable scene change detection (1) or disable (0).
- `--scenechange_sens`: (float, default=40) Sensitivity of scene change detection.
- `--depth`: (int, default=0) Generate Depth Maps (1) or disable (0).
- `--depth_method`: (str, default="small") Depth map generation method:
  - Options: "small", "base", "large".
  
- `--encode_method`: (str, default="x264") Encoding method:
  - Options: x264, x264_animation, nvenc_h264, nvenc_h265, qsv_h264, qsv_h265, h264_amf, hevc_amf.
  
- `--motion_blur`: (int, default=0) Add motion blur using gaussian weighting between frames.

- `--ytdlp`: (str, default="") Download a YouTube video, needs a full URL.
- `--ytdlp_quality`: (int, default=0) Allow 4k/8k videos to be downloaded then reencoded to the selected `--encode_method`.

- `--ensemble`: (int, default=0) Activate Ensemble for higher quality outputs from Rife (doesn't work with ncnn versions for now).
- `--resize`: (int, choices=[0, 1], default=0) Enable resizing (1) or disable (0).
- `--resize_factor`: (int, default=2) Factor for resizing the decoded video. Can also be a float value between 0 and 1 for downscaling.
- `--resize_method`: (str, choices=["fast_bilinear", "bilinear", "bicubic", "experimental", "neighbor", "area", "bicublin", "gauss", "sinc", "lanczos", "spline"], default="bicubic") Resizing method:
  - Options: "lanczos" (recommended for upscaling), "area" (recommended for downscaling).

- `--custom_model`: (str, default="") Choose a different model for supported upscaling arches. Relies on `--upscaling_factor` and `--upscaling_method`. Input must be the full path to a desired .pth or .onnx file.


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
