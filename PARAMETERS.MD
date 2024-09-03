# Anime Scripter Parameters

## Table of Contents
1. [Overview](#overview)
2. [Command-Line Arguments](#command-line-arguments)
3. [Input and Output](#input-and-output)
4. [Processing Options](#processing-options)
   - [Upscaling](#upscaling)
   - [Interpolation](#interpolation)
   - [Scene Change Detection](#scene-change-detection)
   - [Deduplication](#deduplication)
   - [Denoising](#denoising)
   - [Sharpening](#sharpening)
   - [Segmentation](#segmentation)
   - [Auto Cut Clips Detection](#auto-cut-clips-detection)
   - [Depth Map Generation](#depth-map-generation)
   - [Resizing](#resizing)
5. [Encoding Settings](#encoding-settings)
6. [Video Stabilization](#video-stabilization)
7. [Performance Settings](#performance-settings)
8. [Usage Examples](#usage-examples)

## Overview

The Anime Scripter is a powerful tool for enhancing and manipulating videos with a wide range of options. It supports functionalities such as interpolation, upscaling, deduplication, denoising, sharpening, scene change detection, depth map generation, and more. Users can customize the processing pipeline to meet their specific requirements using a comprehensive set of parameters.

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--version` | flag | - | Outputs the script version |
| `--benchmark` | flag | - | Enable benchmarking (no video output, performance testing only) |
| `--offline` | str | "none" | Download models based on user selection |
| `--ae` | flag | False | Indicates if the script is run from the After Effects interface |
| `--preview` | flag | False | Enable previewing the process |
| `--hide_banner` | flag | False | Hide TAS' Banner when processing |

### Offline Mode Options
- `"none"`: Default state, does not download anything
- `"all"`: Downloads all available models
- Single Option (e.g., `"rife4.20"`): Downloads the specified model
- Multiple Options (e.g., `"rife4.20 rife4.18 shufflecugan-tensorrt"`): Downloads the specified models

## Input and Output

| Argument  | Type | Default | Description |
|-----------|------|---------|-------------|
| `--input` | str  | -       | Absolute path of the input video(s). Multiple paths should be separated by semicolons (`;`). |
| `--output` | str | -       | Output string or absolute path for the processed video |
| `--audio` | flag | -       | Enable audio processing |

- **NOTE:** `--input` supports `(".mp4", ".mkv", ".mov", ".avi", ".webm")` containers as well as `.txt`, inputs in `.txt` should be separated by a new line.

### Examples

#### Multiple Input Files
```sh
--input "G:\TheAnimeScripter\input.mp4;G:\TheAnimeScripter\input - Copy.mp4"

# Or if the input is a txt, this is how the process should look like 
--input "G:\TheAnimeScripter\test.txt"

# Contents of TXT:
"G:\TheAnimeScripter\input.mp4" # separate by a new line here
"G:\TheAnimeScripter\input - Copy.mp4"
```
## Processing Options

### Upscaling

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--upscale` | flag | - | Enable upscaling |
| `--upscale_factor` | int | 2 | Factor for upscaling |
| `--upscale_method` | str | "ShuffleCugan" | Upscaling method |
| `--custom_model` | str | "" | Path to a custom model file (.pth or .onnx) |
| `--upscale_skip` | flag | - | Skip processing duplicates for faster perceived upscaling |

#### Upscale Methods
- `"shufflecugan"` / `"shufflecugan-tensorrt"`
- `"compact"` / `"ultracompact"` / `"superultracompact"` (with `-directml` or `-tensorrt` variants)
- `"span"` / `"span-directml"` / `"span-tensorrt"`
- `"open-proteus"` / `"open-proteus-tensorrt"` / `"open-proteus-directml"`
- `"aniscale2"` / `"aniscale2-tensorrt"` / `"aniscale2-directml"`

### Interpolation

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--interpolate` | flag | - | Enable interpolation |
| `--interpolate_factor` | int | 2 | Factor for interpolation |
| `--interpolate_method` | str | "rife" | Interpolation method |
| `--ensemble` | flag | - | Activate Ensemble for higher quality Rife outputs |

#### Interpolate Methods
- `"rife4.6"` / `"rife4.6-ncnn"` / `"rife4.6-tensorrt"`
- `"rife4.15-lite"` / `"rife4.15-lite-ncnn"` / `"rife4.15-lite-tensorrt"`
- `"rife4.16-lite"` / `"rife4.16-lite-ncnn"`
- `"rife4.17"` / `"rife4.17-tensorrt"` / `"rife4.17-ncnn"`
- `"rife4.18"` / `"rife4.18-tensorrt"` / `"rife4.18-ncnn"`
- `"rife4.20"` / `"rife4.20-tensorrt"` / `"rife4.20-ncnn"`
- `"rife4.21"` / `"rife4.21-tensorrt"` / `"rife4.21-ncnn"`
- `"rife4.22"` / `"rife4.22-tensorrt"` / `"rife4.22-ncnn"`
- `"rife4.22-lite"` / `"rife4.22-lite-tensorrt"` / `"rife4.22-lite-ncnn"` 

### Scene Change Detection

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--scenechange` | flag | False | Enable scene change detection |
| `--scenechange_sens` | float | 50 | Sensitivity of scene change detection (0.80 - 0.90) |
| `--scenechange_method` | str | "maxxvit-directml" | Scene change detection method |

#### Scene Change Methods
- `"maxxvit-directml"`
- `"maxxvit-tensorrt"`

### Deduplication

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dedup` | flag | - | Enable deduplication |
| `--dedup_method` | str | "ssim" | Deduplication method |
| `--dedup_sens` | float | 50 | Sensitivity of deduplication |
| `--sample_size` | int | 224 | Sample size for SSIM deduplication |

#### Dedup Methods
- `"ffmpeg"`
- `"ssim"`
- `"mse"`
- `"ssim-cuda"`

### Denoising

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--denoise` | flag | - | Enable denoising |
| `--denoise_method` | str | "scunet" | Denoising model |

#### Denoise Methods
- `"scunet"`
- `"nafnet"`
- `"dpir"`
- `"real-plksr"`

### Sharpening

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--sharpen` | flag | - | Enable sharpening |
| `--sharpen_sens` | float | 50 | Sensitivity of sharpening |

### Segmentation

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--segment` | flag | - | Enable segmentation |
| `--segment_method` | str | "anime" | Segmentation method |

#### Segment Methods
- `"anime"`
- `"anime-tensorrt"`

### Auto Cut Clips Detection

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--autoclip` | flag | - | Enable auto cut clip detection |
| `--autoclip_sens` | float | 40 | Sensitivity of auto cut clip detection |

### Depth Map Generation

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--depth` | flag | - | Generate depth maps |
| `--depth_method` | str | "small" | Depth map generation method |
| `--depth_quality` | str | "high" | Depth map generation quality |

#### Depth Methods
- `"small_v2"` / `"small_v2-tensorrt"` / `"small_v2-directml"`
- `"base_v2"` / `"base_v2-tensorrt"` / `"base_v2-directml"`
- `"large_v2"` / `"large_v2-tensorrt"` / `"large_v2-directml"`

#### Depth Quality selection
- `"low"` / `"high"`
- Additional Notes: This is not reliable with TensorRT and should only be preferred if you own a low end GPU. 



### Resizing

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--resize` | flag | - | Enable resizing |
| `--resize_factor` | float | 2 | Factor for resizing (>1 for upscaling, <1 for downscaling) |
| `--resize_method` | str | "bicubic" | Resizing method |

#### Resize Methods
- `"fast_bilinear"`, `"bilinear"`, `"bicubic"`, `"experimental"`, `"neighbor"`, `"area"`
- `"bicublin"`, `"gauss"`, `"sinc"`, `"lanczos"`, `"spline"`, `"spline16"`, `"spline36"`, `"point"`

## Encoding Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--encode_method` | str | "x264" | Encoding method |
| `--custom_encoder` | str | "" | Custom FFmpeg encoding parameters |

### Encode Methods
- `"x264"`: libx264, preset veryfast, CRF 15
- `"x264_10bit"`: libx264, preset veryfast, CRF 15, high10 profile
- `"x264_animation"`: libx264, preset veryfast, tune animation, CRF 15
- `"x264_animation_10bit"`: libx264, preset veryfast, tune animation, CRF 15, high10 profile
- `"x265"`: libx265, preset veryfast, CRF 15
- `"x265_10bit"`: libx265, preset veryfast, CRF 15, main10 profile
- `"nvenc_h264"`: h264_nvenc, preset p1, CQ 15
- `"nvenc_h265"`: hevc_nvenc, preset p1, CQ 15
- `"nvenc_h265_10bit"`: hevc_nvenc, preset p1, CQ 15, main10 profile
- `"qsv_h264"`: h264_qsv, preset veryfast, global quality 15
- `"qsv_h265"`: hevc_qsv, preset veryfast, global quality 15
- `"qsv_h265_10bit"`: hevc_qsv, preset veryfast, global quality 15, main10 profile
- `"nvenc_av1"`: av1_nvenc, preset p1, CQ 15
- `"av1"`: libsvtav1, preset 8, CRF 15
- `"h264_amf"`: h264_amf, quality speed, RC CQP, QP 15
- `"hevc_amf"`: hevc_amf, quality speed, RC CQP, QP 15
- `"hevc_amf_10bit"`: hevc_amf, quality speed, RC CQP, QP 15, main10 profile
- `"prores"`: prores_ks, profile 4, qscale 15
- `"gif"`: gif, qscale 1, loop 0
- `"image"`: png, quality 1

**Note**: The output pixel format (`-pix_fmt`) can dynamically switch between yuv420p, yuv420p10le, and yuv444p10le based on `--bit_depth` and `--encode_method`.

## Video Stabilization

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--stabilize` | flag | - | Enable video stabilization |

## Performance Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--half` | flag | True | Enable FP16 for improved performance |
| `--buffer_limit` | int | 50 | Set the limit for decoding and encoding buffers |
| `--static` | flag | True | Force static engine generation and bypass the arbitrary set limitations of TAS |

**Note**: Interpolation TRT, Depth TRT, Segment TRT all already are set to static engine generation due to set limitations within the models themselves.

## Usage Examples

**Note**: Replace `.\input\test.mp4` with a full or relative path to a video file or folder. The `--output` argument is optional, as the script can automatically generate a file name based on the input.

When using the packaged `main.exe` file, replace `python .\main.py` with `main.exe` in the following examples.

1. Enable upscaling with half precision mode (default):
   ```sh
   python .\main.py --input .\input\test.mp4 --upscale
   ```

2. Enable interpolation with Rife 4.6:
   ```sh
   python .\main.py --input .\input\test.mp4 --interpolate --interpolate_method rife4.6
   ```

3. Enable upscaling and interpolation with custom models:
   ```sh
   python .\main.py --input .\input\test.mp4 --upscale --upscale_method superultracompact --interpolate --interpolate_method rife4.20
   ```

4. Enable deduplication with custom sensitivity and method:
   ```sh
   python .\main.py --input .\input\test.mp4 --dedup --dedup_sens 60 --dedup_method ssim
   ```

5. Enable denoising using the NAFNet model:
   ```sh
   python .\main.py --input .\input\test.mp4 --denoise --denoise_method nafnet
   ```

6. Enable interpolation with scene change detection:
   ```sh
   python .\main.py --input .\input\test.mp4 --interpolate --scenechange
   ```

7. Generate depth maps using the "large" method and 16-bit workflow:
   ```sh
   python .\main.py --input .\input\test.mp4 --depth --depth_method large_v2 --bit_depth 16bit
   ```

8. Download a YouTube video, upscale with Span-TensorRT, interpolate with Rife 4.17-TensorRT, and use scene change detection:
   ```sh
   python .\main.py --input https://www.youtube.com/watch?v=dQw4w9WgXcQ --upscale --upscale_method span-tensorrt --interpolate --interpolate_method rife4.17-tensorrt --scenechange
   ```
   (Replace the YouTube URL with the actual video URL you want to download)

9. Process video from second 8 to 17 with custom settings:
   ```sh
   python .\main.py --input .\input\test.mp4 --inpoint 8 --outpoint 17 --upscale --upscale_method superultracompact-tensorrt --interpolate --interpolate_method rife4.20 --scenechange --scenechange_method maxxvit-tensorrt --custom_encoder "-c:v libsvtav1 -preset 8 -crf 30 -vf hqdn3d=1.5:1.5:6:6 -pix_fmt yuv420p"
   ```

10. Upscale, Interpolate, Scenechange, x264_animation_10bit and 16bit workflow example:
   ```sh
   python .\main.py --input https://www.youtube.com/watch?v=YRL74JmhVgk --upscale --upscale_method shufflecugan-tensorrt --interpolate --interpolate_method rife4.22-tensorrt --scenechange --scenechange_method maxxvit-tensorrt --encode_method x264_animation_10bit --bit_depth 16bit   
   ```