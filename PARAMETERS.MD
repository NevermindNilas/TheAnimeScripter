# Anime Scripter Parameters

## Table of Contents
1. [Overview](#overview)
2. [Command-Line Arguments](#command-line-arguments)
3. [Input and Output](#input-and-output)
4. [Processing Options](#processing-options)
   - [4.1 Upscaling](#upscaling)
   - [4.2 Interpolation](#interpolation)
   - [4.3 Scene Change Detection](#scene-change-detection)
   - [4.4 Deduplication](#deduplication)
   - [4.5 Denoising](#denoising)
   - [4.6 Sharpening](#sharpening)
   - [4.7 Segmentation](#segmentation)
   - [4.8 Auto Cut Clips Detection](#auto-cut-clips-detection)
   - [4.9 Depth Map Generation](#depth-map-generation)
   - [4.10 Resizing](#resizing)
   - [4.11 Presets](#preset-arguments)
5. [Encoding Settings](#encoding-settings)
6. [Video Stabilization](#video-stabilization)
7. [Performance Settings](#performance-settings)
8. [Usage Examples](#usage-examples)

## 1. Overview

The Anime Scripter is a powerful tool for enhancing and manipulating videos with a wide range of options. It supports functionalities such as interpolation, upscaling, deduplication, denoising, sharpening, scene change detection, depth map generation, and more. Users can customize the processing pipeline to meet their specific requirements using a comprehensive set of parameters.

## 2. Command-Line Arguments

| Argument        | Type | Default | Description                                                     |
|-----------------|------|---------|-----------------------------------------------------------------|
| `--version`     | flag | -       | Outputs the script version                                      |
| `--benchmark`   | flag | -       | Enable benchmarking (no video output, performance testing only) |
| `--offline`     | str  | "none"  | Download models based on user selection                         |
| `--ae`          | flag | False   | Indicates if the script is run from the After Effects interface |
| `--preview`     | flag | False   | Enable previewing the process                                   |
| `--hide_banner` | flag | False   | Hide TAS' Banner when processing                                |

### Offline Mode Options
- `"none"`: Default state, does not download anything
- `"all"`: Downloads all available models
- Single Option (e.g., `"rife4.20"`): Downloads the specified model
- Multiple Options (e.g., `"rife4.20 rife4.18 shufflecugan-tensorrt"`): Downloads the specified models


## 3. Input and Output

| Argument   | Type | Default | Description                                                                                  |
|------------|------|---------|----------------------------------------------------------------------------------------------|
| `--input`  | str  | -       | Absolute path of the input video(s). Multiple paths should be separated by semicolons (`;`). |
| `--output` | str  | -       | Output string or absolute path for the processed video                                       |

- **NOTE:** `--input` supports `(".mp4", ".mkv", ".mov", ".avi", ".webm")` containers as well as `.txt`, inputs in `.txt` should be separated by a new line.

### Examples

#### Multiple Input Files
```sh
--input "G:\TheAnimeScripter\input.mp4;G:\TheAnimeScripter\test.mp4"

# Or if the input is a txt, this is how the process should look like
--input "G:\TheAnimeScripter\test.txt"

# Contents of TXT:
"G:\TheAnimeScripter\test1.mp4" # separate by a new line here
"G:\TheAnimeScripter\test2.mp4"
```
## Processing Options

### Upscaling

| Argument           | Type | Default        | Description                                               |
|--------------------|------|----------------|-----------------------------------------------------------|
| `--upscale`        | flag | -              | Enable upscaling                                          |
| `--upscale_factor` | int  | 2              | Factor for upscaling                                      |
| `--upscale_method` | str  | "ShuffleCugan" | Upscaling method                                          |
| `--custom_model`   | str  | ""             | Path to a custom model file (.pth or .onnx)               |
| `--upscale_skip`   | flag | -              | Skip processing duplicates for faster perceived upscaling |

#### Upscale Methods
- `"shufflecugan"` / `"shufflecugan-tensorrt"`
- `"compact"` / `"ultracompact"` / `"superultracompact"` (with `-directml` or `-tensorrt` variants)
- `"span"` / `"span-directml"` / `"span-tensorrt"`
- `"open-proteus"` / `"open-proteus-tensorrt"` / `"open-proteus-directml"`
- `"aniscale2"` / `"aniscale2-tensorrt"` / `"aniscale2-directml"`

### Interpolation

| Argument               | Type | Default | Description                                                     |
|------------------------|------|---------|-----------------------------------------------------------------|
| `--interpolate`        | flag | -       | Enable interpolation                                            |
| `--interpolate_factor` | int  | 2       | Factor for interpolation                                        |
| `--interpolate_method` | str  | "rife"  | Interpolation method                                            |
| `--ensemble`           | flag | -       | Activate Ensemble for higher quality Rife outputs               |
| `--interpolate_skip`   | flag | -       | Enable duplicate frame skipping in order to improve performance |

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
- `"rife4.25"` / `"rife4.25-tensorrt"` / `N/A`

### Scene Change Detection

| Argument               | Type  | Default            | Description                                         |
|------------------------|-------|--------------------|-----------------------------------------------------|
| `--scenechange`        | flag  | False              | Enable scene change detection                       |
| `--scenechange_sens`   | float | 50                 | Sensitivity of scene change detection (0.80 - 0.90) |
| `--scenechange_method` | str   | "maxxvit-directml" | Scene change detection method                       |

#### Scene Change Methods
- `"maxxvit-directml"`
- `"maxxvit-tensorrt"`

### Deduplication

| Argument         | Type  | Default | Description                        |
|------------------|-------|---------|------------------------------------|
| `--dedup`        | flag  | -       | Enable deduplication               |
| `--dedup_method` | str   | "ssim"  | Deduplication method               |
| `--dedup_sens`   | float | 50      | Sensitivity of deduplication       |
| `--sample_size`  | int   | 224     | Sample size for SSIM deduplication |

#### Dedup Methods
- `"ffmpeg"`
- `"ssim"`
- `"mse"`
- `"ssim-cuda"`

### Denoising

| Argument           | Type | Default  | Description      |
|--------------------|------|----------|------------------|
| `--denoise`        | flag | -        | Enable denoising |
| `--denoise_method` | str  | "scunet" | Denoising model  |

#### Denoise Methods
- `"scunet"`
- `"nafnet"`
- `"dpir"`
- `"real-plksr"`

### Sharpening

| Argument         | Type  | Default | Description               |
|------------------|-------|---------|---------------------------|
| `--sharpen`      | flag  | -       | Enable sharpening         |
| `--sharpen_sens` | float | 50      | Sensitivity of sharpening |

### Segmentation

| Argument           | Type | Default | Description         |
|--------------------|------|---------|---------------------|
| `--segment`        | flag | -       | Enable segmentation |
| `--segment_method` | str  | "anime" | Segmentation method |

#### Segment Methods
- `"anime"`
- `"anime-tensorrt"`

### Auto Cut Clips Detection

| Argument          | Type  | Default | Description                            |
|-------------------|-------|---------|----------------------------------------|
| `--autoclip`      | flag  | -       | Enable auto cut clip detection         |
| `--autoclip_sens` | float | 40      | Sensitivity of auto cut clip detection |

### Depth Map Generation

| Argument          | Type | Default | Description                  |
|-------------------|------|---------|------------------------------|
| `--depth`         | flag | -       | Generate depth maps          |
| `--depth_method`  | str  | "small" | Depth map generation method  |
| `--depth_quality` | str  | "high"  | Depth map generation quality |

#### Depth Methods
- `"small_v2"` / `"small_v2-tensorrt"` / `"small_v2-directml"`
- `"base_v2"` / `"base_v2-tensorrt"` / `"base_v2-directml"`
- `"large_v2"` / `"large_v2-tensorrt"` / `"large_v2-directml"`

#### Depth Quality selection
- `"low"` / `"high"`
- Additional Notes: This is not reliable with TensorRT and should only be preferred if you own a low end GPU.



### Resizing

| Argument          | Type  | Default   | Description                                                |
|-------------------|-------|-----------|------------------------------------------------------------|
| `--resize`        | flag  | -         | Enable resizing                                            |
| `--resize_factor` | float | 2         | Factor for resizing (>1 for upscaling, <1 for downscaling) |
| `--resize_method` | str   | "bicubic" | Resizing method                                            |

#### Resize Methods
- `"fast_bilinear"`, `"bilinear"`, `"bicubic"`, `"experimental"`, `"neighbor"`, `"area"`
- `"bicublin"`, `"gauss"`, `"sinc"`, `"lanczos"`, `"spline"`, `"spline16"`, `"spline36"`, `"point"`

## Preset Arguments

| Argument        | Type   | Default | Description                                                                                           |
|-----------------|--------|---------|-------------------------------------------------------------------------------------------------------|
| `--preset`      | string | -       | A new and semi-baked in preset configuration system with reusability in mind.                         |
| `--preset_list` | flag   | False   | Prints all of the available presets; this comes with a forced full exit of the script after execution |

**Usage:** `--preset {name}` will allow for all of the arguments declared within the initial system arguments to be saved inside a JSON file and then reused.

**Note:** This lacks an updating/removal system for now, so it's a one-off.

## Encoding Settings

| Argument           | Type | Default | Description                       |
|--------------------|------|---------|-----------------------------------|
| `--encode_method`  | str  | "x264"  | Encoding method                   |
| `--custom_encoder` | str  | ""      | Custom FFmpeg encoding parameters |

# Encode Methods

### Standard Encoders
- `"x264"`: libx264, preset veryfast, CRF 15
- `"x264_10bit"`: libx264, preset veryfast, CRF 15, high10 profile
- `"x264_animation"`: libx264, preset veryfast, tune animation, CRF 15
- `"x264_animation_10bit"`: libx264, preset veryfast, tune animation, CRF 15, high10 profile
- `"x265"`: libx265, preset veryfast, CRF 15
- `"x265_10bit"`: libx265, preset veryfast, CRF 15, main10 profile
- `"h266"`: libvvenc, QP 24, preset 0

### Hardware Encoders
- `"nvenc_h264"`: h264_nvenc, preset p1, CQ 15
- `"nvenc_h265"`: hevc_nvenc, preset p1, CQ 15
- `"nvenc_h265_10bit"`: hevc_nvenc, preset p1, CQ 15, main10 profile
- `"nvenc_av1"`: av1_nvenc, preset p1, CQ 15
- `"qsv_h264"`: h264_qsv, preset veryfast, global quality 15
- `"qsv_h265"`: hevc_qsv, preset veryfast, global quality 15
- `"qsv_h265_10bit"`: hevc_qsv, preset veryfast, global quality 15, main10 profile
- `"qsv_vp9"`: vp9_qsv, preset veryfast, global quality 15
- `"h264_amf"`: h264_amf, quality speed, RC CQP, QP 15
- `"hevc_amf"`: hevc_amf, quality speed, RC CQP, QP 15
- `"hevc_amf_10bit"`: hevc_amf, quality speed, RC CQP, QP 15, main10 profile

### Other Formats
- `"av1"`: libsvtav1, preset 8, CRF 15
- `"prores"`: prores_ks, profile 4, qscale 15
- `"gif"`: gif, qscale 1, loop 0
- `"image"`: png, quality 1
- `"vp9"`: libvpx-vp9, preset veryfast, CRF 15

### High Quality Presets
- `"slow_x264"`: libx264, preset slow, CRF 18, high profile, level 4.1, tune animation, x264-params ref=4:bframes=8:b-adapt=2:direct=auto:me=umh:subme=10:merange=24:trellis=2:deblock=-1,-1:psy-rd=1.00,0.15:aq-strength=1.0:rc-lookahead=60, bf 3, g 250, keyint_min 25, sc_threshold 40, qcomp 0.6, qmin 10, qmax 51, maxrate 5000k, bufsize 10000k, movflags +faststart
- `"slow_x265"`: libx265, preset slow, CRF 18, profile main, level 5.1, tune ssim, x265-params ref=6:bframes=8:b-adapt=2:direct=auto:me=umh:subme=7:merange=57:rd=6:psy-rd=2.0:aq-mode=3:aq-strength=0.8:rc-lookahead=60, bf 4, g 250, keyint_min 25, sc_threshold 40, qcomp 0.7, qmin 10, qmax 51, maxrate 5000k, bufsize 10000k, movflags +faststart
- `"slow_av1"`: libsvtav1, preset 4, CRF 30, VBR, maxrate 5000k, bufsize 10000k, tile-columns 2, tile-rows 2, row-mt 1, movflags +faststart

**Note**: The output pixel format (`-pix_fmt`) can dynamically switch between yuv420p, yuv420p10le, and yuv444p10le based on `--bit_depth` and `--encode_method`.

## Video Stabilization

| Argument      | Type | Default | Description                |
|---------------|------|---------|----------------------------|
| `--stabilize` | flag | -       | Enable video stabilization |

## Performance Settings

| Argument         | Type | Default | Description                                                                    |
|------------------|------|---------|--------------------------------------------------------------------------------|
| `--half`         | flag | True    | Enable FP16 for improved performance                                           |
| `--buffer_limit` | int  | 50      | Set the limit for decoding and encoding buffers                                |
| `--static`       | flag | True    | Force static engine generation and bypass the arbitrary set limitations of TAS |

**Note**: Interpolation TRT, Depth TRT, Segment TRT all already are set to static engine generation due to set limitations within the models themselves.

## Usage Examples

**Note**: Replace `.\input\test.mp4` with a full or relative path to a video file or folder. The `--output` argument is optional, as the script can automatically generate a file name based on the input.

When using the packaged `main.exe` file, replace `python .\main.py` with `main.exe` in the following examples.

<h3>
<details>
<summary> 1. Enable upscaling </summary>

   ```sh
   python .\main.py --input .\input\test.mp4 --upscale
   ```
</details>
</h3>

<h3>
<details>
<summary> 2. Enable interpolation with Rife 4.6:</summary>

   ```sh
   python .\main.py --input .\input\test.mp4 --interpolate --interpolate_method rife4.6
   ```
</details>
</h3>

<h3>
<details>
<summary> 3. Enable upscaling and interpolation with custom models:</summary>

```sh
python .\main.py --input .\input\test.mp4 --upscale --upscale_method superultracompact --interpolate --interpolate_method rife4.20
```
</details>
</h3>

<h3>
<details>
<summary>4. Enable deduplication with custom sensitivity and method: </summary>

   ```sh
   python .\main.py --input .\input\test.mp4 --dedup --dedup_sens 60 --dedup_method ssim
   ```
</details>
<h3>

<h3>
<details>
<summary>5. Enable denoising using the NAFNet model:</summary>

   ```sh
   python .\main.py --input .\input\test.mp4 --denoise --denoise_method nafnet
   ```
</details>
</h3>

<h3>
<details>
<summary>6. Enable interpolation with scene change detection: </summary>

   ```sh
   python .\main.py --input .\input\test.mp4 --interpolate --scenechange
   ```
</details>
</h3>

<h3>
<details>
<summary>7. Generate depth maps using the "large" method and 16-bit workflow:</summary>

   ```sh
   python .\main.py --input .\input\test.mp4 --depth --depth_method large_v2 --bit_depth 16bit
   ```
</details>
</h3>

<h3>
<details>
<summary>8. Download a YouTube video, upscale with Span-TensorRT, interpolate with Rife 4.17-TensorRT, and use scene change detection:</summary>

   ```sh
   python .\main.py --input https://www.youtube.com/watch?v=dQw4w9WgXcQ --upscale --upscale_method span-tensorrt --interpolate --interpolate_method rife4.17-tensorrt --scenechange
   ```
   (Replace the YouTube URL with the actual video URL you want to download)
</details>
</h3>

<h3>
<details>
<summary>9. Process video from second 8 to 17 with custom settings:</summary>

   ```sh
   python .\main.py --input .\input\test.mp4 --inpoint 8 --outpoint 17 --upscale --upscale_method superultracompact-tensorrt --interpolate --interpolate_method rife4.20 --scenechange --scenechange_method maxxvit-tensorrt --custom_encoder "-c:v libsvtav1 -preset 8 -crf 30 -vf hqdn3d=1.5:1.5:6:6 -pix_fmt yuv420p"
   ```
</details>
</h3>

<h3>
<details>
<summary>10. Upscale, Interpolate, Scenechange, x264_animation_10bit and 16bit workflow example:</summary>

   ```sh
   python .\main.py --input https://www.youtube.com/watch?v=YRL74JmhVgk --upscale --upscale_method shufflecugan-tensorrt --interpolate --interpolate_method rife4.22-tensorrt --scenechange --scenechange_method maxxvit-tensorrt --encode_method x264_animation_10bit --bit_depth 16bit
   ```
</details>
</h3>
