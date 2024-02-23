#####################################################
#               Available Parameters             #
#####################################################

## Overview:

The Anime Scripter is a powerful tool for enhancing and manipulating videos with a wide range of options. It supports functionalities such as interpolation, upscaling, deduplication, denoising, sharpening, scene change detection, depth map generation, and more. Users can customize the processing pipeline to meet their specific requirements using a comprehensive set of parameters.

### Command-Line Arguments:

- `--version` (bool, action=store_true): Outputs the script version.

#### Input and Output:

- `--input` (str): Absolute path of the input video.
- `--output` (str): Output string or absolute path for the processed video.
- `--audio` (int, default=1): Enable audio processing (1) or disable (0).

#### Interpolation Settings:

- `--interpolate` (int, default=0): Enable interpolation (1) or disable (0).
- `--interpolate_factor` (int, default=2): Factor for interpolation.
- `--interpolate_method` (str, default="rife"): Interpolation method:
   - Options: "rife", "rife4.6", "rife4.14", "rife4.14-lite", "rife4.13-lite", "rife-ncnn", "rife4.6-ncnn", "rife4.14-ncnn", "rife4.14-lite-ncnn", "rife4.13-lite-ncnn".
- `--ensemble` (int, default=0): Activate Ensemble for higher quality outputs from Rife.

#### Upscaling Settings:

- `--upscale` (int, default=0): Enable upscaling (1) or disable (0).
- `--upscale_factor` (int, default=2): Factor for upscaling.
- `--upscale_method` (str, default="ShuffleCugan"): Upscaling method:
    - Options: "cugan / cugan-ncnn / shufflecugan", "swinir", "compact / ultracompact / superultracompact", "span / span-ncnn", "omnisr".
- `--cugan_kind` (str, default="no-denoise"): Cugan denoising kind:
    - Options: "no-denoise", "conservative", "denoise1x", "denoise2x".
- `--custom_model` (str, default=""): Choose a different model for supported upscaling architectures. Relies on `--upscaling_factor` and `--upscaling_method`. Input must be the full path to a desired .pth or .onnx file.

#### Deduplication settings:

- `--dedup` (int, default=0): Enable deduplication (1) or disable (0).
- `--dedup_method` (str, default="ffmpeg"): Deduplication method.
- `--dedup_sens` (float, default=50): Sensitivity of deduplication.

##### Denoising settings:

- `--denoise` (int, default=0): Enable denoising (1) or disable (0).

#### Precision and Trimming:

- `--half` (int, default=1): Use half precision (1) or full precision (0).
- `--inpoint` (float, default=0): Inpoint for the video.
- `--outpoint` (float, default=0): Outpoint for the video.

#### Sharpening Settings:

- `--sharpen` (int, default=0): Enable sharpening (1) or disable (0).
- `--sharpen_sens` (float, default=50): Sensitivity of sharpening.

#### Segmentation Settings:

- `--segment` (int, default=0): Enable segmentation (1) or disable (0).

#### Scene Change Detection:

- `--scenechange` (int, default=0): Enable scene change detection (1) or disable (0).
- `--scenechange_sens` (float, default=40): Sensitivity of scene change detection.

#### Depth Map Generation:

- `--depth` (int, default=0): Generate Depth Maps (1) or disable (0).
- `--depth_method` (str, default="small"): Depth map generation method:
    - Options: "small", "base", "large".

#### YouTube Video Download:

- `--ytdlp` (str, default=""): Download a YouTube video, needs a full URL.
- `--ytdlp_quality` (int, default=0): Allow 4k/8k videos to be downloaded then re-encoded to the selected `--encode_method` or `--custom_encoder`.

#### Resizing:

- `--resize` (int, choices=[0, 1], default=0): Enable resizing (1) or disable (0).
- `--resize_factor` (int, default=2): Factor for resizing the decoded video. Can also be a float value between 0 and 1 for downscaling.
- `--resize_method` (str, choices=["fast_bilinear", "bilinear", "bicubic", "experimental", "neighbor", "area", "bicublin", "gauss", "sinc", "lanczos", "spline", "spline16", "spline36", "point"], default="bicubic"): Resizing method:
    - Options: "lanczos" (recommended for upscaling), "area" (recommended for downscaling).

#### Encoding Settings:

- `--encode_method` (str, default="x264"): Encoding method:
    - Options: x264, x264_animation, x265, nvenc_h264, nvenc_h265, qsv_h264, qsv_h265, h264_amf, hevc_amf, vp9, qsv_vp9, prores.
- `--custom_encoder` (str, default=""): Comprehensive encoding option allowing users to specify their own `ffmpeg` parameters, including filters. Input a string of `ffmpeg` parameters, separated by spaces.

    Example: `--custom_encoder="-c:v libx264 -crf 14 -vf hqdn3d"`

   Note: Parameters specified with `--custom_encoder` will override the default encoding parameters. Ensure familiarity with `ffmpeg` and its parameters before using this option.

#### Performance Settings:
- `--nt` (int, default=1): Allocate the number of threads for the script to use. Relatively dodgy, use with caution.
- `--buffer_limit` (int, default=50): Set the limit for how many frames can be in the decoding and encoding buffers at any given time. This can be used to reduce memory usage at the cost of potential performance loss due to overhead. A value that is too high can lead to Out of Memory errors. Don't set this too high unless you have a lot of RAM to spare.
  
#####################################################
