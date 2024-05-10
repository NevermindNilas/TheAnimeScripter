#####################################################
#               Available Parameters             #
#####################################################

## Overview:

The Anime Scripter is a powerful tool for enhancing and manipulating videos with a wide range of options. It supports functionalities such as interpolation, upscaling, deduplication, denoising, sharpening, scene change detection, depth map generation, and more. Users can customize the processing pipeline to meet their specific requirements using a comprehensive set of parameters.

### Command-Line Arguments:

- `--version` (bool, action=store_true): Outputs the script version.
- `--benchmark` (int, default=0): Enable benchmarking. This will not output any video and it will only test the performance of the script without any aditional processing.
- `--offline` (int, defualt=0): Enable offline mode, this will download all available models and store them locally in the weights folder, due to Github rate limiting and variables like internet connection this can take up to 5 minutes but it will make the script work 100% offline. ( except for YTDLP )
- `--consent` (int, default=0): WIP! Enable fully anonymous logging of usage, this will store data from log.txt and send it to my server, the only data that is being processed from log.txt are: " Arguments " ( excluding input/output ), " System Specs " and " Processing Outputs ". This can help provide better insight on usage and errors.

#### Input and Output:

- `--input` (str): Absolute path of the input video.
- `--output` (str): Output string or absolute path for the processed video.
- `--audio` (int, default=1): Enable audio processing (1) or disable (0).

#### Upscaling Settings:

- `--upscale` (int, default=0): Enable upscaling (1) or disable (0).
- `--upscale_factor` (int, default=2): Factor for upscaling.
- `--upscale_method` (str, default="ShuffleCugan"): Upscaling method:
     - Options: "cugan / shufflecugan / shufflecugan-tensorrt", "compact / ultracompact / superultracompact / compact-directml / ultracompact-directml / superultracompact-directml / compact-tensorrt / ultracompact-tensorrt / superultracompact-tensorrt", "span / span-directml / span-tensorrt", "omnisr", "realesrgan". 
- `--custom_model` (str, default=""): Choose a different model for supported upscaling architectures. Relies on `--upscaling_factor` and `--upscaling_method`. Input must be the full path to a desired .pth or .onnx file.
#### Interpolation Settings:


- `--interpolate` (int, default=0): Enable interpolation (1) or disable (0).
- `--interpolate_factor` (int, default=2): Factor for interpolation.
- `--interpolate_method` (str, default="rife"): Interpolation method:
    - Options: "rife", "rife4.6", "rife4.14", "rife4.15", "rife4.15-lite", "rife-ncnn", "rife4.6-ncnn", "rife4.15-ncnn", "rife4.15-lite-ncnn", "rife4.16-lite-ncnn" "rife4.6-tensorrt", "rife4.14-tensorrt", "rife4.15-tensorrt", "gmfss".
    - Notation: `rife`, `rife-ncnn`, "rife-directml" will default to the latest available version.
- `--ensemble` (int, default=0): Activate Ensemble for higher quality outputs from Rife.

#### Deduplication settings:

- `--dedup` (int, default=0): Enable deduplication (1) or disable (0).
- `--dedup_method` (str, default="ffmpeg"): Deduplication method. Deduplication Methods: "ffmpeg, ssim"
- `--dedup_sens` (float, default=50): Sensitivity of deduplication.
- `--sample_size` (int, default=32): Sample Size for SSIM Dedupping, it resizes the image to a 32 x 32 ( or your selected value ) image and does the similarity calculation. Higher values means lower performance but technically higher accuracy.

##### Denoising settings:

- `--denoise` (int, default=0): Enable denoising (1) or disable (0).
- `--denoise_method` (str, default="scunet"): Choose a denoising model:
     - Options: "scunet", "nafnet", "span", "dpir"

#### Precision and Trimming:

- `--inpoint` (float, default=0): Inpoint for the video, used for accurate seeking based on Timecode variables, mainly used within the AE interface.
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


#### Resizing:

- `--resize` (int, choices=[0, 1], default=0): Enable resizing (1) or disable (0).
- `--resize_factor` (int, default=2): Factor for resizing the decoded video. Can also be a float value between 0 and 1 for downscaling.
- `--resize_method` (str, choices=["fast_bilinear", "bilinear", "bicubic", "experimental", "neighbor", "area", "bicublin", "gauss", "sinc", "lanczos", "spline", "spline16", "spline36", "point"], default="bicubic"): Resizing method:
    - Options: "lanczos" (recommended for upscaling), "area" (recommended for downscaling).

#### Encoding Settings:

- `--encode_method` (str, default="x264"): Encoding method:
    - Options: x264, x264_animation, x265, nvenc_h264, nvenc_h265, qsv_h264, qsv_h265, h264_amf, hevc_amf, vp9, qsv_vp9, prores.
- `--custom_encoder` (str, default=""): Comprehensive encoding option allowing users to specify their own `ffmpeg` parameters, including filters. Input a string of `ffmpeg` parameters, separated by spaces.

    Example: `--custom_encoder="-c:v libx264 -crf 14 -vf hqdn3d -pix_fmt yuv420"`

#### Performance Settings:

- `--half` (int, default=1): (1) for fp16 ,(0) for fp32. FP16 improves the performance, generally by about 50% but can technically result in lower quality results. Generally speaking it's worth using Fp16 for the faster inference.
- `--nt` (int, default=1): Allocate the number of threads for the script to use. Relatively dodgy, use with caution.
- `--buffer_limit` (int, default=50): Set the limit for how many frames can be in the decoding and encoding buffers at any given time. This can be used to reduce memory usage at the cost of potential performance loss due to overhead. A value that is too high can lead to Out of Memory errors ( OOM ). Don't set this too high unless you have a lot of RAM to spare.

#####################################################

## Examples:

Sidenotes:
- Replace `.\input\test.mp4` with a full or relative path to a video file or folder.
- `--output` is not necessary since the script can automatically create a file name for you based on the input's name.
- In case if you want to utilize the script using the packaged main.exe file, replace `python .\main.py` with `main.exe`

Example 1: Enable upscaling with half precision mode on. ( this will default to shufflecugan, also half precision mode is on by default so it's not necessary to type out )

- `python .\main.py --input .\input\test.mp4 --upscale 1 --half 1`

Example 2: Enable Interpolating with custom interpolate method rife 4.6.

- `python .\main.py --input .\input\test.mp4 --interpolate 1 --interpolate_method rife4.6`

Example 3: Enable upscaling and interpolating with custom models.

- `python .\main.py --input .\input\test.mp4 --upscale 1 --upscale_method superultracompact --interpolate 1 --interpolate_method rife4.15`

Example 4: Enable deduplication with a sensitivity of 60 and custom deduplication method.

- `python .\main.py --input .\input\test.mp4 --dedup 1 --dedup_sens 60 --dedup_method ssim`

Example 5: Enable denoising using the nafnet model.

- `python .\main.py --input .\input\test.mp4 --denoise 1 --denoise_method nafnet`

Example 6: Enable scene change detection with a sensitivity of 50.

- `python .\main.py --input .\input\test.mp4 --scenechange 1 --scenechange_sens 50`

Example 7: Enable depth map generation using the "large" method.

- `python .\main.py --input .\input\test.mp4 --depth 1 --depth_method large`

Example 8: Download a YouTube video and then upscale it with shufflecugan.

- `python .\main.py --input https://www.youtube.com/watch?v=dQw4w9WgXcQ --upscale 1 --upscale_method shufflecugan`
- Remember to replace the YouTube URL with the actual URL of the video you want to download.
