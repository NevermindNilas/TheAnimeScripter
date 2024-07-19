#####################################################
#               Available Parameters             #
#####################################################

## Overview:

The Anime Scripter is a powerful tool for enhancing and manipulating videos with a wide range of options. It supports functionalities such as interpolation, upscaling, deduplication, denoising, sharpening, scene change detection, depth map generation, and more. Users can customize the processing pipeline to meet their specific requirements using a comprehensive set of parameters.

### Command-Line Arguments:

- `--version` (action=store_true): Outputs the script version.
- `--benchmark` (action=store_true): Enable benchmarking. This will not output any video and it will only test the performance of the script without any additional processing.
- `--offline` (action=store_true): Enable offline mode, this will download all available models and store them locally in the weights folder, due to Github rate limiting and variables like internet connection this can take up to 5 minutes but it will make the script work 100% offline. ( except for YTDLP )
- `--ae` (actions=store_true, default=False): A simple flag for the script to know if it's ran from the After Effects interface or not.

#### Input and Output:

- `--input` (str): Absolute path of the input video.
- `--output` (str): Output string or absolute path for the processed video.
- `--audio` (action=store_true): Enable audio processing.

#### Upscaling Settings:

- `--upscale` (action=store_true): Enable upscaling.
- `--upscale_factor` (int, default=2): Factor for upscaling.
- `--upscale_method` (str, default="ShuffleCugan"): Upscaling method:
     - Options: "shufflecugan / shufflecugan-tensorrt", "compact / ultracompact / superultracompact / compact-directml / ultracompact-directml / superultracompact-directml / compact-tensorrt / ultracompact-tensorrt / superultracompact-tensorrt", "span / span-directml / span-tensorrt" 
- `--custom_model` (str, default=""): Choose a different model for supported upscaling architectures. Relies on `--upscaling_factor` and `--upscaling_method`. Input must be the full path to a desired .pth or .onnx file.
- `--upscale_skip` (action=store_true): Increase the perceived upscaling performance by not processing duplicates and instead reutilizing the previous frame. Don't trust with 100% certainty

#### Interpolation Settings:

- `--interpolate` (action=store_true): Enable interpolation.
- `--interpolate_factor` (int, default=2): Factor for interpolation.
- `--interpolate_method` (str, default="rife"): Interpolation method:
    - Options: "rife4.6 / rife4.6-ncnn / rife4.6-tensorrt", "rife4.15 / rife4.15-ncnn / rife4.15-tensorrt", "rife4.15-lite / rife4.15-lite-ncnn / rife4.15-lite-tensorrt ", "rife4.16-lite / rife4.16-lite-ncnn", "rife4.17 / rife4.17-tensorrt", "rife4.18 / rife4.18-tensorrt", "gmfss".
- `--ensemble` (action=store_true): Activate Ensemble for higher quality outputs from Rife.

#### Scene Change Settings:

- `--scenechange` (action=store_true, default=False): Enable scene change detection, useful for avoiding interpolation between different scenes of a clip / video in order to avoid artifacting. This only works if `--interpolate` is available, expect performance losses.
- `--scenechange_sens` (float, default=50): Change the sensitivity of the scene change detection, values go from a minimum of 0.80 - 0.90 ( equivalent to --scenechange_sens 0 - --scenechange_sens 100 )

#### Deduplication settings:

- `--dedup` (action=store_true): Enable deduplication.
- `--dedup_method` (str, default="ssim"): Deduplication method. Deduplication Methods: "ffmpeg, ssim, mse, ssim-cuda"
- `--dedup_sens` (float, default=50): Sensitivity of deduplication.
- `--sample_size` (int, default=224): Sample Size for SSIM Dedupping, it resizes the image to a 224 x 224 ( or your selected value ) image and does the similarity calculation. Higher values means lower performance but technically higher accuracy.

##### Denoising settings:

- `--denoise` (action=store_true): Enable denoising.
- `--denoise_method` (str, default="scunet"): Choose a denoising model:
     - Options: "scunet", "nafnet", "dpir"

#### Precision and Trimming:

- `--inpoint` (float, default=0): Inpoint for the video, used for accurate seeking based on Timecode variables, mainly used within the AE interface.
- `--outpoint` (float, default=0): Outpoint for the video.

#### Sharpening Settings:

- `--sharpen` (action=store_true): Enable sharpening.
- `--sharpen_sens` (float, default=50): Sensitivity of sharpening.

#### Segmentation Settings:

- `--segment` (action=store_true): Enable segmentation.
- `--segment_method` (str, default="anime", choices=["anime", "anime-tensorrt"])L Choose the desired segmentation method.

#### Auto Cut Clips Detection:

- `--autoclip` (action=store_true): Enable auto cut clip detection.
- `--autoclip_sens` (float, default=40): Sensitivity of auto cut clip detection.

#### Depth Map Generation:

- `--depth` (action=store_true): Generate Depth Maps.
- `--depth_method` (str, default="small"): Depth map generation method:
     - Options: "small / small-tensorrt / small-directml", "base / base-tensorrt / base-directml ", " large / large-tensorrt / large-directml".

#### Resizing:

- `--resize` (action=store_true): Enable resizing.
- `--resize_factor` (int, default=2): Factor for resizing the decoded video. Can also be a float value between 0 and 1 for downscaling.
- `--resize_method` (str, choices=["fast_bilinear", "bilinear", "bicubic", "experimental", "neighbor", "area", "bicublin", "gauss", "sinc", "lanczos", "spline", "spline16", "spline36", "point"], default="bicubic"): Resizing method:
    - Options: "lanczos" (recommended for upscaling), "area" (recommended for downscaling).

#### Encoding Settings:

- `--encode_method` (str, default="x264"): Encoding method:
    - Options: x264, x264_animation, x265, nvenc_h264, nvenc_h265, qsv_h264, qsv_h265, h264_amf, hevc_amf, prores.
- `--custom_encoder` (str, default=""): Comprehensive encoding option allowing users to specify their own `ffmpeg` parameters, including filters. Input a string of `ffmpeg` parameters, separated by spaces.

    Example: `--custom_encoder="-c:v libx264 -crf 14 -vf hqdn3d -pix_fmt yuv420"`

#### Performance Settings:

- `--half` (action=store_true, default=True): Enable FP16 which improves the performance, generally by about 50% but can technically result in lower quality results. Generally speaking it's worth using Fp16 for the faster inference.
- `--buffer_limit` (int, default=50): Set the limit for how many frames can be in the decoding and encoding buffers at any given time. Useful for when the encoding is the main bottleneck in the process but it does come at a cost of larger ram consumption.

## Manual Usage Examples:

Sidenotes:
- Replace `.\input\test.mp4` with a full or relative path to a video file or folder.
- `--output` is not necessary since the script can automatically create a file name for you based on the input's name.
- In case if you want to utilize the script using the packaged main.exe file, replace `python .\main.py` with `main.exe`

Example 1: Enable upscaling with half precision mode on. ( this will default to shufflecugan, also half precision mode is on by default so it's not necessary to type out )

- `python .\main.py --input .\input\test.mp4 --upscale --half `

Example 2: Enable Interpolating with custom interpolate method rife 4.6.

- `python .\main.py --input .\input\test.mp4 --interpolate --interpolate_method rife4.6`

Example 3: Enable upscaling and interpolating with custom models.

- `python .\main.py --input .\input\test.mp4 --upscale --upscale_method superultracompact --interpolate --interpolate_method rife4.15`

Example 4: Enable deduplication with a sensitivity of 60 and custom deduplication method.

- `python .\main.py --input .\input\test.mp4 --dedup --dedup_sens 60 --dedup_method ssim`

Example 5: Enable denoising using the nafnet model.

- `python .\main.py --input .\input\test.mp4 --denoise --denoise_method nafnet`

Example 6: Enable interpolation coupled with scene change detection.

- `python .\main.py --input .\input\test.mp4 --interpolate --scenechange` 

Example 7: Enable depth map generation using the "large" method.

- `python .\main.py --input .\input\test.mp4 --depth --depth_method large`

Example 8: Download a YouTube video, upscale it with span-tensorrt, interpolate it with rife4.17-tensorrt coupled with scene change detection.

- `python .\main.py --input https://www.youtube.com/watch?v=dQw4w9WgXcQ --upscale --upscale_method span-tensorrt --interpolate --interpolate_method rife4.17-tensorrt --scenechange`
- Remember to replace the YouTube URL with the actual URL of the video you want to download.
