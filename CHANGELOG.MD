
# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [UNRELEASED] - YYYY-MM-DD

### The Anime Scripter

#### New
- Added `H266` encoding support using command `--encode_method h266`.
- Added `slow_x264`, `slow_x265` & `slow_av1` encode methods.

#### Improvements
- Increase buffer size for encoder from `10 -> 20`
- Upgraded Github Actions runner from Python `3.12.4` -> `3.12.7`
- Avoid unnecessary memalocs for decode.
- `--upscale_skip` will now default to CUDA SSIM or CPU SSIM based on the selected `--upscale_method`
- `--upscale_skip` should now be significantly more reliable than before.
- Removed redunant calculations in MSE deduplication improving the performance by `~5%`

#### Notes
- `H266` encoding requires FFMPEG version `7.1`, if you have FFMPEG in System Path make sure the versions matches, otherwise if you downloaded it using TAS, make sure you delete the `ffmpeg` folder located at `%appdata%\Roaming\TheAnimeScripter\` and allow TAS to re-download the latest build.


## [1.9.10] - 2024-09-29

#### Fixed
- Issues where if `--inpoint` and `--outpoint` wasn't declared, FFMPEG would self-destruct.
- Issues where sharpening would go out of bounds relative to the max allowed sharpness from FFMPEG.


## [1.9.9] - 2024-09-28

### The Anime Scripter

#### New
- Enhanced encoding within TAS now allows real-time video processing and viewing with adequate hardware.
- Added `Rife 4.25` and `Rife 4.25-TensorRT`.
- Reintroduced `VP9` and `QSV_VP9` encoding methods to support `.webm` inputs.
- More accurate progress information post-process for comparison purposes.
- All executables now include precompiled `.pyc` code to improve startup times.
- Restructured interpolation pipeline for TensorRT, yielding up to `25%` performance gains.
- Full CUDA workflow now uses `cuda pinned memory workflow` to increase data transfer efficiency.
- Upgraded `Span` and `Span-TensorRT` models to `Spanimation-V2` - special thanks to @TNTWise.

#### Improved
- Rife TensorRT changes significantly reduce the `weights` folder size. It's recommended to delete the folder at `%appdata%\Roaming\TheAnimeScripter\weights` and let TAS regenerate all Rife Engines.
- Enhanced metadata retrieval powered by `pymediainfo`.
- Benchmarking now provides significantly more accurate results.
- Audio & Subtitle Merging now occurs during the encoding phase, reducing overall render time.
- Encoding is now fully parallel to inference, minimizing bottlenecks in the inference thread.
- Improved logging from FFMPEG.
- Enhanced CUDA and encoding handling results in a `30%` performance boost for NVIDIA users.
- TAS now attempts to create the folder automatically if `--output` is specified and the path doesn't exist.
- More robust Try/Except statements reduce code initialization time.
- Improved CUDA synchronization for better reliability.
- Renamed `ffmpegLog.txt` to `ffmpeg.log`.
- Renamed `log.txt` to `TAS.log`.
- System checks are now minimal and log only essential information.
- System checks are now conditional based on the `--benchmark` flag.
- Slight adjustments to `getFFMPEG.py`.
- Improved Upscale-Cuda Code.

#### Removed
- Removed `--audio` and related features in favor of `pymediainfo`'s automated detection.

#### Fixed
- Corrected exaggerated performance reports from TAS.
- Fixed a regression from 1.9.8 that significantly impacted encoding performance.
- Fixed an issue where the first frame of a processed video was black when upscaling.
- Fixed a regression from 1.9.8 where frames were not properly processed.

### Adobe Edition

#### New
- Added `Manual Method` button for a step-by-step tutorial on manually installing TAS' backend in case of download errors.

#### Improved
- Enhanced backend download functionality.

#### Fixed
- Fixed backend download for TAS (Counter: N/A).

#### Improvements
- Slight code adjustments for better memory efficiency, though benefits are likely minimal.


## [1.9.8] - 2024-09-15

### The Anime Scripter

#### NEW
- `--input` will now support:
  - `.txt` files with a list of videos
  - A full string with multiple videos: `"G:\TheAnimeScripter\input\input.mp4;G:\TheAnimeScripter\input\input.mp4"`
- Interpolation will now work prior to upscaling in order to increase the output's quality and to be more VRAM friendly.
- TAS now has proper multi subtitle and audio support.
- `--preset` & `--preset_list`: A new and semi-baked in preset configuration system with reusability in mind.

  **Usage:** `--preset {name}` will allow for all of the arguments declared within the initial system arguments to be saved inside a JSON file and then reused multiple times.

  **Note:** This lacks an updating/removal system for now, so it's a one-off.

  **Preset list:** Simply prints the available presets with their enabled options.
- TAS should now have proper CTRL + C support which extends to a graceful exit of the whole operation.

#### Improvements
- Upscale TRT should now be up to `10%` faster
- New coroutines for CUDA detection to improve performance.
- Decoupling of STDOUT from the main loop will now result in up to 50% performance gains for decoding. This should benefit all processes.
- Overall total model size is now significantly lower.
- TensorRT Handler functionality has been improved tremendously and is now significantly more robust and feature-rich than before (this will come in handy in the future).
- Rewrote some of the chain loop backend.
- Dedup with SSIM-CUDA is now significantly faster.
- SSIM CPU is now up to `2x` faster ( decode limited )
- MSE CPU is now `15%` faster.
- Upgraded Torch: `2.3.1` -> `2.4.1` for the added performance benefits.
- Linux lite builds should be significantly smaller now ( looking for testers & debuggers )
- Encoding should now be up to `50%` more efficient than before, internal results show `dedup ssim` speeds with encode going from `~280FPS` all the way to `~460-480FPS`at `1080P` with a 13700k.

#### Notes
- Currently TAS is slightly limited by STDOUT and STDIN from FFMPEG. I will look to further improve this down the line.
- With the current changes in encoding, I can reasonable argue that CPU and subprocessing are starting to become decent bottlenecks.

#### Regressions
- With the new changes in Decode/Encode, RAM usage will see a hit, I've seen up to `~500mb` more usage in RAM, though that's whilst considering that TAS uses roughly `2.5GB` during a reasonable workload.

#### Removals
- `Scikit-Image` package.

### TAS - Adobe Edition

#### New
- `Enable Preview Windows` checkbox:
  - This will utilize the `--preview` functionality within TAS to display a live preview of the progress, note this comes with a tax in performance.
- New Options for `Remove Background`:
  - `TensorRT` acceleration has been added to segmentation for further boost in performance.
- `Pre-Render` will be even more flexible:
  - The preview functionality within TAS was and still is fairly limited, but with this release you will have a bit more control over the desired output. You will now be able to choose between two baked in functionalities, `Lossless` ( the current default ) and `Quicktime`. For most usecases `Quicktime` is plenty good and there's no need to choose Lossless but feel free to mess around with it.

#### Improvements
- Some UI Cleanup of features that are no longer planed to be added in the near future.
- Improved some background blur related things.
- Since TAS now comes with proper `CTRL + C` support, you can now exit the process gracefully and not worry about annoying alerts and errors.

#### Notes
- Preview is still under active development and I am planning to further improve on it in the near future, but attention will now shift towards the Standalone edition which is about 80-90% of the way done.

## [1.9.7] - 2024-09-02

### The Anime Scripter

#### NEW
- Added Rife4.20-NCNN, 4.21-NCNN, 4.22-NCNN, 4.22-Lite-NCNN

#### Fixes
- CUDA errors seen in the Lite Edition of TAS for Windows.

#### Regressions:
- Some performance gains found in 1.9.6 have been lost due to errors in dependency handling.

### Adobe Edition

#### New
- Added new NCNN Models

#### Improvements:
- TAS backend downloads will now be up to `2300%` faster.
- Fix some race condition issues found in the backend.
- Made sure for 7z to overwrite existing TAS installations
- Improve path handling
- Better error handling.
- In case of download errors TAS should now attempt to redownload the backend.

#### Removals
- Removed some old 4.15-lite & 4.17 models in favor of newer and more performant options.

## NOTES:
- TAS Full is meant for NVIDIA users who want to benefit from CUDA and TensorRT Accelerations.
- TAS Lite is meant for AMD / Intel users only.

## [1.9.6] - 2024-09-02

### The Anime Scripter

##### NEW
- Added `Rife4.22-Lite` and `Rife4.22-Lite-Tensorrt`
- Added `--static` flag to force static engine generation for Upscale TensorRT.
- Temp state for downloaded models will now ensure file integrity in case of internet loss.
- Added `--depth_quality` flag with options `low, high` ( default high ). For the users with lower end system to benefit of higher inference speeds. ( Not reliable for TensorRT )
- Added testing builds for linux-full and linux-lite ( Full is WIP but should be functional with the build-linux.py script )

##### Improvements
- Upgraded base polygraphy from 0.49.12 -> 0.49.13
- Removed unnecessary clamps.
- The metadata, progress bar and everything else related to video details should be MUCH more accurate now, including logging.
- Log.txt should now be more compact with `Arguments` no longer having unused arguments.
- Rife TRT will only build Static Engines in order to improve stability and versatility.
- Rife CUDA is now up to 15% faster than before.
- Rife TRT is now up to 15% faster than before.
- Chained Processes with Upscaling and Interpolation TRT / CUDA will now be more memory efficient and performant.
- Benchmarks are now going to be more accurate for interpolation.
- Upgraded ONNXRuntime-DirectML to 1.19.1
- Improved the progress bar with more information and less unnecessary eye candy.
- Improved stability of image inputs.
- More video metadata requests.
- The download progress bar will now also tell the download speed in MB/s
- Some slight adjustments to the building methodology of TAS.

#### Fixes:
- Some CUDA Race conditions could be met in extremely High FPS workflows ( 500+ FPS ).
- Fix some oversights in Rife 4.22 and Rife 4.22-lite that were reducing the output quality.

#### Adobe Edition

#### NEW
- Added `Rife4.22-Lite` and `Rife4.22-Lite-Tensorrt` options to the dropdowns.
- Added Depth Quality options ( low and high )

#### Fixes
- Fixed issues when paths had spaces in them.

## [1.9.5] - 2024-08-17

### The Anime Scripter
N/A

### Adobe Edition

#### Fixes
- Fixed issues with `Full` or `Lite` Method selection in model download.

## [1.9.4] - 2024-08-17

### The Anime Scripter

#### New
- A new 3D like Banner :)
- Added OpenProteus and AniScale2 by Sirosky
- New lite version of TAS ( WIP )
- Support for multi audio and multi subtitle extraction and addition.
- Added ShuffleSpan by Sudo ( WIP )
- Added `Differential` | `Differential-TensorRT` ( My own attempts ) at a scenechange method ( WIP )
- Added `Shift_LPIPS-TensorRT` Scenechange Methods by Sudo.
- New `--hide_banner` flag to Hide TAS' banner when processing.
- Preview functionality with the command `--preview`, use port `127.0.0.1:5050/frame` in order to check the results of TAS.

#### Improvements
- Improved parameters and examples.
- Upgraded to nightly ONNXRuntime DirectML in order to avoid IR issues.
- Added recursive checks for Model Downloads.
- Significantly better I/O folder handling, this should result in better edge case handling.

#### Fixes
- Fixed a bug with 16bit workflow and x264_animation_10bit
- [Issue #44](https://github.com/NevermindNilas/TheAnimeScripter/issues/45)
- Fixes issues with input folder and output folder cases.
- [Issue #46](https://github.com/NevermindNilas/TheAnimeScripter/issues/46) - Fixed concurrency issues associated with Rife TensorRT and high cpu usages.
- Fixed an issues with span-directml where it would point to a different host.

#### Regressions
- Temporarely reverted back to pytorch 2.2.2 and cuda 12.1 until .dll issues are fixed, this may result in slight performance losses.

#### Removals
- Updater.exe

### Adobe Edition

#### New
- TAS will now dynamically switch between "Full" or "Lite" mode depending on your available GPU.

#### Improved
- Improved Error handling.
- The terminal ( command prompt ) will now start minimized in order to not disrupt your workflow.
- Immense improvements on output handling, TAS will now selectively create and choose specific types of folders based on your current process. This also comes with nice text formatting for the output with keywords like "_DE" for Deduplication and "_SH" for Sharpen.
- Better download functionality with a possibility in increased max download speed.

#### Bug Fixes
- For some users the download would often hang, this should now have been remedied or at least improved and TAS should now attempt to reconnect in case of a error connection.

## [1.9.3] - 2024-08-09

### The Anime Scripter

#### New
- Added Rife4.22 TensorRT
- Added Video Stabilization using Vidgear

#### Improvements

#### Removals

#### Fixed

### The Anime Scripter - Adobe Edition

#### New
- TBA

#### Improvements
- TBA


## [1.9.2] - 2024-08-09

### The Anime Scripter

#### New
- !Upgraded to Python 3.12!
- Rife 4.22 CUDA
- Rife 4.21 CUDA and TensorRT
- Rife 4.20 TensorRT
- Added `--bit_depth` in order to enable higher precision workflows. Options [8bit, 16bit].
- .GIF input support using `--encode_method gif` - STILL A WIP!!!
- .png / .jpeg support using `--encode_method image` - STILL A WIP!!!
- A complete overhaul on how model downloading works, it is recommended to go to Appdata\Roaming\TheAnimeScripter\Weights and deleting the whole folder with this version
- Released before 1.9.1 but wasn't announced, TAS now has nightly builds at: https://github.com/NevermindNilas/TAS-Nightly
- Upgraded SPAN Models to `2x_Spanimation V1.5` - Special thanks to @TNTWise for the models.

#### Improvements
- Up to 25% faster Rife TensorRT Inference performance.
- Better TensorRT Engine Handling in case if the engine / driver is outdated.
- Depth Maps should now produce significantly better outputs compared to previous releases at a roughly 5x deduction in performance.
- Made sure the output is now always clamped
- `--ofline` can now be dynamically selected to download a specific model or list of models based on user selection. Use keyword `all` to download all of the available models.
- The number of models have been drastically cut
- TensorRT Engine workflow should now be drastically more robust than previously with better edge case handling.
- Revised documentation in Parameters.md
- Through a few 'optimizations' in model downloads, the total weights folder should be ~1.5GB lower.
- NCNN Upscaling models should now be dynamically downloaded based on user request
- Marginal performance boosts for Upscale NCNN
- Denoising is up to 2x faster now.
- Prebuilds will now be 100-150mb smaller.
- Audio Merging is now practically instantenous :)
- Compatability with vp9 and .webm processes
- Couple of changes to warplayer should have made it marginally faster and marginally more memory efficient.
- Decoding is now up to 15% faster.
- Deduplication is now up to 10% faster
- Sharpening is now up to 15% faster
- Benchmark will now use a selectable list for deciding the prefered GPU.

#### Removals:
- Rife4.15, Rife4.15-ncnn and Rife4.15-TensorRT

#### Fixes
- TensorRT artifactings when using Depth Maps.
- Depth Maps produced incostistent results with buggy and aliased outputs, TRT looked arguably worse than DirectML across the board.
- Scenechange Models have been renamed from " Maxvit " -> " Maxxvit " to better represent the real model.
- Fix issues with `--custom_encoder`
- Partial fix for issue [44](https://github.com/NevermindNilas/TheAnimeScripter/issues/44)
- TAS' ytdlp should now store the downloads in the output folder of main.exe in case if `--output` is not defined.
- Fixed some bugs / issues with Pathing for Linux.

#### The Anime Scripter - Adobe Edition

#### New
- Enabled auto clip functionality.
- Half-Automated TAS updates.
- `--bit_depth` has been added to the encoding settings.

#### Improvements
- More UI Changes.
- Simplified some backend functionality for better performance

#### Fixed
- Some weird scaling issues.
- TAS Update popping up randomly at times and closing.

## [1.9.1] - 2024-07-30

### The Anime Scripter

#### New
- Added Rife 4.20 CUDA.
- Added Depth Anything V2 - DirectML.
- New variable named `--scenechange_method` with choices `maxvit-directml` & `maxvit-tensorrt` ( special thanks to @sudo for the models ).
- ProRes 444 HQ Encoder ( `--encode_method prores` )
- Added Prores encoding support

#### Improvements
- Up to 15% faster Upscaling Performance when using TensorRT.
- Up to 20% faster Upscaling Performance when using CUDA.
- Up to 10% faster Interpolation Performance when using TensorRT.
- Up to 30% faster Interpolation Performance when using CUDA.
- Decoupled Scenechange from interpolation.
- Up to 20% faster SceneChange performance when using DirectML & TensorRT.
- Rife UHD Mode should allow for a bit more flexibility.
- Scenechange should no longer be reliant on the floating point precision of the interpolation process.
- More dynamic imports for faster startup times.
- Upgraded Torch version from 2.2.2 -> 2.4.0.
- Upgraded CUDA version from 12.1 -> 12.5.
- Reformated and cleaned up the code.
- `--help` should display a bit nicer now compared to previous releases and it should have more `helpers` added to them.

#### Fixed
- Issue [43](https://github.com/NevermindNilas/TheAnimeScripter/issues/43).
- Issue [41](https://github.com/NevermindNilas/TheAnimeScripter/issues/41).
- A couple of issues with scenechange caching.
- Fix issues with benchmark.py
- DLL missing with onnxruntime-directml

#### The Anime Scripter - Adobe Edition

#### New
- New installation method with code signing for a more secure and robust installation method.
- Dynamic Downloading and Updating of TAS.
- Pre-Render now works with more than 1 active / selected layer

#### Improvements
- A lot of UI Changes to streamline the look of TAS and make it more compact ( still a WIP )
- The `About` Page has been changed into a Help Icon on the left Bottom.
- More Tooltips and informations on everything that is going on.
- A progress bar for when you download a new release of TAS ( still a WIP )
- The 'executable' is now ONLY 230kb on initial launch and it will soon allow for a more broader usecase with both a "Full" and "Lite" version of TAS.
- The dropdowns now have proper hinting and sectioning of which model is for which gpu architechture.
- Added new Models and Encoders to the UI.
- Improved the performance and reduced some jankyness from the backend.
- Changed preset template from lossless to highquality to reduce overall space for rendering

#### Fixed
- Fixed a bug where inpoint / outpoint would go out of bounds.
- Fixed a bug where Youtube imports failed if no composition was selected.
- Fixed a bug where pre-render would not not adjust the work area properly before processing.
