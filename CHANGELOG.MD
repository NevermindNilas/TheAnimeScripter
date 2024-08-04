
# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
 

## [UNRELEASED] - yyyy-mm-dd

### The Anime Scripter

#### New
- Rife 4.20 TensorRT
- Rife 4.21 CUDA and TensorRT
- Added `--bit_depth` in order to enable higher precision workflows. Options [8bit, 16bit].
- .GIF input support using `--encode_method gif` - STILL A WIP!!!
- .png / .jpeg support using `--encode_method image` - STILL A WIP!!!
- A complete overhaul on how model downloading works, it is recommended to go to Appdata\Roaming\TheAnimeScripter\Weights and deleting the whole folder with this version
- Released before 1.9.1 but wasn't announced, TAS now has nightly builds at: https://github.com/NevermindNilas/TAS-Nightly
- Upgraded SPAN Models to `2x_Spanimation V1.5` - Special thanks to @TNTWise for the models.

#### Improvements
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

#### Removals:
- Rife4.15, Rife4.15-ncnn and Rife4.15-TensorRT

#### Fixes
- TensorRT artifactings when using Depth Maps.
- Depth Maps produced incostistent results with buggy and aliased outputs, TRT looked arguably worse than DirectML across the board.
- Scenechange Models have been renamed from " Maxvit " -> " Maxxvit " to better represent the real model.
- Fix issues with `--custom_encoder`

#### The Anime Scripter - Adobe Edition

#### New
- TBA

#### Improvements
- More UI Changes.

#### Fixed
- TBA

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