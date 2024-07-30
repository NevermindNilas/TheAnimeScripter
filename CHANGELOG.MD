
# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
 
## [Unreleased] - yyyy-mm-dd
 
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

#### Improvements
- A lot of UI Changes to streamline the look of TAS and make it more compact ( still a WIP )
- The `About` Page has been changed into a Help Icon on the left Bottom.
- More Tooltips and informations on everything that is going on.
- A progress bar for when you download a new release of TAS ( still a WIP )
- The 'executable' is now ONLY 230kb on initial launch and it will soon allow for a more broader usecase with both a "Full" and "Lite" version of TAS.
- The dropdowns now have proper hinting and sectioning of which model is for which gpu architechture.
- Added new Models and Encoders to the UI.
- Improved the performance and reduced some jankyness from the backend.

#### Fixed
- Fixed a bug where inpoint / outpoint would go out of bounds.
- Fixed a bug where Youtube imports failed if no composition was selected.