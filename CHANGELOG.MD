
# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
 
## [Unreleased] - yyyy-mm-dd
 
## The Anime Scripter

### New
- Added Rife 4.20 CUDA and Tensorrt.
- Added Depth Anything V2 - DirectML.
- New variable named `--scenechange_method` with choices `maxvit-directml` & `maxvit-tensorrt` ( special thanks to @sudo for the models ).
- Added Prores encoding support

### Improvements
- Up to 15% faster Upscaling Performance when using TensorRT.
- Up to 20% faster Upscaling Performance hwen using CUDA.
- Up to 10% faster Interpolation Performance when using TensorRT.
- Up to 30% faster Interpolation Performance when using CUDA.
- Decoupled Scenechange from interpolation.
- Up to 20% faster SceneChange performance when using DirectML & TensorRT.
- Rife UHD Mode should allow for a bit more flexibility.
- Scenechange should no longer be reliant on the floating point precision of the interpolation process.
- More dynamic imports for faster startup times.
- Upgraded Torch version from 2.2.2 -> 2.4.0.
- Upgraded CUDA version from 12.1 -> 12.5.
 
### Fixed
- Issue [43](https://github.com/NevermindNilas/TheAnimeScripter/issues/43).
- Issue [41](https://github.com/NevermindNilas/TheAnimeScripter/issues/41).
- A couple of issues with scenechange caching.

## The Anime Scripter - Adobe Edition

### New
- TBA

### Improvements
- TBA

### Fixed
- TBA