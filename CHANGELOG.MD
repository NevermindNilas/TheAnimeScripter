# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [2.5.2] - 2025-12-24
### Hotfix
- Fixed wrong wheel for VMAF-Torch


## [2.5.1] - 2025-12-24
### Hotfix
- Fixed broken links

## [2.5.0] - 2025-12-24
### IMPORTANT NOTES:
- N/A

### Added
- Added `distilldrba`, `distilldrba-lite`, `distilldrba-lite-tensorrt` interpolate methods.
- Added `vmaf` & `vmaf-cuda` dedup method.
- Added my very first AI Model, Saryn with options: `saryn`, `saryn-tensorrt` & `saryn-directml`.

### Changed
- Improved output handling and name generation.
- Removed `pytorch-lightning` from the requirements.
- All Depth OG models will now use CELUX for decoding for better performance and better output quality.

### Fixed
- Fixed an issue where if the progress bar contained non UTF-8 compliant text, it would result in a crash.
- Better text handling in the terminal.

## Adobe Edition
### New
- Added a brand new Queue System to Adobe Edition
- Added all the new upscale, interpolate and dedup methods.
- Added Alpha support for Layer inpoint and outpoint effectively bypassing the pre-render step.

### Changed
- Rewrote a significat amount of the code.
- Improved many elements of the UI.
  
### Fixed
- Errors with access violation 3221225477.

## [2.4.9] - 2025-12-01
### IMPORTANT NOTES:
- I have taken over the development of [Celux](https://github.com/NevermindNilas/CeLux), the primary decoding backend for TAS. In order to improve output quality I have had to migrate many of its core color conversion logic from `swscale` to `libyuv` as well as rework some of its internal pipinelines. This change was non-trivial and performance may be affected in most of the workflows for a significantly better I/O handling and quality. All notable changes will be in that repo. Production of TAS will see a major slowdown now that I have single handedly maintain 3 repositories ( TAS, TAS-AdobeEdition & Celux ) so please be patient with updates.

## The Anime Scripter
### Added
- Added `av1` decoding support through `libdav1d` in Celux.

### Changed
- A significant rework of the Dependency Handler.

### Fixed
- An issue where certain dependencies would not be properly uninstalled when using `--download_requirements`.

## Adobe Edition
### Added
- Graph Editor is now out of Early Alpha, with improved stability, responsiveness and usability.
- Graph Editor now has a `.flow` import option in order to easily migrate from other plugins.
- Added a way in which the nodes can stick to the walls while holding shift in the Graph Editor.
- Double clicking the nodes will reset them to their default position in the Graph Editor.
- Added a final fallback to the pre-render file seeking logic which will now look for the latest file in the TAS Pre-Renders folder if all else fails.
- Added a settings tab and moved all advanced settings there.
- Added a "What's New" disclosure to the About Tab which will show the latest changes in each version.

### Changed
- Minor UI tweaks.
- Merged the extra tab with the layers tab.

## [2.4.8] - 2025-11-15
### Added
- Added an auto-refresh feature to the `--preview` functionality.
- 3 new restore models: `hurrdeblur`, `hurrdeblur-tensorrt`, `hurrdeblur-directml` by Zarxrax
- Added restore methods chaining through `--restore_method model model` option.
- Added command `--json` to pass arguments through a .json file, this is primarily used for the Adobe Edition but can be used in CLI as well.

### Changed
- Added back flask and related dependencies.

### Fixed
- Issues with rife onnx exports.
- Possible cuda race condition when peeking future frames.
- Fixed an issue with fastlinedarken + fastlinedarken-tensorrt.

## Adobe Edition
### Added
- AnimeSR support.
- Hurrdeblur support.
- Default presets to the chain tab to ease the first time user experience.
- Added support for restore methods chaining through the UI.

### Changed
- Improved the Graph Tab performance and responsiveness.
- Removed some hardly used features from the graph editor.
- Added a search for preset option in the Graph Editor.
- Migrated to a --json approach for passing arguments from the Adobe Edition to TAS backend for better path safety.


## [2.4.7] - 2025-11-02
### Added
- Added `AnimeSR` support with `animesr` & `animesr-tensorrt` commands ( DirectML support may be added in the future altho' performance may be subpar )

### Changed
- UX improvements to the YTDLP logic.
- Completely rewrote the `--preview` functionality for better performance and lower overhead, it should now use up at most `0.5-1%` of the CPU rather than up to 5-10% previously.

### Fixed
- Issues with `--preview` not showing the actual output frame but instead the input frame.
- Temporarily reverted the `tensorrt` version in order to fix cuda-runtime issues.

### Removed
- Flask and related dependencies.



## Adobe Edition
### Added
- Added a `Reinstall TAS` button to the extra TAB.

### Changed
- Updated dependencies to their latest versions.

### Fixed
- N/A

## [2.4.6] - 2025-10-24
### Added
- As per community request, I've `Depth Anything V2 Giant` model support with commands:
- `giant_v2` & `og_giant_v2`, TRT and DirectML support may be added in the future depending on demand.

### Changed
- Removed some unused warnings.

### Fixed
- Missing dependency `onnxscript`.

## Adobe Edition
### Added
- `Depth Anything V2 Giant` model support in the UI.
- Quick presets options to the `Chain Tab` which also includes self made presets. 

### Changed
- Defaulted / Reverted back to `PreRender_XXXXXX` format for pre-renders.
- Some small UI tweaks.

## [2.4.5] - 2025-10-18
### Added
- FFMPEG build logging.
- A more robust dependency handler which should allow for a safe fallback for older NVIDIA GPUs ( GTX 1000 series and older )
- Added 6 new restore methods:
  - `deh264_real`, `deh264_real-tensorrt`, `deh264_real-directml` by @Phhofm
  - `deh264_span`, `deh264_span-tensorrt`, `deh264_span-directml` by @tntwise

### Changed
- Improved the robustness of the GPU detection logic.
- The arguments sections should only show the actually used arguments now.
- Bumped python to `v3.13.9`
- Bumped `torch` to `2.9.0`
- Migrated back from `.7z` to `.zip` for packaging in order to simplify the build process.

### Fixed
- Fixed a bug where certain unlucky users ( Non-NVidia ) would download the wrong dependencies.

## Adobe Edition

### Added
- Added `deh264_real` and `deh264_span` restore methods.
- Added a " Github Sponsors " button.
- Shortcuts options for the whole UI including settings in the `About` Tab.

### Changed
- Updated some descriptions in the UI for better clarity.
- Updated all of the dependencies to their latest versions.

### Fixed
- Potential fix for `.mov` files missing.

## [2.4.4] - 2025-09-18
### Added
- New upscaling methods: `fallin_soft`, `fallin_strong` with TensorRT and DirectML support by @renarchi

### Changed
- Bumped Python to `3.13.7`

## Adobe Edition

### New
- Added `Fallin Soft` and `Fallin Strong` to the Upscale Methods dropdown.
- Added `Trim to Work Area` button which automatically trims all selected layers ( incl comps ) to the duration of the work area.
- Added `Freeze on Current Frame` button which automatically freezes the layers ( incl comps ) to the desired frame.

### Changed
- Updated all dependencies to their latest versions.
- Made `Fallin Soft` as the default upscaling model.
- Minor UI tweaks.
- Removed Span from the upscale drodown.
- Removed the "NO" button from the Dependency Downloader of TAS as well as changing some logic where it will interchangeably use "Update" or "Download" based on the current state of the dependencies.

## [2.4.3.] - 2025-09-02
### Changed
- Reverted back to the old progress bar logic due to some issues with the new one.

## [2.4.2] - 2025-09-02
### Fixed
- Fixed an issue where ffmpeg would fail to download due to Charmap issues in Adobe Edition
- Theoretically fixed an issue for AMD Users where certain variables would not be defined.


## [2.4.1] - 2025-08-31
### Fixed
- An issue with appdata path seeking.

## [2.4.0] - 2025-08-31
### WARNING
- TAS is now using pytorch 2.8.0 + cuda 12.9, this means that older GPUs ( Maxwell, Pascal, and Volta ) with capabilities <= sm 70 have been fully deprecated and will no longer be supported. This means that TAS will no longer work on these GPUs, if you are using a GPU with sm 70 or lower, please stick to TAS 2.3.8 or earlier.
- With the migration to Celux, I strongly recommend deleting previous installations of TAS and reinstalling the latest version to avoid any potential issues with the new decoding backend.

### New
- Migrated, yet again, from Vapoursynth to `celux` for decoding, this should result in better performance and lower CPU usage.
- Added `--output_scale` parameter for custom output resolution scaling (e.g., `--output_scale 2560x1440`)
  - Supports any resolution in WIDTHxHEIGHT format (e.g., 2560x1440, 3840x2160, 1920x1080)
  - Uses bilinear scaling for minimal artifacts
  - Applied during FFmpeg encoding for optimal performance
  - Works with all processing methods (upscaling, interpolation, etc.)
- Added a lightweight dependency check that doesn’t import heavy libraries
  - Uses `importlib.metadata.version` to see if a package is installed
  - Uses `importlib.util.find_spec` to see if the module is discoverable
  - Prints a simple list like `Missing distributions detected: ['tensorrt']` before installing
- Added a small map for common package→module name differences
  - Example: `opencv-python` → `cv2`, `onnxruntime-gpu` → `onnxruntime`
- Added report of download speed to the progress bar.

#### Changed
- Improvements of up to `25%` in encoding bottlenecked workflows.
- Improvements of up to `10%` in Cuda/TensorRT workflows, partly due to the new torch/tensorrt updates.
- Dependency check flow is faster and simpler
  - Instead of importing `torch` just to “test it”, we now check presence via metadata
  - Only runs `pip install` when something is actually missing or mismatched
- Speed up `pip` calls
  - Added `--disable-pip-version-check` to `pip install`/`pip uninstall` to shave startup time
- Made the forced download helper more robust
  - Avoids referencing `isNvidia` when a specific requirements file is provided
- Slight modifications to imports to further tune start-up times.
- Upgraded to python `3.13.6`
- Up to `10%` faster GMFSS performance.
- Redesigned the progress bar.
- Added supression of the flask banner.

#### Fixed
- Fixed an issue with distill-directml depth methods where the output was not being properly handled.
- Fixed an issue with distill-directml depth methods where fp32 would crash.

## Adobe Edition
### New
- Added `Xpress 8k` compression method to the TAS-Portable folder resulting in up to a 50% reduction in final binary size, highly experimental and not recommended for everyone.
- An attempt to solve an issue where Pre-Renders would not work properly in the Adobe Edition due to different UI language.
- Added Resize Output option to the Adobe Edition, this will resize the output to the specified resolution, this is primarily used for upscaling workflows.
- Added back `ProRes` Encoding.
- Added breadcrumbs to the `Chain` tab to showcase each step the Input takes.

### Changed
- Improved the design of the `LogTab` and `AboutTab`

### Fixed
- Fixed an issue where the screenshot functionality would not parse 3rd party plugins properly.
- Fixed an issue where the ProRes necessary extension ( .mov ) wasn't properly added to the output file.

## [2.3.8] - 2025-08-03
#### Changed
- Updated some of the imports in the depth methods to be more efficient 

#### Fixed
- Missing `einops` dependency.

### Adobe Edition

#### Changed
- Update the UI, making it more responsive, compact and easier to use.
- Added UTF-8 encoding to the Adobe Edition to prevent issues with non-ASCII characters in paths ( WIP fix, not 100% yet ).
- Added an Error Toast when the composition width or height is non divisible by 2.

#### Fixed
- Fixed an issue where the progress bar would not update properly when using the Adobe Edition.

## [2.3.7] - 2025-07-30
#### Added
- Download update to the AdobeEdition progress bar.
  
#### Changed
- N/A

### Adobe Edition

#### Fixed
- Fixed an issue where the copy logs button would not work properly in the Adobe Edition.


## [2.3.6] - 2025-07-28
### AdobeEdition
#### Fixed
- String handling for output generation.
- Graph tab would crash the UI ( WIP fix, not 100%

## [2.3.5] - 2025-07-28
#### New
- Added the following depth methods:
  - `distill_small_v2-tensorrt`
  - `distill_base_v2-tensorrt`
  - `distill_large_v2-tensorrt`
  - `distill_small_v2-directml`
  - `distill_base_v2-directml`
  - `distill_large_v2-directml`
  - `og_small_v2-tensorrt`
  - `og_base_v2-tensorrt`
  - `og_large_v2-tensorrt`
  - `og_distill_small_v2-tensorrt`
  - `og_distill_base_v2-tensorrt`
  - Benchmark: Encoding + Decoding with `og_small_v2-tensorrt`, fp16, 1080p video, 13700k + 3090:
  Elapsed Time: 0:00:12 • ETA: 0:00:00 • FPS: 119.36 • Mem: 954.6MB • Frames: 1495/1495

#### Changed
- Fully rewrote `parameters.md` to reflect the new changes and to better explain the usage of TAS.
- Better lazy importing of the `ProgressBarLogic` in `main.py` have improved the startup time of TAS by ~100ms.

#### Fixed
- Issues with yt-dlp where it would add extra container extensions to the output file, causing a discrepancy between the expected and actual output file format.

## TAS Adobe Edition
### New
- Added resolution preview in upscale settings dialog - now you can see input dimensions (like 1920×1080) and output dimensions (3840×2160) before processing
- Added FPS preview in interpolation settings dialog - shows current composition FPS and what FPS you will get after interpolation (example: 24.000 FPS → 48.000 FPS for 2x)
- New functions getCompDimensions() and getCompFPS() in After Effects script to get composition information
Viewport zoom slider was added to `ToolBox`
- Added a brand new Graph Editor ( Alpha )
- Added a function to dynamically fetch the available space in C drive and display it in the download tas dependencies dialog.
- Migrated from `yarn/npm` to `bun` for package management, improving build performance and dependency management.

### Changed
- Made big performance improvements to main interface - now it renders much faster and uses less memory
- Converted many functions to use useCallback and useMemo for better React performance
- Moved progress bar display to separate component file for cleaner code organization
- Resolution and FPS previews automatically fetch data when you open the settings dialogs
- Slight refactor to the dependencies dialog to improve readability.
- Removed Motion and fade in/out effects in order to retain snappiness. 
- Migrated from using `exec` to `spawn` for process execution, improving performance and reliability.
- A complete overhaul to `logTab` to make it more performant and user-friendly.
- Updated the `executeProcess` function to handle errors more gracefully and provide better user feedback.

### Fixed
- Fixed issue where executeProcess function was used before it was declared (TypeScript error)
- Loading states now show properly when fetching composition information


## [2.3.2] - 2025-07-19
#### Fixed
- Fixed a misshap in the `requirements.txt` file where `Flask-Cors` was not included, causing issues with CORS handling in the Adobe Edition.

## [2.3.1] - 2025-07-18
#### Added
- Added support for more color formats / pix formats.

#### Changed
- Updated the `aeComms.py` to use a more efficient event stream logic.
- Deprecated 'socketio' in favor of a less headachey approach.

#### Fixed
- Issues such as: `Decoding error: Resize error: Resize error 3074: no path between colorspaces (2/2/2 => 0/2/2). May need to specify additional colorspace parameters` with https://github.com/NevermindNilas/TheAnimeScripter/issues/128


## [2.3.0] - 2025-07-16
#### Added
- N/A

#### Changed
- Minor tweaks to TensorRT Handler.
- Minor tweaks to the progress bar logic.
- Turned `--ae` into a string argument rather than a boolean and it will now require a url to be parsed for more flexibility.
- Added some more logging to `aecomms.py` for better debugging.
- Removed `--realtime` argument and related logic.

#### Fixed
- Fixed a typo when using compile mode and upscale.
- Potential fix for deadlocks when installing dependencies.

## [2.2.9] - 2025-07-07
### Adobe Edition

#### Added 
- Autoscroll to the bottom for the log tab.
- Memoization and caching to the logging logic making it faster and less resource heavy ( probably min maxing but it should be significantly better now ).
- Added Report Issue Button to the Log Tab which copies the logs and opens the Github Issues page.
- Added support for shortened youtube links ( youtu.be )

#### Changed
- Better copy feature for the logs with proper newline handling.
- Added smooth autoscrolling in the logTab.
- Renamed application settings to advanced settings
- Renamed GTX (CUDA) to CUDA models

#### Fixed
- FFMPEG Encoding command becoming red.


## [2.2.9] - 2025-07-04
#### New
- Added new SocketIO based approach to the progress bar, this should result in a more responsive and faster parsing between TAS and AE.
- Switched to @TNTWise's spandrel fork and removed spandrel from the dependencies.
- Added a new option to `build.py` through argparse, `--develop` this is primarily used for development purposes and will not be documented.

#### Changed
- Up to `50%` faster inference speeds with `og_small_v2`, `og_base_v2`, `og_large_v2`, `og_distill_small_v2`, and `og_distill_base_v2` depth models with absolutely 0 quality loss.

### Adobe Edition

#### New
- Added a new `Progress Bar` which is now based on SocketIO and should be more responsive and faster.
- The status bar will now show the current status of TensorRT workflows, this is primarily used for the `engine generation` step for now but will be further expanded in the future.

#### Changed
- Switch animations from a sliding animation to a fade-in animation, also decreased the animation duration to 0.2s.
- Small tweaks to the Progress Bar.

#### Fixed
- Layer sorting was using the workarea start and end points rather than the actual layer start and end points.

#### Known Issues
- ProRes encoding is not currently working.
- AutoCutClip is not currently working.
- DPIR is not currently working.

## [2.2.8] - 2025-06-23
#### Added
- Added `triton-windows` as well as support for `torch.compile()` using command `--compile_mode` with options `["default", "max", "max-graphs"]`. Initial benchmarks show impressive gains for CUDA specific workflows. Tested with a 1080p video on a 13700k with a 3090 and `--benchmark` mode on and a 1 frame warm-up period. Results:
  - **SuperUltraCompact:** 80.80fps → 92.07fps - Mode: Max-Graphs
  - **Rife 4.6:** 140.48fps → 165.51fps - Mode: Max-Graphs
  - **Small Depth Anything V2:** 108.74fps → 131.19fps - Mode: Max-Graphs
  - **GMFSS:** 6.05fps → 9.51fps - Mode: Max
  - **OG Small Depth Anything V2:** 15.72fps → 16.02fps (likely OpenCV bottlenecked) - Mode: Max-Graphs
  - DO NOTE THAT THIS IS STILL UNDER DEVELOPMENT AND IT ONLY WORKS ON RTX GPUs.

#### Changed
- Added precision to the printed tensor shapes when building an engine.
- Updated the bugreport.md

#### Fixed
- Fixed an issue where Adobe Edition would not work due to missmatch in input pix formats.

## [2.2.7] - 2025-06-21
#### Added
- N/A

#### Changed
- Removed some unused debugging code from progressbarLogic.py

#### Fixed
- A mishap on my side when dependencies were being downloaded resulting in unnecessary checks, upgrades and caches. Please run a `.\python.exe -m pip cache purge` if you are upgrading from a previous version.

### Adobe Edition
#### New
- Thanks to the community behind Motion, I was able to add a fully custom animation styling to TAS. Tabs will now `swipe` from the left to right, the progress bar will have a cool little animation both when appearing and disappearing ( and more in the future )
- Added a selective `fp16` or `fp32` precision to the `Extra Tab`, primarily focused on helping users with lower end systems who were running into black frame outputs.

#### Changed
- Removed the preview image primarily because it wasn't really useful for much and it only added extra logic to worry about.
- Small tweaks around `useEffects`.
- Updated some vite-cep dependencies.

#### Fixed
- A potential deadlock when streaming the logs to the logsTab.


## [2.2.5] - 2025-06-20
#### Added
- A whole decoding backend migration to `Vapoursynth` including better I/O Support (WIP!!!)
- Added support for automatic deprecated dependencies removal, see more in `deprecated-requirements.txt` & `src.utils.DependencyHandler`
- Added `VideoTimeStamps` library by @moi15moi for accurate `--inpoint` and `--outpoint` handling.
- New depth methods, `distill_large_v2` and `og_distill_large_v2`

#### Changed
- Switched from yuv420p encoding to rgb24 in order to further minimize CPU usage across the board. This change should imply about a 5% lower CPU usage with minal loss to performance.
- Upgraded TensorRT to `10.12` , this brings a series of goodies with it, I am seeing an up to 10% performance lift in non encode / decode bottlenecked workflows as well as significantly faster engine generation times ( 30s -> 8s for SUC-TensorRT, 30s -> 20s for Rife4.6-TensorRT )
- Upgraded Pytorch to `2.7.1`
- Upgraded Torchvision to `0.22.1`
- More error handling around handle dependencies
- Migrated from `Basswood-AV` -> `Vapoursynth`, this should bring with it faster performance and better I/O handling. I am entertaining the idea of using a full Vapoursynth workflow ( through `vspipe` ) although personal benchmarks show up to `2x` lower performance in upscaling - cuda.
- Removed `explicit batch` from TRT and added `strongly typed`. 
- Upgraded the packaged python from `3.13.3` to `3.13.5`.
- Up to `50%` faster inference speeds with `rife-ncnn`.
- Updated build instructions.
- Updated the `slow_av1` encode-method.
- Optimized the logic behind the progress bar. Adobe updates should now be nearly instantaneous. 
- The progress bar will now also log `fps` and `eta` for AdobeEdition.

#### Fixed
- Fixed issues with GMFSS.
- Fixed issues with Scunet-TensorRT
- Fixed a bug where AMD / Intel GPU users would default to full dependencies esentially downloading more than they actually needed.
- Fixed issues with `--inpoint` and `--outpoint`
- Fixed issues regarding `bt709` & `yuv420p` formats, this further extends to anything compatible with `yuv420p` such as `rgb24` and `bgr24`. Special thanks to [@JET](https://github.com/Jaded-Encoding-Thaumaturgy) community for the help in debugging.

#### Known issues
- `--resize_method` is currently not implemented, due to the new decoding approach.
- I/O support is wonky until I will be able to test more color ranges and inputs, please report any issues.

### Adobe Edition
#### New
- Added dynamic output examples to the `Upscale` and `Depth` Settings.
- Added `distill large` to the depth extraction methods.
- Added a dynamic ETA and FPS to the progress bar.

#### Changed
- Renamed `NVIDIA GTX GPUs` sections to simply `NVIDIA GPUs`
- Added `CUDA` extension wherever approapiate.
- Simplified and memoized some logic.
- The progress bar will now properly update once every 1s cycle ( assuming an update did happen on the backend side )
- Better text formatting for the progress bar.
- TAS will now use `Chain` & `Depth` and `RmvBackground` as the generic keyword for each process output.
- Added logic to use the layer name when pre-rendering if only 1 layer is selected. 

#### Notes: 
- INTERNALIZATION HAS YET TO BE ADDED, non English characters may or may not break TAS' workflow.

#### Fixed
- Potentially fixed issues for AMD/Intel users when downloading the backend.

## [2.2.2] - 2025-05-30
### Adobe Edition
#### Fixed
- Double render issues with precomps
- WorkArea wasn't properly set if the work area was out of boundaries
- No warning if no layer is selected with ' Extract Depth Map ' and ' Remove Background '


## [2.2.1] - 2025-05-28
#### Added
- Restore method `gater3` | `gater3-directml`, special thanks to @umzi2 ( tensorrt and cuda under investigation )
- Added experimental `scunet-tensorrt` backend ( VERY VRAM HUNGRY! )

#### Changed
- Small tweaks to the upscale cuda pipeline and cuda model optimizer.
- Improvements to the restore cuda and trt pipelines.
- Changed the dependencyChecker logic simplifying it.

#### Fixed
- A regression to start-up times caused by overengineering the dependencyHandler.py

### Adobe Edition
#### New
- Added `Gater3 | Gater3DirectML | SCUNet-TensorRT` to the main UI
- Added `Remove Dead Frames` option to ToolBox, this works all on the AE side and is based on SSIM.

#### Changes
- Added a `debouncer` to the settings savings logic. 
- Created a helper function for the `contextualHelper(s)` in order to make the UI more consistent.
- Defaulted to a `8px` gap size basically everywhere. ( more work needed )
- Improved the robustness of the Automated backend download handler as well as improved its logging efficiency.

#### Fixed
- Fixed an issue where the automated download functionality would freeze upon completion locking the user out of TAS ( whoopsies )

## [2.2.0] - 2025-05-26
#### Added
- A fully redesigned and naive build method.
- Added `--download_requirements` file to download the requirements as a helper functionality for Adobe After Effects.
- Added `--cleanup` to uninstall unused dependencies or migrated ones ( e.g. PyAV -> BassWood-AV )
- Added `png` output support with `--encode_method png`

#### Changed
- With the recent changes to Pytorch and TensorRT and the ever increasing wheel sizes, I was forced to adapt a more dynamic approach to dependency handling, TAS will now dynamically download most of the libraries and only ship core needed libraries or very small ones, this lowers initial download sizes from 2.5GB ( compressed ) all the way to +-30MB ( compressed ), obviously this implies a major rework but down the line it will be easier to update TAS since dependencies don't change as often as the underlying code of TAS.
- Reworked `argumentsChecker.py` for better maintainabillity
- Upgraded pytorch to 2.7.0 adding BlackWell support.
- New usage method in the CLI will be `.\python main.py` based.
- Changed the detection method for `autoclip` to `AdaptiveDetector` for the higher performance and accuracy.
- Added a forced break even if `frame` is `None`.
- Switched from `pyav` to `basswood-av`
- Upgraded to TensorRT v10.10
- Logging all currently available torch.cuda devices.

#### Deprecations
- Fully removed `anipy-cli` from TAS.
- Temporarely removed `gputil` until I find a better solution.

#### Fixed
- `shufflecugan-ncnn` defaulting to non-existent `directml` backend when not available.
- Fixed an inversion issue regarding `--autoclip_sens`
- Fixed a cuda race condition when chaining upscale trt models with other cuda / trt workflows.
- Fixed a bug in gmfss
- Fixed an issue where Rife TensorRT would use double the necessary VRAM to generate an engine.
- Fixed an issue where Rife CUDA's timestep generation would overshoot and create artifacts.
- Fixed an issue where Fractional Interpolation wouldn't evenly distribute across the video, special thanks to @HolyWU for the base implementation idea. ( Hopefully I was able to translate it well enough )
- Fixed an issue with yt-dlp where it would not be able to find .mp4 queries.

### Adobe Edition
#### New
- The UI comes in 1 more new flavor with larger scale, specifically for 4k/5k monitors.

#### Changed
- Moved encoder / outputs related settings to `Application Settings` in the `Extra` Tab.
- Simplified the about tab
- Added a visual warning if the Youtube URL is not in the correct format.
- Disclosure panels have been changed to `Accordion` for better spacing and the autoclosing feature.
- Precompose will no longer create alerts and will instead use `Toasts`
- Simplified Toast generation in order to not stack the UI with toasts.
- Updated all react-spectrum dependencies.
- `Enhanced Error Handling:`
  - Added comprehensive try/catch blocks around critical file operations
  - Ensured all errors are properly logged and presented to the user
- `Improved Folder Management:`
  - Added proper parent directory validation and creation
  - More reliable cleanup of legacy TAS folders
  - Better handling of empty installation folders
- `Robust Version Checking:`
  - Better error handling for version detection failures
  - Improved version comparison logic
  - Added specific error messages for version check failures
- `Code Structure Improvements:`
  - Reorganized logical flow for better readability
  - Added more descriptive variable names
  - Used finally block to ensure initialization state is always updated
- `Enhanced Debugging Support:`
  - Added more detailed console logging
  - Better status reporting during dependency checks

#### Fixed
- Few minor bugfixes.

## [2.1.5] - 2025-04-14
### The Anime Scripter
#### Added
- Improved model download progress tracking with percentage-based milestones (20%, 40%, 60%, 80%, 100%) and file size reporting in logs

#### Changed
- Removed `torchmetrics` & replaced all SSIM relevant functions with a custom SSIM.
- A major redesign of the video decoding/encoding pipeline resulting in lower CPU usage, lower memory consumption and better encoding performance across the board.

#### Fixed
- Potential import errors caused by `signal.windows` missing in the bootstrapper of pyinstaller(?).
- fixed `--outpoint`.

#### Known issues
- `--inpoint` doesn't currently work.

### Adobe Edition
#### Added
- N/A

#### Changed
- Largely insignificant tweaks to some margins.
- Removed log overflow generated by CEP on runtime.

#### Fixed
- N/A

## [2.1.3] - 2025-04-08
#### Added
- Test(s)
- Slowmotion support through `--slowmo`
- `constants.py` to more easily maintain the code and to avoid excessive imports and declarations.
- Experimental fractional interpolation support.
- Added `memory usage` to the progress bar for more info.
- Added `"og_small_v2"` / `"og_base_v2"` / `"og_large_v2"` / `"og_distill_small_v2"` /  `"og_distill_base_v2"` which as the name implies are a near original paper implementation of Depth Anything V2 and Distill Depth Anything V2. These are roughly 10-20% faster than the original paper whilst retaining, visually, the same quality.
- Added `anime1080fixer-directml` restore method.

#### Changed
- Upgraded to `python 3.13` including most dependencies.
- Added an automated memdealloc in `buildbuffer`
- Migrated to `onnx-weekly` until `onnx v18.0` gets released.
- Linux support will be indefinitely postponed and will no longer be in any shape or form a priority of TAS.
- Disabled audio when running `--depth`
- Removed `h266`
- Cleaned up the code.
- Some small adjustments to `main.py`
- Increased the performance and accuracy of all dedup methods.
- Lowered CPU usage by `~5%` on my system, this should be more significant for lower end cpus.
- Significant improvements to `--preview` with better performance and lower overhead than before.
- Switched from `celux` to `pyav` due to lack of development.
- Bumped Pytorch to `2.6.0`
- Deprecated: `--resize_method` and `--decode_threads`, in the future these well be readed with a better design, for now use `--custom_encoder`.

#### Known Issues:
- `--inpoint` and `--outpoint` are not functional due to the migration from `celux` to `pyav`, I will need to see how to set interest points based on them.

#### Fixed
- Fixed more issues with non `utf-8` compliant paths. [Thanks Sergio]
- Fixed an issue with `slow_nvenc_av1` [Thanks Sergio]

### Adobe Edition
#### Added
- Added a `copy to clipboard` button to the `logs` tab.
- Added `slowmo` button to the interpolation tab.
- Added `Force destroy all TAS processes` button which kills any lingering tasks in the background, just in case.

#### Changed
- Significant UI refresh across the board.
- "Path Errors" will now result in toasts rather than annoying alerts.
- Removed a couple of unnecessary alerts, there should only now be one or two left for critical pieces. Down the line these will also be Toasts.
- Added a `latest version` indicator the update toast generator.
- Changed the `Interpolation Factor` dropdown to an editable area + dropdown with support for custom values and fractional factors.
- Updated some descriptions and icons.
- Added a `twitter` button to the socials.
- Removed some unnecessary logs

#### Fixed
- N/A 

## [2.1.2] - 2025-03-14
#### Added
- N/A

#### Changed
- More error handlers for easier debugging

#### Fixed
- An issue where utf-8 decoding would break Adobe Edition.



## [2.1.1] - 2025-03-13
#### Added
- 2 New Depth Models, `distill_small_v2` and `distill_base_v2` which provide better quality with better performance.
- `medium` depth quality for better output results for marginal performance loss.

#### Changed
- Small tweaks to depth maps which should generate better outputs overall.

### Fixed
- A bug within the download models importing
- 

## [2.1.0] - 2025-03-12
#### Added
- An automated error handler for when the user wants to use a model which does not work on their system ( WIP ).

### Changed
- Removed unnecessary clamping functionality on decode
- Updated the mpv configs.
- Lowered the base strenght of `fastlinedarken`.
- `psutil` and `gputil` as an temporary alternative to `wmi`.
- Upgraded TensorRT to `10.9.0` adding Nvidia Blackwell ( RTX 5000 ) support.
- Slightly simplified `main.py`
- Added more logging for when the user processes a directory rather than a simple file.
- Slight adjustments to the code to be more resource efficient.
- Minor improvements to initial launch time ( ~150ms )

### Fixed
- Performance regression when encoding
- Fixed some padding issues with `rife4.25-lite`

### Adobe Edition
#### Added
- Added a progress bar during inference.
- Added a `Logs` panel for easier debugging.
- Added a progress circle when TAS is processing.

#### Changed
- Huge chunks of UI generations are now dynamically imported and loaded to make maintaining easier and potentially improve performance.
- Fully rewrote the Toast generation from the ground up to be more maintainable.
- The command prompt will no longer show up when processing.
- Fully redesigned the `About` Tab.
- Many minor UI Tweaks.
- Added an example for segmentation.
- Changed the run button from only having a logo to an actual text `Run Chain` which is also slightly separated from the rest of the UI elements.
- Run Chain will now be disabled during processing.

#### Fixed
- N/a
---

## [2.0.5] - 2025-02-17

### The Anime Scripter
#### Added
- N/A

#### Changed
- Bumped numpy version

#### Fixed
- Fixed a bug in TensorRT shape parsing when resolution is above 1920x1080

### Adobe Edition
#### Added
- N/A

#### Changed
- Added a fallback to lossless when QuickTime isn’t available

#### Fixed
- N/A

---

## [2.0.4] - 2025-02-16

### The Anime Scripter
#### Added
- Updated `rtmosr` models to @umzi2’s "v2" models

#### Changed
- Bumped Python to `3.12.9`
- Improved compression logic for better results:
  - TAS FULL reduced to 1.7GB (from 1.82GB)
  - TAS LITE reduced to 155MB (from 186MB)
  - Overall extracted filesize decreased by ~100MB
- Improved subprocess performance in Python 3.12.9 (approx. 0.25s faster)

#### Removed
- Removed unneeded files and libraries from TAS

#### Fixed
- N/A

### Adobe Edition
#### Added
- Added `lossless` encode method
- Added tip(s)
- Enforced a Pre-Render step to avoid potential bugs
- Render Queue panel now auto-closes after rendering
- Added `Add Null Layer` button
- Added a visual indicator for inference speed
- Added dropdowns in the About Tab for System Requirements and GPU Compatibility

#### Changed
- Moved all Pre-Render options to the Encode Settings dialog
- Simplified backend logic around pre-render
- Renamed `rtmosr` to `rtmosr v2`
- Added an example to Depth Maps (more options coming)

#### Fixed
- Fixed visual anomalies with tooltips

#### Removed
- Removed Pre-Render button(s)

---

## [2.0.3] - 2025-02-03

### The Anime Scripter
#### Added
- N/A

#### Changed
- N/A

#### Fixed
- Fixed an issue where a static step was still parsed even if the current interpolation method didn’t support it

### Adobe Edition
#### Added
- N/A

#### Changed
- N/A

#### Fixed
- N/A

---

## [2.0.2] - 2025-02-02

### The Anime Scripter
#### Added
- Added `ffprobe` for more accurate metadata seeking
- Introduced experimental `--static_step` command to improve inference speed for Rife CUDA (available for models `rife4.6` and `rife4.15-lite`)

#### Changed
- Increased bitrate for `slow_nvenc_h265`
- Bumped celux from `0.5.4` to `0.6.0`
- Refactored parts of the ffmpeg code
- Improved error handling for yt-dlp
- Added a warning when using anipycli

#### Fixed
- Fixed a bug in `TRT Handler.py`
- Fixed issue with SVT AV1 generating unneeded login
- Fixed subtitle issues
- Fixed padding issues with `rife4.25`

#### Removed
- Removed `PyMediaInfo` for accuracy reasons
- Removed the mbps meter from the download progress bar
- Removed the `upscale_skip` and `interpolate_skip` commands

### Adobe Edition
#### Added
- Added `TensorRT Static Engine Generation` button in the upscaling settings

#### Changed
- Improved error handling
- Renamed some UI elements

#### Fixed
- N/A

#### Removed
- Removed `Check For updates` button

---

## [2.0.1] - 2025-01-12

### The Anime Scripter
#### Added
- Introduced `rtmosr-directml` and `rtmosr-tensorrt` upscaling methods by @umzi
- Added `slow_nvenc_av1` encoding preset

#### Changed
- Enhanced depth maps for TensorRT and DirectML
- Renamed log files (`TAS.log` → `TAS-Log.log`; `ffmpeg.log` → `FFmpeg-Log.log`)
- Generalized logging and reduced unnecessary terminal prints from ffmpeg (when using x265)
- Improved realtime playback with MPV preview and switched playback from FFmpeg to MPV for `--realtime`
- Skipped encoding during `--realtime` to save system resources
- Enhanced TRT shape printing and improved ffmpeg command handling
- Added forced exit (`sys.exit()`) if FFMPEG crashes
- Changed subtitle handling from `srt` to `copy` to retain metadata

#### Removed
- Deprecated `--depth_quality` option (models now run internally at 518x518)

#### Fixed
- Fixed shape issues with Segment Anime TensorRT
- Fixed issues with `rawvideo` and `celux`
- Fixed problems in the 16bit workflow with `nvenc_h264`

### Adobe Edition
#### Added
- Added a “Take a Screenshot” feature in the Toolbox
- Introduced a revamped About Panel with additional TAS info
- Added System Requirements in the About panel

#### Changed
- Improved the “Extracting TAS…” indicator
- Made minor backend adjustments for better reusability
- Defaulted depth quality to `low`
- Added error handling for image inputs (TAS works only with video)
- Updated UI elements and descriptions
- Refactored code for clarity

#### Fixed
- Fixed issues with the Remove Background segment not updating correctly
- Fixed repeated unnecessary tasks on UI mount

---

## [2.0.0] - 2024-12-29

### The Anime Scripter
#### Added
- Partially ported TAS to Adobe Premiere (core functionality now supported)
- Added Full CUDA Graph Acceleration for TensorRT and CUDA workflows (up to 25% performance boost)
- Introduced variable `--decode_threads` for more granularity
- Added `anime1080fixer-directml` support for restoration
- Added `slow_nvenc_h264` & `slow_nvenc_h265`
- Introduced new features: `--restore_method`, `fastlinedarken`, and `fastlinedarken-tensorrt`
- Added support for dynamic `upscale_factor` (for CUDA workflows)
- Introduced experimental `--realtime` mode for real-time output preview
- Reintroduced GMFSS support
- Added Dynamic Flow Scale via `--dynamic_scale` (experimental)
- Temporarily added video input metadata collection
- Added Flow-based deduplication with FlownetS (experimental)

#### Changed
- Replaced the existing progress bar with Rich Progress for faster startup and improved FPS accuracy
- Improved inference speed for 16bit workloads (up to 50% faster)
- Enhanced error handling around ffmpeg calls
- Upgraded TensorRT to version 10.7.0
- Refactored the TRT Handler backend for better logging and granularity
- Optimized normalization by removing redundant operations
- Simplified CUDA logs and calls
- Converted the `--half` argument from boolean to string (default: True; options: [True, False])
- Improved segmentation efficiency using CUDA
- Bumped Pytorch from `2.5.0` to `2.5.1`
- Added a torch.fx model optimization backend (up to 20% CUDA inference speed boost)
- Upgraded numpy to version 2.2.1

#### Notes
- Due to the TensorRT upgrade, all engines must be regenerated

#### Removed
- Removed Polygraphy in favor of the built-in TensorRT API
- Removed the `--sample_size` command and pre-compilation of TAS’ bytecode
- Removed image support and Alive-Progress (now using Rich-Progress)

#### Fixed
- Fixed issues with Rife NCNN
- Fixed CUDA race conditions at interpolation factors above 2x
- Fixed audio copying for .webm inputs
- Fixed depth TensorRT flickering in low quality mode
- Fixed black output issues with Upscale NCNN
- Fixed a bug causing the first frame to be black in DirectML workflows
- Fixed Rife Elexor and related import issues
- Fixed misuse of `--encode_method` with .webm containers
- Fixed a bug where TAS would output only a temp file if `--output` was declared without other processing

### Adobe Edition
#### Notes
- This version is built against AE 2025. Some functionalities may not work with older versions.
- Update checking remains manual for now.

#### Added
- Introduced an auto Update Checking system
- Added proper Pre-Comp support to avoid user errors
- Added new interpolation and encoding methods
- Added quick action buttons for “Add Adjustment Layer” and “Add Solid Layer”
- Introduced an “Arrange Layers” function for easier sorting
- Added tooltips for enhanced user guidance
- Added full-path indicators for backend download locations
- Introduced Toast notifications as an alternative to alerts
- Added an “Open Changelogs” button

#### Changed
- Renamed certain UI elements and improved error handling
- Enhanced preview window responsiveness and backend reusability

#### Fixed
- Fixed Remove Background update issues
- Fixed repeated tasks on UI mount

---

## [1.9.11] - 2024-10-26

### The Anime Scripter
#### Added
- Added H266 encoding support via `--encode_method h266`
- Added `slow_x264`, `slow_x265`, and `slow_av1` encoding methods
- Introduced `rife_elexor` and `rife_elexor-tensorrt`
- Renamed `--denoise` to `--restore`
- Added Anime1080Fixer by @Zarxrax to `--restore`
- Added early TRT backend for `anime1080fixer-tensorrt`
- Added support for `yuv420p10le` and `yuv422p8le` inputs
- Enabled anipy support using `--input "anime"`

#### Changed
- Restructured TAS’ backend for improved I/O handling
- Upgraded the GitHub Actions runner (Python 3.12.4 → 3.12.7)
- Optimized memory allocations and defaults for `--upscale_skip`
- Removed redundant calculations in deduplication
- Upgraded Pytorch from 2.4.0 to 2.5.0 and improved input/output handling

#### Fixed
- Fixed variable issues in the 16bit workflow
- Fixed RIFE TRT duplicate frame issues
- Fixed benchmark script issues caused by backend changes

#### Breaking Changes
- All engines must be rebuilt due to the TensorRT upgrade

### Adobe Edition
#### Fixed
- Fixed model selection issues between Full and Lite methods

---

## [1.9.10] - 2024-09-29

### The Anime Scripter
#### Fixed
- Fixed issues where missing `--inpoint` and `--outpoint` caused FFMPEG crashes
- Fixed sharpening values exceeding allowed bounds

---

## [1.9.9] - 2024-09-28

### The Anime Scripter
#### Added
- Enhanced encoding now supports real-time video processing with adequate hardware
- Added Rife 4.25 and Rife 4.25-TensorRT
- Reintroduced VP9 and QSV_VP9 encoding for .webm inputs
- Precompiled `.pyc` files now included for faster startup
- Restructured the interpolation pipeline for TensorRT (up to 25% performance gain)
- Implemented a CUDA pinned memory workflow to boost data transfer efficiency
- Upgraded Span models to Spanimation-V2

#### Changed
- Reduced the Rife TensorRT weights folder size (recommend deleting the folder to regenerate engines)
- Enhanced metadata retrieval using pymediainfo
- Improved benchmarking accuracy and merged audio/subtitle processing during encoding
- Enabled parallel processing between encoding and inference
- Improved FFMPEG logging and overall system checks
- Auto-create output folders if missing

#### Fixed
- Fixed exaggerated performance reports
- Fixed a regression from 1.9.8 impacting encoding performance
- Fixed the first frame being black in upscaling
- Fixed frame processing issues from 1.9.8

### Adobe Edition
#### Added
- Added a Manual Method button for step-by-step TAS backend installation

#### Changed
- Enhanced backend download functionality

---

## [1.9.8] - 2024-09-15

### The Anime Scripter
#### Added
- Extended `--input` to support:
  - `.txt` files with video lists
  - Multiple videos in a single string (separated by semicolons)
- Moved interpolation before upscaling for higher quality and VRAM efficiency
- Added support for multi-subtitle and multi-audio extraction
- Introduced `--preset` and `--preset_list` for saving and reusing configurations
- Improved CTRL+C support for graceful termination

#### Changed
- Enhanced upscale TRT performance (~10% faster)
- Implemented new CUDA detection coroutines for better performance
- Decoupled STDOUT from the main loop (up to 50% performance boost)
- Reduced total model size and improved TensorRT engine robustness
- Rewrote backend chained processes and optimized deduplication (both SSIM-CUDA and CPU)
- Upgraded Torch from 2.3.1 to 2.4.1
- Improved Linux build sizes and encoding efficiency

#### Removed
- Removed the Scikit-Image package

#### Regressions
- Noted a RAM usage increase (~500MB) due to new encoding changes

### The Anime Scripter - Adobe Edition
#### Added
- Enabled Preview Windows using `--preview`
- Added new options for Remove Background with TensorRT acceleration
- Made Pre-Render more flexible with a choice between Lossless (default) and QuickTime

#### Changed
- UI cleanup and minor adjustments

#### Fixed
- Fixed UI issues and minor scaling problems

---

## [1.9.7] - 2024-09-02

### The Anime Scripter
#### Added
- Added Rife4.20-NCNN, 4.21-NCNN, 4.22-NCNN, and 4.22-Lite-NCNN

#### Fixed
- Fixed CUDA errors in the Lite Edition for Windows

#### Regressions
- Some performance gains from 1.9.6 were lost due to dependency handling issues

### Adobe Edition
#### Added
- Added new NCNN Models

#### Changed
- Backend downloads now up to 2300% faster
- Fixed race condition issues and improved path/error handling

#### Removed
- Removed older 4.15-lite & 4.17 models in favor of newer, more performant options

---

## [1.9.6] - 2024-09-02

### The Anime Scripter
#### Added
- Introduced Rife4.22-Lite and Rife4.22-Lite-Tensorrt
- Added `--static` flag to force static engine generation for Upscale TensorRT
- Added temporary state for ensuring file integrity during model downloads
- Introduced `--depth_quality` flag (options: low, high; default is high)
- Provided testing builds for linux-full and linux-lite

#### Changed
- Upgraded base polygraphy from 0.49.12 to 0.49.13
- Removed unnecessary clamps and optimized metadata/progress reporting
- Made log.txt more compact by removing unused arguments
- Ensured Rife TRT builds static engines for improved stability
- Improved Rife CUDA and TRT performance (up to 15% faster)
- Enhanced chained processes for better memory efficiency and benchmarking
- Upgraded ONNXRuntime-DirectML to 1.19.1 and improved download progress reporting

#### Fixed
- Fixed CUDA race conditions in extremely high FPS workflows
- Fixed issues in Rife 4.22 and Rife 4.22-lite that reduced output quality

### Adobe Edition
#### Added
- Added Rife4.22-Lite and Rife4.22-Lite-Tensorrt options to the dropdown menus
- Introduced Depth Quality options (low and high)

#### Fixed
- Fixed path issues when spaces are present

---

## [1.9.5] - 2024-08-17

### The Anime Scripter
- No significant changes

### Adobe Edition
#### Fixed
- Fixed issues with Full vs. Lite method selection during model downloads

---

## [1.9.4] - 2024-08-17

### The Anime Scripter
#### Added
- Introduced a new 3D-like Banner
- Added OpenProteus and AniScale2 by Sirosky
- Launched a new lite version of TAS (WIP)
- Added support for multi-audio and multi-subtitle extraction
- Introduced ShuffleSpan by Sudo (WIP)
- Added Differential and Differential-TensorRT scene change methods (WIP)
- Added Shift_LPIPS-TensorRT scene change methods by Sudo
- Added `--hide_banner` flag to suppress the TAS banner
- Introduced preview functionality via `--preview` (access via 127.0.0.1:5050/frame)

#### Changed
- Improved parameter examples and upgraded to nightly ONNXRuntime DirectML
- Added recursive checks for model downloads
- Enhanced I/O folder handling for better edge-case management

#### Fixed
- Fixed a bug with 16bit workflows and x264_animation_10bit
- Resolved issues with input/output folder case sensitivity
- Fixed concurrency issues with Rife TensorRT and high CPU usage
- Fixed span-directml host misdirection

#### Regressions
- Temporarily reverted to pytorch 2.2.2 and cuda 12.1 due to .dll issues

#### Removed
- Removed Updater.exe

### Adobe Edition
#### Added
- TAS now dynamically switches between “Full” and “Lite” modes based on GPU availability

#### Changed
- Improved error handling and minimized terminal disruption (now starts minimized)
- Enhanced output formatting with keyword highlighting (e.g., _DE for deduplication)
- Improved download functionality with potential for higher speeds

---

## [1.9.3] - 2024-08-09

### The Anime Scripter
#### Added
- Added Rife4.22 TensorRT
- Introduced Video Stabilization using Vidgear

#### Changed
- N/A

#### Removed
- N/A

### Adobe Edition
#### Added
- TBA

#### Changed
- TBA

---

## [1.9.2] - 2024-08-09

### The Anime Scripter
#### Added
- Upgraded to Python 3.12
- Added Rife 4.22 CUDA, Rife 4.21 CUDA and TensorRT, and Rife 4.20 TensorRT
- Introduced `--bit_depth` option (choices: 8bit, 16bit)
- Added experimental .GIF input support via `--encode_method gif`
- Added experimental .png/.jpeg support via `--encode_method image`
- Overhauled model downloading (recommend deleting the weights folder)
- Upgraded SPAN models to 2x_Spanimation V1.5

#### Changed
- Achieved up to 25% faster Rife TensorRT inference
- Improved TensorRT engine robustness and reduced available model count
- Streamlined deduplication and encoding performance

#### Removed
- Removed Rife4.15, Rife4.15-ncnn, and Rife4.15-TensorRT

#### Fixed
- Fixed TensorRT artifacting in Depth Maps
- Resolved inconsistent Depth Map outputs and aliasing issues
- Fixed issues with `--custom_encoder`
- Partially addressed issue [#44](https://github.com/NevermindNilas/TheAnimeScripter/issues/44)
- Fixed pathing issues for Linux

### The Anime Scripter - Adobe Edition
#### Added
- Enabled auto clip functionality
- Introduced half-automated TAS updates
- Added `--bit_depth` to encoding settings

#### Changed
- Various UI improvements

#### Fixed
- Fixed scaling issues and random TAS update popups

---

## [1.9.1] - 2024-07-30

### The Anime Scripter
#### Added
- Added Rife 4.20 CUDA
- Introduced Depth Anything V2 – DirectML
- Added `--scenechange_method` (options: maxvit-directml & maxvit-tensorrt)
- Added ProRes 444 HQ Encoder via `--encode_method prores` and ProRes encoding support

#### Changed
- Improved upscaling performance with TensorRT (up to 15% faster)
- Improved upscaling performance with CUDA (up to 20% faster)
- Enhanced interpolation performance with TensorRT (up to 10% faster) and CUDA (up to 30% faster)
- Decoupled scene change from interpolation with a performance boost (up to 20% faster with DirectML/TensorRT)
- Optimized Rife UHD mode and overall code formatting
- Upgraded Torch from 2.2.2 to 2.4.0 and CUDA from 12.1 to 12.5
- Reformatted help output for clarity

#### Fixed
- Fixed issues [#43](https://github.com/NevermindNilas/TheAnimeScripter/issues/43) and [#41](https://github.com/NevermindNilas/TheAnimeScripter/issues/41)
- Fixed scene change caching and benchmark script issues
- Resolved missing DLL issues with onnxruntime-directml

### Adobe Edition
#### Added
- Introduced a new secure installation method with code signing
- Enabled dynamic downloading and updating of TAS
- Enhanced Pre-Render to support multiple active/selected layers

#### Changed
- Redesigned the UI for a more compact layout (WIP)
- Moved the About page to a Help icon
- Improved tooltips and progress indicators
- Reduced the initial executable size (230kb) with Full and Lite versions available
- Updated dropdown hints and model sectioning
- Added new models and encoders; changed preset template from lossless to high quality

#### Fixed
- Fixed inpoint/outpoint boundary issues
- Fixed YouTube import issues when no composition was selected
- Fixed Pre-Render work area adjustment issues
