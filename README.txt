*****************************************************
*                The Anime Scripter                 *
*****************************************************

Creator: NevermindNilas
Discord: https://discord.gg/GkCrawMuZ6
Github: https://github.com/NevermindNilas/TheAnimeScripter

The script is provided as is, without any warranty. Feel free to join the Discord server and report any issues to Nilas. Learn more about the license in License.md.

Minimum Specs:
- CPU: Any Quad Core CPU
- RAM: 16 GB
- GPU: Any Vulkan Capable GPU with 4GB of VRAM
- Disk: ~6 GB

Recommended Specs:
- CPU: Any Octa Core CPU
- RAM: 32 GB
- GPU: NVidia 1000 series / RX 6000 series
- Disk: ~6 GB

*****************************************************

Installation:
Read more about the installation tutorial or self-building the .exe file here:
https://github.com/NevermindNilas/TheAnimeScripter?tab=readme-ov-file#%EF%B8%8F-getting-started

*****************************************************

Things to consider:
- If you have an NVIDIA GPU of series 1000 or higher, you can utilize virtually anything within the script due to CUDA capable accelerations.
- For AMD / Intel GPU users, please use the NCNN Versions of the upscalers / interpolators to avoid falling back to CPU and ensure optimal performance.
- GMFSS may run out of VRAM for high-resolution clips; use it with 8 GB or more VRAM for 1080p footage.
- Rife Ensemble trigger internal calls to the warp algorithm, providing better results with some loss in performance. Enable it if the performance loss is acceptable.
- Resize multiplier supports fractional values like 0.5 for downscaling the clip (3840x2160 -> 1920x1080).
- All upscalers, except CUGAN, have 2x models (1920x1080 -> 3840x2160). CUGAN can go up to 4x but expect significant slowdowns depending on the clip's resolution.
- Depth Map and Segmentation require CUDA-capable GPUs for acceleration; otherwise, the script will fallback to CPU with potential performance issues.

*****************************************************

